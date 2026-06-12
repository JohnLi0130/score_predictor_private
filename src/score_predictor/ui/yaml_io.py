from __future__ import annotations

from copy import deepcopy
from typing import Any

import yaml

from score_predictor.market_implied import infer_lambdas_from_market
from score_predictor.odds import fair_1x2_probs, fair_over_under_probs
from score_predictor.predictor import match_input_from_dict
from score_predictor.schemas import MatchInput

from .form_helpers import (
    COMMON_CORRECT_SCORES,
    audit_only_score_warning,
    copy_default_form_state,
    optional_float,
    optional_int,
    rows_from_table,
    split_text_items,
)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _line_key(line: float | str) -> str:
    return f"{float(line):g}"


def dump_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def load_yaml_payload(content: bytes | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(content, dict):
        data = content
    else:
        text = content.decode("utf-8") if isinstance(content, bytes) else content
        data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML must contain a mapping/object.")
    return data


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _payload_markets(payload: dict[str, Any] | None) -> dict[str, Any]:
    return _as_mapping((payload or {}).get("markets"))


def _market_source(market: dict[str, Any]) -> str:
    source = market.get("source") or market.get("odds_source")
    if not source and isinstance(market.get("odds_1x2"), dict):
        source = market["odds_1x2"].get("source")
    return str(source or "").strip()


def _extract_calibration_market(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    markets = _payload_markets(payload)
    international = _as_mapping(markets.get("international"))
    if international:
        return deepcopy(international)
    calibration = _as_mapping(markets.get("calibration"))
    if calibration:
        return deepcopy(calibration)
    return deepcopy(_as_mapping(payload.get("market")))


def _looks_like_sporttery_market(market: dict[str, Any]) -> bool:
    source = _market_source(market).casefold()
    return (
        "sporttery" in source
        or "体彩" in source
        or "sporttery_total_goals" in market
        or "total_goals_odds" in market
        or "half_full_time" in market
        or "half_full_time_odds" in market
    )


def _extract_value_comparison_market(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    markets = _payload_markets(payload)
    sporttery = _as_mapping(markets.get("sporttery"))
    if sporttery:
        return deepcopy(sporttery)
    value_market = _as_mapping(markets.get("value_comparison"))
    if value_market:
        return deepcopy(value_market)
    market = _as_mapping(payload.get("market"))
    if market and _looks_like_sporttery_market(market):
        return deepcopy(market)
    return {}


def _legacy_market_from_calibration(
    calibration: dict[str, Any],
    a_source_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    legacy = deepcopy(_as_mapping((a_source_payload or {}).get("market")))
    if legacy:
        return legacy

    odds_1x2 = _as_mapping(calibration.get("odds_1x2"))
    over_under = calibration.get("over_under") or calibration.get("over_under_markets")
    asian_handicap = calibration.get("asian_handicap")
    legacy = {
        "odds_source": calibration.get("source") or "calibration",
        "odds_1x2": {
            key: odds_1x2[key]
            for key in ("home", "draw", "away")
            if key in odds_1x2
        },
    }
    if over_under:
        legacy["over_under"] = deepcopy(over_under)
        legacy["over_under_markets"] = deepcopy(over_under)
        if isinstance(over_under, list):
            legacy["over_under_odds"] = {
                _line_key(row["line"]): {
                    "over_odds": row.get("over_odds", row.get("over")),
                    "under_odds": row.get("under_odds", row.get("under")),
                }
                for row in over_under
                if isinstance(row, dict) and row.get("line") is not None
            }
    if isinstance(asian_handicap, list) and asian_handicap:
        legacy["asian_handicap"] = deepcopy(asian_handicap[0])
    elif asian_handicap:
        legacy["asian_handicap"] = deepcopy(asian_handicap)
    return legacy


def _source_list(payload: dict[str, Any] | None, key: str) -> list[str]:
    roles = _as_mapping((payload or {}).get("market_roles"))
    values = roles.get(key) or []
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values if str(value).strip()]


def _dedupe_sources(*groups: list[str]) -> list[str]:
    sources: list[str] = []
    for group in groups:
        for source in group:
            normalized = str(source).strip().lower()
            if normalized and normalized not in sources:
                sources.append(normalized)
    return sources


def merge_prediction_payload(
    base_payload: dict[str, Any] | None,
    a_source_payload: dict[str, Any] | None,
    b_source_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge UI/base input with international and Sporttery odds channels."""
    base = deepcopy(base_payload or {})
    a_payload = deepcopy(a_source_payload or {})
    b_payload = deepcopy(b_source_payload or {})
    merged = deepcopy(a_payload or base)

    if not merged.get("match"):
        merged["match"] = deepcopy(base.get("match") or b_payload.get("match") or {})
    if not merged.get("prediction_time"):
        merged["prediction_time"] = (
            base.get("prediction_time")
            or b_payload.get("prediction_time")
            or b_payload.get("snapshot_time")
            or "pre_match"
        )

    calibration = _extract_calibration_market(a_payload or merged)
    value_comparison = _extract_value_comparison_market(b_payload) or _extract_value_comparison_market(base)

    markets = {}
    if calibration:
        markets["international"] = calibration
    if value_comparison:
        markets["sporttery"] = value_comparison
    if markets:
        merged["markets"] = markets

    if calibration:
        merged["market"] = _legacy_market_from_calibration(calibration, a_payload or merged)

    calibration_sources = _dedupe_sources(
        _source_list(a_payload, "calibration_sources"),
        _source_list(base, "calibration_sources"),
        ["international", "the_odds_api"] if a_payload else [],
    )
    value_sources = _dedupe_sources(
        _source_list(b_payload, "value_comparison_sources"),
        _source_list(base, "value_comparison_sources"),
        ["sporttery"] if value_comparison else [],
    )
    merged["odds_channels"] = {
        "international": {
            "role": "primary_calibration",
            "source": calibration.get("source", "the_odds_api") if calibration else "manual",
            "provider": (
                (calibration.get("provider") or {}).get("bookmaker")
                if isinstance(calibration.get("provider"), dict)
                else calibration.get("provider", "pinnacle")
            ) if calibration else "manual",
            "weight": 1.0,
        },
        "sporttery": {
            "role": "supplemental_calibration",
            "source": "yaml",
            "provider": "sporttery",
            "weight": 0.35,
        },
    }
    merged["market_roles"] = {
        "calibration_sources": calibration_sources,
        "value_comparison_sources": value_sources or ["sporttery"],
    }

    return merged


def load_yaml_to_form_state(content: bytes | str | dict[str, Any]) -> dict[str, Any]:
    return form_state_from_yaml_dict(load_yaml_payload(content))


def form_state_from_yaml_dict(data: dict[str, Any]) -> dict[str, Any]:
    state = copy_default_form_state()
    match = data.get("match") if isinstance(data.get("match"), dict) else {}
    markets = data.get("markets") if isinstance(data.get("markets"), dict) else {}
    value_market = (
        markets.get("sporttery")
        if isinstance(markets.get("sporttery"), dict)
        else markets.get("value_comparison")
        if isinstance(markets.get("value_comparison"), dict)
        else {}
    )
    international_market = (
        markets.get("international")
        if isinstance(markets.get("international"), dict)
        else markets.get("calibration")
        if isinstance(markets.get("calibration"), dict)
        else {}
    )
    market = data.get("market") if isinstance(data.get("market"), dict) else {}
    if international_market:
        market = international_market
    if not market and value_market:
        market = value_market
    settings = data.get("settings") if isinstance(data.get("settings"), dict) else {}
    market_roles = data.get("market_roles") if isinstance(data.get("market_roles"), dict) else {}
    internal = data.get("internal_model") if isinstance(data.get("internal_model"), dict) else {}
    team_context = data.get("team_context") if isinstance(data.get("team_context"), dict) else {}

    if match:
        state.update(
            {
                "match_id": match.get("match_id", state["match_id"]),
                "date": _first_present(
                    match.get("date"),
                    match.get("kickoff_time"),
                    state["date"],
                ),
                "home_team": match.get("home_team", state["home_team"]),
                "away_team": match.get("away_team", state["away_team"]),
                "competition": match.get("competition", state["competition"]),
                "stage": match.get("stage", state["stage"]),
                "neutral_site": bool(
                    match.get("neutral_site")
                    or (match.get("venue") or {}).get("venue_type") == "neutral"
                ),
                "timezone": match.get("timezone", state["timezone"]),
            }
        )
    elif isinstance(data.get("match"), str):
        parts = str(data["match"]).split(" vs ", 1)
        if len(parts) == 2:
            state["home_team"], state["away_team"] = parts
        state["date"] = data.get("kickoff_time", state["date"])
        state["timezone"] = data.get("timezone", state["timezone"])

    state["snapshot_time"] = data.get("prediction_time", state["snapshot_time"])
    state["odds_source"] = data.get("odds_source", market.get("odds_source", state["odds_source"]))

    odds_1x2 = market.get("odds_1x2") or data.get("odds_1x2") or {}
    if odds_1x2:
        state["odds_home_win"] = odds_1x2.get("home", state["odds_home_win"])
        state["odds_draw"] = odds_1x2.get("draw", state["odds_draw"])
        state["odds_away_win"] = odds_1x2.get("away", state["odds_away_win"])

    ou_source = (
        market.get("over_under_odds")
        or market.get("over_under_markets")
        or market.get("over_under")
        or market.get("alternate_totals")
        or data.get("over_under_markets")
        or data.get("over_under_odds")
    )
    ou_rows: list[dict[str, Any]] = []
    if isinstance(ou_source, dict):
        for line, row in ou_source.items():
            if isinstance(row, dict):
                ou_rows.append(
                    {
                        "line": optional_float(row.get("line", line)),
                        "over_odds": optional_float(row.get("over_odds", row.get("over"))),
                        "under_odds": optional_float(row.get("under_odds", row.get("under"))),
                    }
                )
    elif isinstance(ou_source, list):
        for row in ou_source:
            if isinstance(row, dict):
                ou_rows.append(
                    {
                        "line": optional_float(row.get("line")),
                        "over_odds": optional_float(row.get("over_odds", row.get("over"))),
                        "under_odds": optional_float(row.get("under_odds", row.get("under"))),
                    }
                )
    if not ou_rows and isinstance(data.get("over_under"), dict):
        row = data["over_under"]
        ou_rows.append(
            {
                "line": optional_float(row.get("line")),
                "over_odds": optional_float(row.get("over_odds", row.get("over"))),
                "under_odds": optional_float(row.get("under_odds", row.get("under"))),
            }
        )
    if ou_rows:
        state["ou_rows"] = ou_rows

    btts = market.get("btts") or data.get("btts") or {}
    if btts:
        state["btts_yes_odds"] = btts.get("yes", btts.get("yes_odds", state["btts_yes_odds"]))
        state["btts_no_odds"] = btts.get("no", btts.get("no_odds", state["btts_no_odds"]))

    correct_score = (
        market.get("correct_score_odds")
        or market.get("correct_score")
        or data.get("correct_score_odds")
        or {}
    )
    if isinstance(correct_score, dict) and isinstance(correct_score.get("scores"), dict):
        correct_score = correct_score["scores"]
    if isinstance(correct_score, dict):
        rows = [{"score": score, "odds": correct_score.get(score)} for score in COMMON_CORRECT_SCORES]
        for score, odds in correct_score.items():
            if score not in COMMON_CORRECT_SCORES:
                rows.append({"score": score, "odds": odds})
        state["correct_score_rows"] = rows

    other = market.get("correct_score_other") or data.get("correct_score_other") or {}
    state["home_other"] = other.get("home_other", state["home_other"])
    state["draw_other"] = other.get("draw_other", state["draw_other"])
    state["away_other"] = other.get("away_other", state["away_other"])

    asian = market.get("asian_handicap") or data.get("asian_handicap") or {}
    if isinstance(asian, list):
        asian = asian[0] if asian else {}
    if asian:
        state["asian_handicap_line"] = asian.get("line")
        state["asian_handicap_home_odds"] = asian.get("home_odds")
        state["asian_handicap_away_odds"] = asian.get("away_odds")

    rqspf = market.get("rqspf") or data.get("rqspf") or {}
    if rqspf:
        state["rqspf_handicap"] = rqspf.get("handicap")
        state["rqspf_home_odds"] = rqspf.get("home", rqspf.get("home_odds", rqspf.get("win")))
        state["rqspf_draw_odds"] = rqspf.get("draw", rqspf.get("draw_odds"))
        state["rqspf_away_odds"] = rqspf.get("away", rqspf.get("away_odds", rqspf.get("loss")))

    state.update({key: team_context.get(key, state[key]) for key in team_context if key in state})
    state["internal_lambda_home"] = _first_present(
        internal.get("home_lambda"),
        internal.get("internal_lambda_home"),
        state["internal_lambda_home"],
    )
    state["internal_lambda_away"] = _first_present(
        internal.get("away_lambda"),
        internal.get("internal_lambda_away"),
        state["internal_lambda_away"],
    )
    state["market_weight"] = _first_present(
        internal.get("market_weight"),
        settings.get("market_weight"),
        state["market_weight"],
    )
    for key in (
        "dc_enabled",
        "max_goals",
        "market_only_mode",
        "h2h_weight",
        "totals_weight",
        "alternate_totals_weight",
        "correct_score_weight",
        "btts_weight",
        "spreads_weight",
        "ou_weight",
        "x1x2_weight",
        "sporttery_1x2_weight",
        "sporttery_handicap_3way_weight",
        "sporttery_total_goals_weight",
        "sporttery_correct_score_weight",
        "sporttery_half_full_weight",
        "team_adjustment_strength",
    ):
        if key in settings:
            state[key] = settings[key]
    if market_roles:
        state["calibration_sources"] = "\n".join(
            str(source) for source in market_roles.get("calibration_sources", [])
        )
        state["value_comparison_sources"] = "\n".join(
            str(source) for source in market_roles.get("value_comparison_sources", [])
        )
    if not state["internal_lambda_home"] or not state["internal_lambda_away"]:
        state["market_only_mode"] = True
    return state


def _valid_decimal_odds(value: Any) -> float | None:
    number = optional_float(value)
    if number is None or number <= 1.0:
        return None
    return number


def _build_over_under(rows: Any) -> dict[str, dict[str, float]]:
    markets: dict[str, dict[str, float]] = {}
    for row in rows_from_table(rows):
        line = optional_float(row.get("line"))
        over = _valid_decimal_odds(row.get("over_odds"))
        under = _valid_decimal_odds(row.get("under_odds"))
        if line is None or line <= 0 or over is None or under is None:
            continue
        markets[_line_key(line)] = {"over_odds": over, "under_odds": under}
    return markets


def _build_correct_score(rows: Any) -> dict[str, float]:
    scores: dict[str, float] = {}
    for row in rows_from_table(rows):
        score = str(row.get("score") or "").strip()
        odds = _valid_decimal_odds(row.get("odds"))
        if odds is None or "-" not in score:
            continue
        scores[score] = odds
    return scores


def _build_intelligence(state: dict[str, Any]) -> dict[str, Any]:
    home_absent = split_text_items(state.get("home_key_players_missing"))
    away_absent = split_text_items(state.get("away_key_players_missing"))
    sources = []
    if str(state.get("source_notes") or "").strip():
        sources.append({"note": str(state["source_notes"]).strip()})
    return {
        "source_mode": "manual",
        "official_squads_available": False,
        "official_lineups_available": False,
        "match_type": "world_cup",
        "match_importance": {
            "home": str(state.get("motivation_notes") or ""),
            "away": str(state.get("motivation_notes") or ""),
        },
        "injuries_suspensions": {
            "home": {"absent": home_absent, "doubtful": [], "suspended": []},
            "away": {"absent": away_absent, "doubtful": [], "suspended": []},
        },
        "rest_days": {
            "home": optional_int(state.get("home_rest_days")),
            "away": optional_int(state.get("away_rest_days")),
        },
        "tactical_notes": {
            "weather_note": str(state.get("weather_note") or ""),
            "injury_notes": str(state.get("injury_notes") or ""),
            "lineup_notes": str(state.get("lineup_notes") or ""),
            "schedule_notes": str(state.get("schedule_notes") or ""),
        },
        "sources": sources,
    }


def _estimate_market_lambdas(payload: dict[str, Any]) -> tuple[float, float]:
    market = payload.get("market") or {}
    odds = market.get("odds_1x2") or {}
    fair = fair_1x2_probs(float(odds["home"]), float(odds["draw"]), float(odds["away"]))
    over_probability = None
    over_under_line = 2.5
    ou_markets = market.get("over_under_odds") or {}
    if ou_markets:
        selected_line = "2.5" if "2.5" in ou_markets else next(iter(ou_markets))
        selected = ou_markets[selected_line]
        over_under_line = float(selected_line)
        fair_ou = fair_over_under_probs(
            float(selected["over_odds"]),
            float(selected["under_odds"]),
        )
        over_probability = fair_ou["over"]
    return infer_lambdas_from_market(
        fair,
        over_probability=over_probability,
        over_under_line=over_under_line,
    )


def build_yaml_from_form_state(state: dict[str, Any]) -> dict[str, Any]:
    neutral_site = bool(state.get("neutral_site"))
    market: dict[str, Any] = {
        "odds_source": state.get("odds_source") or "manual",
        "odds_1x2": {
            "home": float(state["odds_home_win"]),
            "draw": float(state["odds_draw"]),
            "away": float(state["odds_away_win"]),
        },
    }
    over_under = _build_over_under(state.get("ou_rows"))
    if over_under:
        market["over_under_odds"] = over_under

    btts_yes = _valid_decimal_odds(state.get("btts_yes_odds"))
    btts_no = _valid_decimal_odds(state.get("btts_no_odds"))
    if btts_yes is not None and btts_no is not None:
        market["btts"] = {"yes": btts_yes, "no": btts_no}

    correct_score = _build_correct_score(state.get("correct_score_rows"))
    if correct_score:
        market["correct_score_odds"] = correct_score

    other = {
        "home_other": optional_float(state.get("home_other")),
        "draw_other": optional_float(state.get("draw_other")),
        "away_other": optional_float(state.get("away_other")),
    }
    if any(value is not None for value in other.values()):
        market["correct_score_other"] = other

    asian = {
        "line": optional_float(state.get("asian_handicap_line")),
        "home_odds": _valid_decimal_odds(state.get("asian_handicap_home_odds")),
        "away_odds": _valid_decimal_odds(state.get("asian_handicap_away_odds")),
    }
    if all(value is not None for value in asian.values()):
        market["asian_handicap"] = asian

    rqspf = {
        "handicap": optional_float(state.get("rqspf_handicap")),
        "home": _valid_decimal_odds(state.get("rqspf_home_odds")),
        "draw": _valid_decimal_odds(state.get("rqspf_draw_odds")),
        "away": _valid_decimal_odds(state.get("rqspf_away_odds")),
    }
    if all(value is not None for value in rqspf.values()):
        market["rqspf"] = rqspf

    sporttery_market: dict[str, Any] = {
        "source": "yaml",
        "provider": "sporttery",
        "weight": 0.35,
        "sporttery_1x2": deepcopy(market["odds_1x2"]),
    }
    if all(value is not None for value in rqspf.values()):
        sporttery_market["sporttery_handicap_3way"] = deepcopy(rqspf)
    if correct_score:
        sporttery_market["sporttery_correct_score"] = {"scores": deepcopy(correct_score)}
    if any(value is not None for value in other.values()):
        sporttery_market.setdefault("sporttery_correct_score", {"scores": {}})
        sporttery_market["sporttery_correct_score"]["scores"].update(
            {key: value for key, value in other.items() if value is not None}
        )
    if _build_over_under(state.get("sporttery_total_goals_rows")):
        sporttery_market["sporttery_total_goals"] = {
            "odds": _build_over_under(state.get("sporttery_total_goals_rows"))
        }

    payload: dict[str, Any] = {
        "match": {
            "match_id": state.get("match_id") or "",
            "date": state.get("date") or "",
            "home_team": state.get("home_team") or "Home",
            "away_team": state.get("away_team") or "Away",
            "competition": state.get("competition") or "",
            "stage": state.get("stage") or "",
            "kickoff_time": state.get("date") or "",
            "timezone": state.get("timezone") or "Asia/Shanghai",
            "venue": {"venue_type": "neutral" if neutral_site else "home"},
            "neutral_site": neutral_site,
            "target": "90min_score",
        },
        "prediction_time": state.get("snapshot_time") or "pre_match",
        "odds_channels": {
            "international": {
                "role": "primary_calibration",
                "source": state.get("odds_source") or "manual",
                "provider": state.get("odds_source") or "manual",
                "weight": 1.0,
            },
            "sporttery": {
                "role": "supplemental_calibration",
                "source": "yaml",
                "provider": "sporttery",
                "weight": 0.35,
            },
        },
        "markets": {
            "international": {
                "source": state.get("odds_source") or "manual",
                "provider": state.get("odds_source") or "manual",
                "weight": 1.0,
                **deepcopy(market),
            },
            "sporttery": sporttery_market,
        },
        "market": market,
        "market_roles": {
            "calibration_sources": split_text_items(state.get("calibration_sources")),
            "value_comparison_sources": split_text_items(
                state.get("value_comparison_sources")
            )
            or ["sporttery"],
        },
        "settings": {
            "max_goals": int(state.get("max_goals") or 8),
            "dc_enabled": bool(state.get("dc_enabled")),
            "market_only_mode": bool(state.get("market_only_mode")),
            "h2h_weight": float(state.get("h2h_weight") or 1.0),
            "totals_weight": float(state.get("totals_weight") or 1.0),
            "alternate_totals_weight": float(state.get("alternate_totals_weight") or 0.8),
            "market_weight": float(state.get("market_weight") or 1.0),
            "correct_score_weight": float(state.get("correct_score_weight") or 0.35),
            "btts_weight": float(state.get("btts_weight") or 0.6),
            "spreads_weight": float(state.get("spreads_weight") or 0.5),
            "ou_weight": float(state.get("ou_weight") or 1.0),
            "x1x2_weight": float(state.get("x1x2_weight") or 1.0),
            "sporttery_1x2_weight": float(state.get("sporttery_1x2_weight") or 0.15),
            "sporttery_handicap_3way_weight": float(
                state.get("sporttery_handicap_3way_weight") or 0.15
            ),
            "sporttery_total_goals_weight": float(
                state.get("sporttery_total_goals_weight") or 0.30
            ),
            "sporttery_correct_score_weight": float(
                state.get("sporttery_correct_score_weight") or 0.20
            ),
            "sporttery_half_full_weight": 0.0,
            "team_adjustment_strength": float(
                state.get("team_adjustment_strength") or 1.0
            ),
        },
        "team_context": {
            key: state.get(key)
            for key in (
                "home_elo",
                "away_elo",
                "home_fifa_rank",
                "away_fifa_rank",
                "home_rest_days",
                "away_rest_days",
                "home_key_players_missing",
                "away_key_players_missing",
                "home_lineup_strength",
                "away_lineup_strength",
                "weather_note",
                "injury_notes",
                "lineup_notes",
                "schedule_notes",
                "motivation_notes",
                "source_notes",
            )
        },
        "intelligence": _build_intelligence(state),
        "adjustments": {
            "home_factors": [1.0],
            "away_factors": [1.0],
            "reasons": ["No manual pre-match lambda adjustment in UI input."],
        },
        "notes": ["Synthetic UI input; not a betting recommendation."],
    }

    market_only = bool(state.get("market_only_mode"))
    if market_only:
        try:
            home_lambda, away_lambda = _estimate_market_lambdas(payload)
        except (KeyError, TypeError, ValueError):
            home_lambda, away_lambda = 1.2, 1.0
            payload["notes"].append("Market-only mode used fallback internal lambda.")
        payload["notes"].append("Internal lambda was not provided. Market-only mode was used.")
    else:
        home_lambda = optional_float(state.get("internal_lambda_home"))
        away_lambda = optional_float(state.get("internal_lambda_away"))
        if home_lambda is None or away_lambda is None:
            raise ValueError("internal_lambda_home and internal_lambda_away are required.")

    payload["internal_model"] = {
        "internal_lambda_home": float(home_lambda),
        "internal_lambda_away": float(away_lambda),
        "market_weight": float(state.get("market_weight") or 1.0),
    }
    warnings = audit_only_score_warning(state)
    if warnings:
        payload["warnings"] = warnings
    return payload


def match_input_from_form_state(state: dict[str, Any]) -> MatchInput:
    return match_input_from_dict(build_yaml_from_form_state(state))


def download_yaml_button(
    st_module: Any,
    payload: dict[str, Any],
    file_name: str = "match.yaml",
    key: str | None = None,
) -> None:
    st_module.download_button(
        "下载输入 YAML",
        data=dump_yaml(payload),
        file_name=file_name,
        mime="text/yaml",
        key=key,
    )
