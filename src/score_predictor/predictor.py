from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import yaml

from .adjustments import apply_multiplicative_adjustments
from .audit import build_audit
from .data_quality import compute_data_quality
from .ensemble import blend_lambdas
from .intelligence.adjustments_from_intel import build_intelligence_adjustments
from .intelligence.prematch_context import (
    is_prematch_context_payload,
    prematch_context_to_intelligence,
)
from .intelligence.schemas import IntelligenceInput
from .market_implied import infer_lambdas_from_market, probs_from_lambdas
from .odds import fair_1x2_probs, fair_over_under_probs
from .poisson import score_matrix
from .report import summarize_prediction
from .schemas import MatchInput
from .v3.ensemble import build_v3_prediction

DISCLAIMER = "Only for entertainment and probability modeling; not betting advice."


def _normalize_over_under_item(
    value: Any,
    default_line: float | str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    line = value.get("line", default_line)
    over_odds = value.get("over_odds", value.get("over"))
    under_odds = value.get("under_odds", value.get("under"))
    if line is None or over_odds is None or under_odds is None:
        return None
    return {
        "line": float(line),
        "over_odds": float(over_odds),
        "under_odds": float(under_odds),
    }


def _normalize_over_under_markets(
    primary: Any,
    extra: Any,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    primary_market = _normalize_over_under_item(primary)
    markets: list[dict[str, Any]] = []

    if isinstance(extra, list):
        for item in extra:
            market = _normalize_over_under_item(item)
            if market is not None:
                markets.append(market)
    elif isinstance(extra, dict):
        single_market = _normalize_over_under_item(extra)
        if single_market is not None:
            markets.append(single_market)
        else:
            for line, item in extra.items():
                market = _normalize_over_under_item(item, default_line=line)
                if market is not None:
                    markets.append(market)

    if primary_market is None and markets:
        primary_market = next(
            (market for market in markets if market["line"] == 2.5),
            markets[0],
        )

    deduped: dict[str, dict[str, Any]] = {}
    for market in markets:
        deduped[f"{market['line']:g}"] = market
    return primary_market, list(deduped.values())


def _normalize_btts(*sources: dict[str, Any]) -> dict[str, float] | None:
    for source in sources:
        btts = source.get("btts")
        if isinstance(btts, dict):
            yes = btts.get("yes", btts.get("yes_odds"))
            no = btts.get("no", btts.get("no_odds"))
            if yes is None:
                yes = btts.get(True)
            if no is None:
                no = btts.get(False)
            if yes is not None and no is not None:
                return {"yes": float(yes), "no": float(no)}
        yes = source.get("btts_yes_odds")
        no = source.get("btts_no_odds")
        if yes is not None and no is not None:
            return {"yes": float(yes), "no": float(no)}
    return None


def _normalize_rqspf(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    handicap = value.get("handicap")
    home = value.get("home", value.get("home_odds"))
    draw = value.get("draw", value.get("draw_odds"))
    away = value.get("away", value.get("away_odds"))
    if handicap is None or home is None or draw is None or away is None:
        return None
    return {
        "handicap": float(handicap),
        "home": float(home),
        "draw": float(draw),
        "away": float(away),
    }


def _normalize_odds_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, float] = {}
    for outcome, odds in value.items():
        if odds is None:
            continue
        normalized[str(outcome)] = float(odds)
    return normalized


def _normalize_1x2(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    home = value.get("home", value.get("home_odds"))
    draw = value.get("draw", value.get("draw_odds"))
    away = value.get("away", value.get("away_odds"))
    if home is None or draw is None or away is None:
        return None
    return {"home": float(home), "draw": float(draw), "away": float(away)}


def _score_key(value: Any) -> str | None:
    match = re.fullmatch(r"(\d{1,2})\s*[-:]\s*(\d{1,2})", str(value).strip())
    if not match:
        return None
    return f"{int(match.group(1))}-{int(match.group(2))}"


def _normalize_correct_score_market(value: Any) -> tuple[dict[str, float], dict[str, float]]:
    if not isinstance(value, dict):
        return {}, {}
    source = value.get("scores") if isinstance(value.get("scores"), dict) else value
    scores: dict[str, float] = {}
    other: dict[str, float] = {}
    for outcome, odds in source.items():
        if odds is None:
            continue
        score = _score_key(outcome)
        if score is not None:
            scores[score] = float(odds)
        elif str(outcome) in {"home_other", "draw_other", "away_other"}:
            other[str(outcome)] = float(odds)
    return scores, other


def _normalize_total_goals_market(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    odds = value.get("odds") if isinstance(value.get("odds"), dict) else value
    normalized: dict[str, float] = {}
    for outcome, price in odds.items():
        key = str(outcome).strip()
        if key == "7":
            key = "7+"
        if key in {"0", "1", "2", "3", "4", "5", "6", "7+"} and price is not None:
            normalized[key] = float(price)
    return normalized


def _normalize_half_full_market(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    odds = value.get("odds") if isinstance(value.get("odds"), dict) else value
    return _normalize_odds_mapping(odds)


def _normalize_sporttery_handicap_3way(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    handicap = value.get("handicap", value.get("line"))
    home = value.get("home", value.get("home_win", value.get("win")))
    draw = value.get("draw", value.get("draw_odds"))
    away = value.get("away", value.get("away_win", value.get("loss")))
    if handicap is None or home is None or draw is None or away is None:
        return None
    return {
        "handicap": float(handicap),
        "home": float(home),
        "draw": float(draw),
        "away": float(away),
    }


def _normalize_odds_channels(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "international": {
                "role": "primary_calibration",
                "source": "the_odds_api",
                "provider": "pinnacle",
                "weight": 1.0,
            },
            "sporttery": {
                "role": "supplemental_calibration",
                "source": "yaml",
                "provider": "sporttery",
                "weight": 0.35,
            },
        }
    return value


def _normalize_market_roles(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "calibration_sources": [],
            "value_comparison_sources": ["sporttery"],
            "roles_configured": False,
        }
    return {
        "calibration_sources": value.get("calibration_sources", []),
        "value_comparison_sources": value.get(
            "value_comparison_sources",
            ["sporttery"],
        ),
        "roles_configured": True,
    }


def _nested_market(data: dict[str, Any], role: str) -> dict[str, Any]:
    markets = data.get("markets")
    if not isinstance(markets, dict):
        return {}
    market = markets.get(role)
    return dict(market) if isinstance(market, dict) else {}


def _international_market(data: dict[str, Any]) -> dict[str, Any]:
    return _nested_market(data, "international") or _nested_market(data, "calibration")


def _sporttery_market(data: dict[str, Any]) -> dict[str, Any]:
    return _nested_market(data, "sporttery") or _nested_market(data, "value_comparison")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _normalize_intelligence_payload(data: dict[str, Any]) -> Any:
    prematch_context = data.get("prematch_context")
    if isinstance(prematch_context, dict) and is_prematch_context_payload(prematch_context):
        return prematch_context_to_intelligence(prematch_context)
    if is_prematch_context_payload(data):
        return prematch_context_to_intelligence(data)
    return data.get("intelligence")


def _estimate_internal_lambdas_from_market(
    odds_1x2: dict[str, Any] | None,
    over_under: dict[str, Any] | None,
) -> tuple[float, float, str | None]:
    if not isinstance(odds_1x2, dict):
        return 1.2, 1.0, "market_only_mode_used_default_internal_lambda"
    try:
        fair_1x2 = fair_1x2_probs(
            float(odds_1x2["home"]),
            float(odds_1x2["draw"]),
            float(odds_1x2["away"]),
        )
        over_probability = None
        over_under_line = 2.5
        if isinstance(over_under, dict):
            over_under_line = float(over_under.get("line", over_under_line))
            if over_under.get("over_odds") is not None and over_under.get("under_odds") is not None:
                over_under_fair = fair_over_under_probs(
                    float(over_under["over_odds"]),
                    float(over_under["under_odds"]),
                )
                over_probability = over_under_fair["over"]
        home, away = infer_lambdas_from_market(
            fair_1x2,
            over_probability=over_probability,
            over_under_line=over_under_line,
        )
        return home, away, None
    except (KeyError, TypeError, ValueError):
        return 1.2, 1.0, "market_only_mode_used_default_internal_lambda"


def _fill_market_only_internal_model(normalized: dict[str, Any]) -> dict[str, Any]:
    settings = normalized.get("settings") or {}
    if not isinstance(settings, dict) or not _truthy(settings.get("market_only_mode")):
        return normalized

    internal = normalized.get("internal_model") or {}
    if not isinstance(internal, dict):
        internal = {}
    home = internal.get("home_lambda", internal.get("internal_lambda_home"))
    away = internal.get("away_lambda", internal.get("internal_lambda_away"))
    if home is not None and away is not None:
        return normalized

    market_home, market_away, warning = _estimate_internal_lambdas_from_market(
        normalized.get("odds_1x2"),
        normalized.get("over_under"),
    )
    normalized["internal_model"] = {
        **internal,
        "home_lambda": market_home,
        "away_lambda": market_away,
    }
    notes = list(normalized.get("notes") or [])
    notes.append("Internal lambda was not provided. Market-only mode was used.")
    normalized["notes"] = list(dict.fromkeys(notes))
    if warning:
        warnings = list(normalized.get("warnings") or [])
        warnings.append(warning)
        normalized["warnings"] = list(dict.fromkeys(warnings))
    return normalized


def _normalize_input_data(data: dict[str, Any]) -> dict[str, Any]:
    if "match" not in data or not isinstance(data.get("match"), dict):
        normalized = dict(data)
        international_market = _international_market(data)
        sporttery_market = _sporttery_market(data)
        normalized["calibration_market"] = international_market
        normalized["value_comparison_market"] = sporttery_market
        normalized["international_market"] = international_market
        normalized["sporttery_market"] = sporttery_market
        primary_ou, ou_markets = _normalize_over_under_markets(
            data.get("over_under"),
            data.get("over_under_markets", data.get("over_under_odds")),
        )
        if primary_ou is not None:
            normalized["over_under"] = primary_ou
        if ou_markets:
            normalized["over_under_markets"] = ou_markets
        btts = _normalize_btts(data)
        if btts is not None:
            normalized["btts"] = btts
        correct_score_odds, correct_score_other = _normalize_correct_score_market(
            data.get("correct_score_odds", data.get("correct_score"))
        )
        if correct_score_odds:
            normalized["correct_score_odds"] = correct_score_odds
        sporttery_correct_score, sporttery_other = _normalize_correct_score_market(
            data.get("sporttery_correct_score_odds")
            or data.get("sporttery_correct_score")
            or sporttery_market.get("sporttery_correct_score")
            or sporttery_market.get("correct_score_odds")
            or sporttery_market.get("correct_score")
        )
        if sporttery_correct_score:
            normalized["sporttery_correct_score_odds"] = sporttery_correct_score
        other = dict(correct_score_other)
        other.update(sporttery_other)
        if other:
            normalized["correct_score_other_odds"] = other
        rqspf = _normalize_rqspf(data.get("rqspf"))
        if rqspf is not None:
            normalized["rqspf"] = rqspf
        sporttery_1x2 = _normalize_1x2(
            data.get("sporttery_1x2")
            or (data.get("sporttery") or {}).get("spf")
            or sporttery_market.get("sporttery_1x2")
            or sporttery_market.get("odds_1x2")
            or sporttery_market.get("spf")
        )
        if sporttery_1x2 is not None:
            normalized["sporttery_1x2"] = sporttery_1x2
        sporttery_handicap = _normalize_sporttery_handicap_3way(
            data.get("sporttery_handicap_3way")
            or (data.get("sporttery") or {}).get("rqspf")
            or sporttery_market.get("sporttery_handicap_3way")
            or sporttery_market.get("rqspf")
            or sporttery_market.get("sporttery_handicap")
        )
        if sporttery_handicap is not None:
            normalized["sporttery_handicap_3way"] = sporttery_handicap
        normalized["market_roles"] = _normalize_market_roles(data.get("market_roles"))
        normalized["odds_channels"] = _normalize_odds_channels(data.get("odds_channels"))
        normalized["odds_movement_settings"] = data.get("odds_movement_settings") or {}
        normalized["market_snapshots"] = list(data.get("market_snapshots") or [])
        normalized["intelligence"] = _normalize_intelligence_payload(data)
        normalized["sporttery_total_goals_odds"] = _normalize_total_goals_market(
            data.get("sporttery_total_goals_odds")
            or data.get("sporttery_total_goals")
            or (data.get("sporttery") or {}).get("total_goals")
            or (data.get("sporttery") or {}).get("total_goals_odds")
            or sporttery_market.get("sporttery_total_goals_odds")
            or sporttery_market.get("sporttery_total_goals")
            or sporttery_market.get("total_goals")
        )
        normalized["half_full_time_odds"] = _normalize_half_full_market(
            data.get("half_full_time_odds")
            or data.get("sporttery_half_full")
            or (data.get("sporttery") or {}).get("half_full_time")
            or (data.get("sporttery") or {}).get("half_full_time_odds")
            or sporttery_market.get("sporttery_half_full")
            or sporttery_market.get("half_full_time")
            or sporttery_market.get("half_full_time_odds")
        )
        return _fill_market_only_internal_model(normalized)

    match = data["match"]
    market = data.get("market", {})
    sporttery = data.get("sporttery", {})
    international_market = _international_market(data)
    sporttery_market = _sporttery_market(data)
    calibration_market = international_market
    value_comparison_market = sporttery_market
    internal = data.get("internal_model", {})
    venue = match.get("venue", {}) or {}
    primary_ou, ou_markets = _normalize_over_under_markets(
        market.get("over_under", data.get("over_under")),
        market.get(
            "over_under_markets",
            market.get("over_under_odds", data.get("over_under_odds")),
        ),
    )
    btts = _normalize_btts(market, data)
    correct_score_odds, correct_score_other = _normalize_correct_score_market(
        international_market.get("correct_score_odds")
        or international_market.get("correct_score")
        or market.get("correct_score_odds")
        or market.get("correct_score")
        or data.get("correct_score_odds")
        or data.get("correct_score")
    )
    sporttery_correct_score_odds, sporttery_other = _normalize_correct_score_market(
        sporttery_market.get("sporttery_correct_score")
        or sporttery_market.get("correct_score_odds")
        or sporttery_market.get("correct_score")
        or sporttery.get("correct_score")
        or sporttery.get("correct_score_odds")
        or data.get("sporttery_correct_score")
        or data.get("sporttery_correct_score_odds")
    )
    correct_score_other.update(sporttery_other)
    rqspf = _normalize_rqspf(
        market.get("rqspf")
        or market.get("sporttery_handicap")
        or data.get("rqspf")
        or sporttery.get("rqspf")
    )
    sporttery_1x2 = _normalize_1x2(
        sporttery_market.get("sporttery_1x2")
        or sporttery_market.get("odds_1x2")
        or sporttery_market.get("spf")
        or sporttery.get("spf")
        or data.get("sporttery_1x2")
    )
    sporttery_handicap_3way = _normalize_sporttery_handicap_3way(
        sporttery_market.get("sporttery_handicap_3way")
        or sporttery_market.get("rqspf")
        or sporttery_market.get("sporttery_handicap")
        or sporttery.get("rqspf")
        or data.get("sporttery_handicap_3way")
    )
    sporttery_total_goals_odds = _normalize_total_goals_market(
        sporttery_market.get("sporttery_total_goals")
        or sporttery_market.get("sporttery_total_goals_odds")
        or sporttery_market.get("total_goals")
        or sporttery.get("total_goals")
        or sporttery.get("total_goals_odds")
        or data.get("sporttery_total_goals")
        or data.get("sporttery_total_goals_odds")
    )
    half_full_time_odds = _normalize_half_full_market(
        sporttery_market.get("sporttery_half_full")
        or sporttery_market.get("half_full_time")
        or sporttery_market.get("half_full_time_odds")
        or sporttery.get("half_full_time")
        or sporttery.get("half_full_time_odds")
        or data.get("sporttery_half_full")
        or data.get("half_full_time_odds")
    )

    normalized = {
        "match": f"{match.get('home_team', 'Home')} vs {match.get('away_team', 'Away')}",
        "kickoff_time": (
            match.get("commence_time_utc")
            or match.get("kickoff_time_beijing")
            or match.get("commence_time")
            or match.get("kickoff_time")
            or match.get("date")
        ),
        "timezone": match.get("timezone", "Asia/Shanghai"),
        "target": match.get("target", "90min_score"),
        "venue_type": venue.get("venue_type", match.get("venue_type", "unknown")),
        "prediction_time": data.get("prediction_time", "pre_match"),
        "odds_1x2": (
            international_market.get("odds_1x2")
            or market.get("odds_1x2")
            or market.get("spf")
            or sporttery_1x2
            or sporttery.get("spf")
        ),
        "over_under": primary_ou,
        "over_under_markets": ou_markets,
        "btts": btts,
        "correct_score_odds": correct_score_odds or {},
        "correct_score_other_odds": correct_score_other or {},
        "sporttery_1x2": sporttery_1x2,
        "sporttery_handicap_3way": sporttery_handicap_3way,
        "sporttery_correct_score_odds": sporttery_correct_score_odds or {},
        "sporttery_total_goals_odds": sporttery_total_goals_odds,
        "half_full_time_odds": half_full_time_odds,
        "asian_handicap": market.get("asian_handicap"),
        "rqspf": rqspf,
        "calibration_market": calibration_market,
        "value_comparison_market": value_comparison_market,
        "international_market": international_market,
        "sporttery_market": sporttery_market,
        "market_snapshots": list(data.get("market_snapshots") or []),
        "market_roles": _normalize_market_roles(data.get("market_roles")),
        "odds_channels": _normalize_odds_channels(data.get("odds_channels")),
        "odds_movement_settings": data.get("odds_movement_settings") or {},
        "internal_model": {
            "home_lambda": internal.get(
                "home_lambda", internal.get("internal_lambda_home")
            ),
            "away_lambda": internal.get(
                "away_lambda", internal.get("internal_lambda_away")
            ),
        },
        "settings": {
            "market_weight": internal.get(
                "market_weight", data.get("settings", {}).get("market_weight", 0.65)
            ),
            "max_goals": data.get("settings", {}).get("max_goals", 7),
            "score_model": data.get("settings", {}).get("score_model", "poisson"),
            "dc_enabled": data.get("settings", {}).get("dc_enabled", False),
            "market_only_mode": data.get("settings", {}).get("market_only_mode", False),
            "h2h_weight": data.get("settings", {}).get("h2h_weight", 1.0),
            "x1x2_weight": data.get("settings", {}).get("x1x2_weight", 1.0),
            "totals_weight": data.get("settings", {}).get("totals_weight", 1.0),
            "ou_weight": data.get("settings", {}).get("ou_weight", 1.0),
            "alternate_totals_weight": data.get("settings", {}).get("alternate_totals_weight", 0.8),
            "btts_weight": data.get("settings", {}).get("btts_weight", 0.6),
            "spreads_weight": data.get("settings", {}).get("spreads_weight", 0.5),
            "correct_score_weight": data.get("settings", {}).get(
                "correct_score_weight", 0.35
            ),
            "sporttery_1x2_weight": data.get("settings", {}).get(
                "sporttery_1x2_weight", 0.15
            ),
            "sporttery_handicap_3way_weight": data.get("settings", {}).get(
                "sporttery_handicap_3way_weight", 0.15
            ),
            "sporttery_total_goals_weight": data.get("settings", {}).get(
                "sporttery_total_goals_weight", 0.30
            ),
            "sporttery_correct_score_weight": data.get("settings", {}).get(
                "sporttery_correct_score_weight", 0.20
            ),
            "sporttery_half_full_weight": data.get("settings", {}).get(
                "sporttery_half_full_weight", 0.0
            ),
            "team_adjustment_strength": data.get("settings", {}).get(
                "team_adjustment_strength", 1.0
            ),
        },
        "adjustments": data.get("adjustments", {}),
        "notes": data.get("notes", []),
        "warnings": data.get("warnings", []),
        "intelligence": _normalize_intelligence_payload(data),
    }
    return _fill_market_only_internal_model(normalized)


def match_input_from_dict(data: dict[str, Any]) -> MatchInput:
    return MatchInput(**_normalize_input_data(data))


def load_match_input(path: str | Path) -> MatchInput:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Match input YAML must contain a mapping/object.")
    return match_input_from_dict(data)


def _confidence(
    match_input: MatchInput,
    warnings: list[str],
    top_score_prob: float,
    data_quality: dict[str, Any] | None = None,
    intel_adjustments: dict[str, Any] | None = None,
    intel: IntelligenceInput | None = None,
) -> str:
    score = 0
    if match_input.odds_1x2:
        score += 2
    if match_input.over_under:
        score += 1
    if match_input.asian_handicap:
        score += 1
    if match_input.internal_model:
        score += 1
    if warnings:
        score -= 1
    if top_score_prob >= 0.14:
        score += 1
    if data_quality:
        if data_quality["level"] == "high":
            score += 2
        elif data_quality["level"] == "medium":
            score += 1
        else:
            score -= 1
    if intel_adjustments:
        narrative = intel_adjustments["diagnostics"]["narrative_heat"]
        if narrative["level"] == "high":
            score -= 1

    if score >= 7:
        confidence = "high"
    elif score >= 5:
        confidence = "medium"
    elif score >= 3:
        confidence = "low"
    else:
        confidence = "very_low"

    if intel is not None:
        if not intel.official_lineups_available and confidence == "high":
            confidence = "medium"
        if intel.match_type in {"friendly", "club_friendly"} and not intel.official_lineups_available:
            confidence = "low" if confidence in {"high", "medium"} else confidence

    return confidence


def _legacy_confidence(match_input: MatchInput, warnings: list[str], top_score_prob: float) -> str:
    score = 0
    if match_input.odds_1x2:
        score += 2
    if match_input.over_under:
        score += 1
    if match_input.asian_handicap:
        score += 1
    if match_input.internal_model:
        score += 1
    if warnings:
        score -= 1
    if top_score_prob >= 0.14:
        score += 1

    if score >= 5:
        return "medium"
    if score >= 3:
        return "medium-low"
    return "low"


def _clamp_lambda(value: float) -> float:
    return max(0.05, min(5.0, value))


def _main_drivers(
    fair_1x2: dict[str, float],
    final_home: float,
    final_away: float,
    over_under_line: float,
    over_probability: float | None,
    adjustments_applied: bool,
) -> list[str]:
    drivers: list[str] = []
    strongest_market = max(fair_1x2, key=fair_1x2.get)
    drivers.append(f"market_{strongest_market}_lean")

    if final_home > final_away * 1.25:
        drivers.append("final_lambda_home_edge")
    elif final_away > final_home * 1.25:
        drivers.append("final_lambda_away_edge")
    else:
        drivers.append("balanced_final_lambdas")

    if over_probability is not None:
        if over_under_line <= 2.25 or over_probability < 0.48:
            drivers.append("market_low_total_goals")
        elif over_under_line >= 2.75 or over_probability > 0.52:
            drivers.append("market_high_total_goals")
        else:
            drivers.append("market_neutral_total_goals")

    if adjustments_applied:
        drivers.append("pre_match_lambda_adjustments")

    return drivers


def predict(
    match_input: MatchInput,
    intelligence: IntelligenceInput | None = None,
    dc_enabled: bool | None = None,
) -> dict[str, Any]:
    intel = intelligence if intelligence is not None else match_input.intelligence
    fair_1x2 = fair_1x2_probs(
        match_input.odds_1x2.home,
        match_input.odds_1x2.draw,
        match_input.odds_1x2.away,
    )

    over_under_line = 2.5
    over_under_fair: dict[str, float] | None = None
    over_probability: float | None = None
    if match_input.over_under:
        over_under_line = match_input.over_under.line
        over_under_fair = fair_over_under_probs(
            match_input.over_under.over_odds,
            match_input.over_under.under_odds,
        )
        over_probability = over_under_fair["over"]

    market_home, market_away = infer_lambdas_from_market(
        fair_1x2,
        over_probability=over_probability,
        over_under_line=over_under_line,
    )

    intel_adjustments = build_intelligence_adjustments(
        intel,
        base_market_weight=match_input.settings.market_weight,
    )

    internal_home_for_blend = (
        market_home
        if match_input.settings.market_only_mode
        else match_input.internal_model.home_lambda
    )
    internal_away_for_blend = (
        market_away
        if match_input.settings.market_only_mode
        else match_input.internal_model.away_lambda
    )
    blended_home, blended_away = blend_lambdas(
        market_home,
        market_away,
        internal_home_for_blend,
        internal_away_for_blend,
        market_weight=intel_adjustments["market_weight"],
    )

    blended_home *= intel_adjustments["total_lambda_factor"]
    blended_away *= intel_adjustments["total_lambda_factor"]
    blended_home *= intel_adjustments["home_lambda_factor"]
    blended_away *= intel_adjustments["away_lambda_factor"]

    final_home, final_away = apply_multiplicative_adjustments(
        blended_home,
        blended_away,
        home_factors=match_input.adjustments.home_factors,
        away_factors=match_input.adjustments.away_factors,
    )
    final_home = _clamp_lambda(final_home)
    final_away = _clamp_lambda(final_away)

    score_df = score_matrix(
        final_home,
        final_away,
        max_goals=match_input.settings.max_goals,
    )
    summary = summarize_prediction(score_df, over_under_line=over_under_line, top_n=5)

    v3_dc_enabled = match_input.settings.dc_enabled if dc_enabled is None else dc_enabled
    audit = build_audit(intel)
    v3_result = build_v3_prediction(
        match_input,
        market_initial_lambda=(market_home, market_away),
        intel_adjustments=intel_adjustments,
        data_quality=compute_data_quality(match_input, intel),
        audit=audit,
        dc_enabled=v3_dc_enabled,
    )

    warnings = list(match_input.warnings)
    if match_input.asian_handicap:
        warnings.append("asian_handicap_recorded_not_used_in_v0")
    if not match_input.over_under:
        warnings.append("over_under_missing_market_total_less_reliable")
    if v3_dc_enabled:
        warnings.append("dixon_coles_enabled_in_v3_section")
    else:
        warnings.append("dixon_coles_not_enabled_poisson_only")
    warnings.extend(intel_adjustments["warnings"])

    top_prob = float(summary["max_probability_score"]["prob"])
    data_quality = compute_data_quality(match_input, intel)
    confidence = _legacy_confidence(match_input, warnings, top_prob)
    if intel is not None:
        confidence = _confidence(
            match_input,
            warnings,
            top_prob,
            data_quality=data_quality,
            intel_adjustments=intel_adjustments,
            intel=intel,
        )
    adjustments_applied = any(
        factor != 1.0
        for factor in (
            match_input.adjustments.home_factors + match_input.adjustments.away_factors
        )
    ) or any(
        intel_adjustments[key] != 1.0
        for key in ["home_lambda_factor", "away_lambda_factor", "total_lambda_factor"]
    )
    main_drivers = _main_drivers(
        fair_1x2,
        final_home,
        final_away,
        over_under_line,
        over_probability,
        adjustments_applied,
    )
    main_drivers.extend(intel_adjustments["drivers"])
    main_drivers = list(dict.fromkeys(main_drivers))

    return {
        "match": match_input.match,
        "kickoff_time": match_input.kickoff_time,
        "timezone": match_input.timezone,
        "target": match_input.target,
        "prediction_time": match_input.prediction_time,
        "venue_type": match_input.venue_type,
        "disclaimer": DISCLAIMER,
        "market": {
            "fair_1x2": fair_1x2,
            "fair_over_under": over_under_fair,
            "lambda": {
                "home": market_home,
                "away": market_away,
                "total": market_home + market_away,
            },
            "fitted_probabilities": probs_from_lambdas(
                market_home,
                market_away,
                over_under_line=over_under_line,
            ),
        },
        "internal_lambda": {
            "home": match_input.internal_model.home_lambda,
            "away": match_input.internal_model.away_lambda,
            "total": (
                match_input.internal_model.home_lambda
                + match_input.internal_model.away_lambda
            ),
        },
        "blend": {
            "market_weight": intel_adjustments["market_weight"],
            "base_market_weight": match_input.settings.market_weight,
            "internal_weight": 1.0 - intel_adjustments["market_weight"],
            "blended_lambda_before_adjustments": {
                "home": blended_home,
                "away": blended_away,
                "total": blended_home + blended_away,
            },
        },
        "adjustments": {
            "home_factors": match_input.adjustments.home_factors,
            "away_factors": match_input.adjustments.away_factors,
            "reasons": match_input.adjustments.reasons,
        },
        "intelligence_adjustments": {
            "home_lambda_factor": intel_adjustments["home_lambda_factor"],
            "away_lambda_factor": intel_adjustments["away_lambda_factor"],
            "total_lambda_factor": intel_adjustments["total_lambda_factor"],
        },
        "intelligence": intel_adjustments["diagnostics"],
        "data_quality": data_quality,
        "audit": audit,
        "final_lambda": {
            "home": final_home,
            "away": final_away,
            "total": final_home + final_away,
        },
        "over_under_line": over_under_line,
        "probabilities": summary["probabilities"],
        "top_scores": summary["top_scores"],
        "max_probability_score": summary["max_probability_score"],
        "confidence": confidence,
        "main_drivers": main_drivers,
        "warnings": warnings,
        "v3": v3_result,
        "notes": match_input.notes,
    }
