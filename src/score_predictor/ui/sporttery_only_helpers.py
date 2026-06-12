from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any

from score_predictor.history.store import save_prediction_history
from score_predictor.predictor import match_input_from_dict, predict


SPORTTERY_MODEL_WEIGHTS: dict[str, float] = {
    "sporttery_1x2_weight": 1.00,
    "sporttery_total_goals_weight": 0.85,
    "sporttery_correct_score_weight": 0.35,
    "sporttery_handicap_3way_weight": 0.45,
    "sporttery_half_full_weight": 0.00,
}

ODDS_MOVEMENT_SETTINGS: dict[str, Any] = {
    "enabled": True,
    "affect_confidence": True,
    "affect_market_quality": True,
    "affect_lambda": True,
    "late_window_hours": 6,
    "max_lambda_adjustment": 0.05,
    "max_total_lambda_adjustment": 0.06,
    "max_rho_adjustment": 0.025,
    "movement_weights": {
        "sporttery_1x2_movement": 0.12,
        "sporttery_handicap_3way_movement": 0.10,
        "sporttery_correct_score_movement": 0.08,
        "sporttery_total_goals_movement": 0.12,
        "sporttery_half_full_movement": 0.00,
    },
}

DISCLAIMER_ZH = "本系统仅用于概率建模、数据分析和赛后复盘，不构成投注建议。"
TOTAL_GOALS_BUCKETS = ("0", "1", "2", "3", "4", "5", "6", "7+")
HALF_FULL_KEYS = {"HH", "HD", "HA", "DH", "DD", "DA", "AH", "AD", "AA"}
SKIPPED_MARKET_KEYS = {
    "history",
    "notes",
    "warnings",
    "source",
    "provider",
    "snapshot_time",
    "published_at",
    "timestamp",
    "status",
    "role",
    "weight",
    "odds_channels",
    "markets",
    "metadata",
    "data_sources",
}


class SportteryPayloadError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        field_path: str | None = None,
        expected: str | None = None,
        actual: Any = None,
        checked_paths: list[str] | None = None,
        top_level_keys: list[str] | None = None,
    ) -> None:
        details = [message]
        if field_path:
            details.append(f"字段路径：{field_path}")
        if expected:
            details.append(f"期望：{expected}")
        if actual is not None:
            details.append(f"实际：{type(actual).__name__}")
        if checked_paths:
            details.append(f"已检查路径：{', '.join(checked_paths)}")
        if top_level_keys:
            details.append(f"顶层 keys：{', '.join(top_level_keys)}")
        super().__init__("\n".join(details))
        self.field_path = field_path
        self.expected = expected
        self.actual = actual
        self.checked_paths = checked_paths or []
        self.top_level_keys = top_level_keys or []


def _as_mapping(value: Any) -> dict[str, Any]:
    return deepcopy(value) if isinstance(value, dict) else {}


def _copy_meta(source: dict[str, Any], *, source_path: str) -> dict[str, Any]:
    result = {"source_path": source_path}
    history = source.get("history")
    if isinstance(history, list):
        result["history"] = deepcopy(history)
    for key in ("snapshot_time", "published_at", "timestamp"):
        if source.get(key) is not None:
            result[key] = source[key]
    return result


def _get_path(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _first_path(payload: dict[str, Any], paths: list[str]) -> tuple[str | None, Any]:
    for path in paths:
        value = _get_path(payload, path)
        if value is not None:
            return path, value
    return None, None


def _odds_container(value: Any) -> dict[str, Any]:
    market = _as_mapping(value)
    if isinstance(market.get("odds"), dict):
        return _as_mapping(market.get("odds"))
    if isinstance(market.get("scores"), dict):
        return _as_mapping(market.get("scores"))
    return market


def _required_decimal(value: Any, field_path: str) -> float:
    if isinstance(value, (list, dict)):
        raise SportteryPayloadError(
            "解析体彩 YAML 时发现字段类型错误。",
            field_path=field_path,
            expected="数字",
            actual=value,
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SportteryPayloadError(
            "解析体彩 YAML 时发现字段类型错误。",
            field_path=field_path,
            expected="数字",
            actual=value,
        ) from exc
    if number <= 1.0:
        raise SportteryPayloadError(
            "解析体彩 YAML 时发现无效赔率。",
            field_path=field_path,
            expected="大于 1 的数字",
            actual=value,
        )
    return number


def _optional_decimal(value: Any, field_path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        raise SportteryPayloadError(
            "解析体彩 YAML 时发现字段类型错误。",
            field_path=field_path,
            expected="数字",
            actual=value,
        )
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 1.0 else None


def _lookup_outcome(source: dict[str, Any], *names: str) -> tuple[str | None, Any]:
    for name in names:
        if name in source:
            return name, source[name]
    return None, None


def _first_mapping(*values: Any) -> dict[str, Any]:
    for value in values:
        mapping = _as_mapping(value)
        if mapping:
            return mapping
    return {}


def _valid_odds(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 1.0 else None


def _with_history(normalized: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    history = source.get("history")
    if isinstance(history, list) and history:
        normalized["history"] = deepcopy(history)
    for key in ("snapshot_time", "published_at", "timestamp"):
        if source.get(key) is not None:
            normalized[key] = source[key]
    return normalized


def _normalize_1x2(value: Any) -> dict[str, Any]:
    market = _as_mapping(value)
    home = _valid_odds(market.get("home", market.get("home_odds")))
    draw = _valid_odds(market.get("draw", market.get("draw_odds")))
    away = _valid_odds(market.get("away", market.get("away_odds")))
    if home is None or draw is None or away is None:
        return {}
    return _with_history({"home": home, "draw": draw, "away": away}, market)


def _normalize_handicap_3way(value: Any) -> dict[str, Any]:
    market = _as_mapping(value)
    line = market.get("line", market.get("handicap"))
    try:
        handicap = float(line)
    except (TypeError, ValueError):
        return {}
    home = _valid_odds(market.get("home", market.get("home_win", market.get("win"))))
    draw = _valid_odds(market.get("draw", market.get("draw_odds")))
    away = _valid_odds(market.get("away", market.get("away_win", market.get("loss"))))
    if home is None or draw is None or away is None:
        return {}
    return _with_history(
        {"line": handicap, "home": home, "draw": draw, "away": away},
        market,
    )


def _score_key(value: Any) -> str | None:
    match = re.fullmatch(r"(\d{1,2})\s*[-:]\s*(\d{1,2})", str(value).strip())
    if not match:
        return None
    return f"{int(match.group(1))}-{int(match.group(2))}"


def _normalize_correct_score(value: Any) -> dict[str, Any]:
    market = _as_mapping(value)
    source = _as_mapping(market.get("scores")) or _as_mapping(market)
    scores: dict[str, float] = {}
    for outcome, odds in source.items():
        price = _valid_odds(odds)
        if price is None:
            continue
        score = _score_key(outcome)
        if score is not None:
            scores[score] = price
        elif str(outcome) in {"home_other", "draw_other", "away_other"}:
            scores[str(outcome)] = price
    if not scores:
        return {}
    return _with_history({"scores": scores}, market)


def _normalize_total_goals(value: Any) -> dict[str, Any]:
    market = _as_mapping(value)
    source = _as_mapping(market.get("odds")) or _as_mapping(market)
    odds: dict[str, float] = {}
    for outcome, price in source.items():
        key = str(outcome).strip()
        if key == "7":
            key = "7+"
        number = _valid_odds(price)
        if key in TOTAL_GOALS_BUCKETS and number is not None:
            odds[key] = number
    if not odds:
        return {}
    return _with_history({"odds": odds}, market)


def _normalize_half_full(value: Any) -> dict[str, Any]:
    market = _as_mapping(value)
    source = _as_mapping(market.get("odds")) or _as_mapping(market)
    odds: dict[str, float] = {}
    for outcome, price in source.items():
        key = str(outcome).strip().upper()
        number = _valid_odds(price)
        if key in HALF_FULL_KEYS and number is not None:
            odds[key] = number
    if not odds:
        return {}
    return _with_history(odds, market)


SPORTTERY_1X2_PATHS = [
    "odds_1x2",
    "sporttery_1x2",
    "sporttery_1x2.odds",
    "market.odds_1x2",
    "market.sporttery_1x2",
    "market.sporttery_1x2.odds",
    "market.spf",
    "markets.sporttery.sporttery_1x2",
    "markets.sporttery.sporttery_1x2.odds",
    "markets.sporttery.odds_1x2",
    "markets.sporttery.spf",
    "markets.calibration.odds_1x2",
    "markets.calibration.sporttery_1x2",
    "markets.calibration.sporttery_1x2.odds",
]

SPORTTERY_HANDICAP_PATHS = [
    "sporttery_handicap_3way",
    "handicap_3way",
    "rqspf",
    "market.sporttery_handicap_3way",
    "market.handicap_3way",
    "market.rqspf",
    "markets.sporttery.sporttery_handicap_3way",
    "markets.sporttery.handicap_3way",
    "markets.sporttery.rqspf",
    "markets.calibration.sporttery_handicap_3way",
    "markets.calibration.handicap_3way",
    "markets.calibration.rqspf",
]

SPORTTERY_CORRECT_SCORE_PATHS = [
    "sporttery_correct_score",
    "sporttery_correct_score_odds",
    "correct_score_odds",
    "correct_score",
    "market.sporttery_correct_score",
    "market.sporttery_correct_score_odds",
    "market.correct_score_odds",
    "market.correct_score",
    "markets.sporttery.sporttery_correct_score",
    "markets.sporttery.sporttery_correct_score_odds",
    "markets.sporttery.correct_score_odds",
    "markets.sporttery.correct_score",
    "markets.calibration.sporttery_correct_score",
    "markets.calibration.correct_score_odds",
]

SPORTTERY_TOTAL_GOALS_PATHS = [
    "sporttery_total_goals",
    "sporttery_total_goals_odds",
    "total_goals",
    "market.sporttery_total_goals",
    "market.sporttery_total_goals_odds",
    "market.total_goals",
    "markets.sporttery.sporttery_total_goals",
    "markets.sporttery.sporttery_total_goals_odds",
    "markets.sporttery.total_goals",
    "markets.calibration.sporttery_total_goals",
    "markets.calibration.total_goals",
]

SPORTTERY_HALF_FULL_PATHS = [
    "sporttery_half_full",
    "half_full_time",
    "half_full_time_odds",
    "market.sporttery_half_full",
    "market.half_full_time",
    "market.half_full_time_odds",
    "markets.sporttery.sporttery_half_full",
    "markets.sporttery.half_full_time",
    "markets.sporttery.half_full_time_odds",
    "markets.calibration.sporttery_half_full",
    "markets.calibration.half_full_time",
]


def extract_sporttery_1x2(payload: dict[str, Any]) -> dict[str, Any] | None:
    path, value = _first_path(payload, SPORTTERY_1X2_PATHS)
    if path is None:
        return None
    market = _as_mapping(value)
    source = _odds_container(value)
    home_key, home_value = _lookup_outcome(source, "home", "home_odds", "win")
    draw_key, draw_value = _lookup_outcome(source, "draw", "draw_odds")
    away_key, away_value = _lookup_outcome(source, "away", "away_odds", "loss")
    if home_key is None or draw_key is None or away_key is None:
        return None
    return {
        "odds": {
            "home": _required_decimal(home_value, f"{path}.{home_key}"),
            "draw": _required_decimal(draw_value, f"{path}.{draw_key}"),
            "away": _required_decimal(away_value, f"{path}.{away_key}"),
        },
        **_copy_meta(market, source_path=path),
    }


def extract_sporttery_handicap_3way(payload: dict[str, Any]) -> dict[str, Any] | None:
    path, value = _first_path(payload, SPORTTERY_HANDICAP_PATHS)
    if path is None:
        return None
    market = _as_mapping(value)
    source = _odds_container(value)
    line = source.get("line", source.get("handicap"))
    if line is None:
        return None
    try:
        handicap = float(line)
    except (TypeError, ValueError) as exc:
        raise SportteryPayloadError(
            "解析体彩 YAML 时发现字段类型错误。",
            field_path=f"{path}.line",
            expected="数字",
            actual=line,
        ) from exc
    home_key, home_value = _lookup_outcome(source, "home", "home_win", "home_odds", "win")
    draw_key, draw_value = _lookup_outcome(source, "draw", "draw_odds")
    away_key, away_value = _lookup_outcome(source, "away", "away_win", "away_odds", "loss")
    if home_key is None or draw_key is None or away_key is None:
        return None
    return {
        "line": handicap,
        "odds": {
            "home": _required_decimal(home_value, f"{path}.{home_key}"),
            "draw": _required_decimal(draw_value, f"{path}.{draw_key}"),
            "away": _required_decimal(away_value, f"{path}.{away_key}"),
        },
        **_copy_meta(market, source_path=path),
    }


def extract_sporttery_correct_score(payload: dict[str, Any]) -> dict[str, Any] | None:
    path, value = _first_path(payload, SPORTTERY_CORRECT_SCORE_PATHS)
    if path is None:
        return None
    market = _as_mapping(value)
    source = _odds_container(value)
    scores: dict[str, float] = {}
    for outcome, price in source.items():
        if str(outcome) in SKIPPED_MARKET_KEYS:
            continue
        score = _score_key(outcome)
        if score is not None:
            odds = _optional_decimal(price, f"{path}.{outcome}")
            if odds is not None:
                scores[score] = odds
        elif str(outcome) in {"home_other", "draw_other", "away_other"}:
            odds = _optional_decimal(price, f"{path}.{outcome}")
            if odds is not None:
                scores[str(outcome)] = odds
    if not scores:
        return None
    return {"scores": scores, **_copy_meta(market, source_path=path)}


def extract_sporttery_total_goals(payload: dict[str, Any]) -> dict[str, Any] | None:
    path, value = _first_path(payload, SPORTTERY_TOTAL_GOALS_PATHS)
    if path is None:
        return None
    market = _as_mapping(value)
    source = _odds_container(value)
    odds: dict[str, float] = {}
    for outcome, price in source.items():
        if str(outcome) in SKIPPED_MARKET_KEYS:
            continue
        key = str(outcome).strip()
        if key == "7":
            key = "7+"
        if key in TOTAL_GOALS_BUCKETS:
            number = _optional_decimal(price, f"{path}.{outcome}")
            if number is not None:
                odds[key] = number
    if not odds:
        return None
    return {"odds": odds, **_copy_meta(market, source_path=path)}


def extract_sporttery_half_full(payload: dict[str, Any]) -> dict[str, Any] | None:
    path, value = _first_path(payload, SPORTTERY_HALF_FULL_PATHS)
    if path is None:
        return None
    market = _as_mapping(value)
    source = _odds_container(value)
    odds: dict[str, float] = {}
    for outcome, price in source.items():
        if str(outcome) in SKIPPED_MARKET_KEYS:
            continue
        key = str(outcome).strip().upper()
        if key in HALF_FULL_KEYS:
            number = _optional_decimal(price, f"{path}.{outcome}")
            if number is not None:
                odds[key] = number
    if not odds:
        return None
    return {"odds": odds, **_copy_meta(market, source_path=path)}


def normalize_sporttery_payload(raw_payload: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(raw_payload or {})
    one_x_two = extract_sporttery_1x2(payload)
    if one_x_two is None:
        raise SportteryPayloadError(
            "缺少体彩胜平负 1X2。",
            checked_paths=SPORTTERY_1X2_PATHS,
            top_level_keys=sorted(str(key) for key in payload.keys()),
        )

    match = _match_payload(payload, None)
    sporttery_market: dict[str, Any] = {
        "source": "yaml",
        "provider": "sporttery",
        "weight": 1.0,
        "sporttery_1x2": one_x_two,
    }
    optional_markets = {
        "sporttery_handicap_3way": extract_sporttery_handicap_3way(payload),
        "sporttery_correct_score": extract_sporttery_correct_score(payload),
        "sporttery_total_goals": extract_sporttery_total_goals(payload),
        "sporttery_half_full": extract_sporttery_half_full(payload),
    }
    for key, value in optional_markets.items():
        if value:
            sporttery_market[key] = value
    return {
        "match": match,
        "prediction_time": payload.get("prediction_time")
        or payload.get("snapshot_time")
        or "pre_match",
        "odds_channels": {
            "sporttery": {
                "role": "primary_calibration",
                "source": "yaml",
                "provider": "sporttery",
                "weight": 1.0,
            }
        },
        "markets": {"sporttery": sporttery_market},
    }


def _markets(payload: dict[str, Any]) -> dict[str, Any]:
    return _as_mapping(payload.get("markets"))


def _sporttery_source(payload: dict[str, Any]) -> dict[str, Any]:
    markets = _markets(payload)
    return _as_mapping(markets.get("sporttery"))


def _legacy_market(payload: dict[str, Any]) -> dict[str, Any]:
    return _as_mapping(payload.get("market"))


def _sporttery_section(payload: dict[str, Any]) -> dict[str, Any]:
    return _as_mapping(payload.get("sporttery"))


def extract_sporttery_market(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_sporttery_payload(payload)
    return (normalized.get("markets") or {}).get("sporttery") or {}


def _split_match_text(value: Any) -> tuple[str, str]:
    text = str(value or "")
    if " vs " in text:
        home, away = text.split(" vs ", 1)
        return home.strip() or "Home", away.strip() or "Away"
    return "Home", "Away"


def _match_payload(payload: dict[str, Any], match_overrides: dict[str, Any] | None) -> dict[str, Any]:
    match = _as_mapping(payload.get("match"))
    if not match and isinstance(payload.get("match"), str):
        home, away = _split_match_text(payload.get("match"))
        match = {"home_team": home, "away_team": away}
    match.update({key: value for key, value in (match_overrides or {}).items() if value not in (None, "")})
    match.setdefault("match_id", "")
    match.setdefault("home_team", payload.get("home_team") or "Home")
    match.setdefault("away_team", payload.get("away_team") or "Away")
    match.setdefault("competition", payload.get("competition") or "World Cup")
    match.setdefault("stage", payload.get("stage") or "")
    match.setdefault(
        "kickoff_time",
        payload.get("kickoff_time")
        or payload.get("kickoff_time_beijing")
        or payload.get("date")
        or "2026-06-12 20:00",
    )
    match.setdefault("timezone", payload.get("timezone") or "Asia/Shanghai")
    match.setdefault("venue", {"venue_type": "neutral"})
    match.setdefault("neutral_site", True)
    match["target"] = "90min_score"
    return match


def _settings(
    payload: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    settings = {
        "market_only_mode": True,
        "dc_enabled": True,
        "max_goals": 8,
        "market_weight": 1.0,
        "x1x2_weight": 1.0,
        "h2h_weight": 1.0,
        "totals_weight": 1.0,
        "ou_weight": 1.0,
        "alternate_totals_weight": 0.0,
        "btts_weight": 0.0,
        "spreads_weight": 0.0,
        "correct_score_weight": 0.0,
        "team_adjustment_strength": 1.0,
        **SPORTTERY_MODEL_WEIGHTS,
    }
    settings.update(_as_mapping(payload.get("settings")))
    settings.update({key: value for key, value in (overrides or {}).items() if value is not None})
    settings["sporttery_half_full_weight"] = 0.0
    settings["market_only_mode"] = bool(settings.get("market_only_mode", True))
    return settings


def _movement_settings(
    payload: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    settings = deepcopy(ODDS_MOVEMENT_SETTINGS)
    incoming = _as_mapping(payload.get("odds_movement_settings"))
    if incoming:
        movement_weights = dict(settings["movement_weights"])
        movement_weights.update(_as_mapping(incoming.get("movement_weights")))
        settings.update({key: value for key, value in incoming.items() if key != "movement_weights"})
        settings["movement_weights"] = movement_weights
    if overrides:
        movement_weights = dict(settings["movement_weights"])
        movement_weights.update(_as_mapping(overrides.get("movement_weights")))
        settings.update({key: value for key, value in overrides.items() if key != "movement_weights"})
        settings["movement_weights"] = movement_weights
    settings["movement_weights"]["sporttery_half_full_movement"] = 0.0
    return settings


def normalize_sporttery_only_payload(
    payload: dict[str, Any],
    *,
    match_overrides: dict[str, Any] | None = None,
    prematch_context: dict[str, Any] | None = None,
    settings_overrides: dict[str, Any] | None = None,
    movement_settings_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = deepcopy(payload or {})
    base = normalize_sporttery_payload(source)
    sporttery_market = (base.get("markets") or {}).get("sporttery") or {}
    one_x_two = _as_mapping(sporttery_market.get("sporttery_1x2"))
    one_x_two_odds = _as_mapping(one_x_two.get("odds"))
    if not one_x_two_odds:
        raise SportteryPayloadError(
            "缺少体彩胜平负 1X2。",
            checked_paths=SPORTTERY_1X2_PATHS,
            top_level_keys=sorted(str(key) for key in source.keys()),
        )

    match = deepcopy(base.get("match") or {})
    match.update({key: value for key, value in (match_overrides or {}).items() if value not in (None, "")})
    match["target"] = "90min_score"
    legacy_market: dict[str, Any] = {
        "odds_source": "sporttery_yaml",
        "odds_1x2": {
            "home": one_x_two_odds["home"],
            "draw": one_x_two_odds["draw"],
            "away": one_x_two_odds["away"],
        },
    }
    if sporttery_market.get("sporttery_handicap_3way"):
        handicap = sporttery_market["sporttery_handicap_3way"]
        handicap_odds = _as_mapping(handicap.get("odds"))
        legacy_market["rqspf"] = {
            "handicap": handicap.get("line", handicap.get("handicap")),
            "home": handicap_odds.get("home", handicap.get("home")),
            "draw": handicap_odds.get("draw", handicap.get("draw")),
            "away": handicap_odds.get("away", handicap.get("away")),
        }
    if sporttery_market.get("sporttery_correct_score"):
        legacy_market["correct_score_odds"] = deepcopy(
            sporttery_market["sporttery_correct_score"]
        )
    if sporttery_market.get("sporttery_total_goals"):
        legacy_market["sporttery_total_goals"] = deepcopy(
            sporttery_market["sporttery_total_goals"]
        )
    if sporttery_market.get("sporttery_half_full"):
        legacy_market["half_full_time"] = deepcopy(
            sporttery_market["sporttery_half_full"]
        )

    normalized: dict[str, Any] = {
        "match": match,
        "prediction_time": base.get("prediction_time", "pre_match"),
        "odds_channels": deepcopy(base.get("odds_channels") or {}),
        "markets": {"sporttery": sporttery_market},
        "market": legacy_market,
        "market_roles": {
            "calibration_sources": ["sporttery"],
            "value_comparison_sources": [],
        },
        "settings": _settings(source, settings_overrides),
        "odds_movement_settings": _movement_settings(
            source,
            movement_settings_overrides,
        ),
        "market_snapshots": deepcopy(source.get("market_snapshots") or []),
        "adjustments": deepcopy(
            source.get(
                "adjustments",
                {
                    "home_factors": [1.0],
                    "away_factors": [1.0],
                    "reasons": ["No manual pre-match lambda adjustment in sporttery-only input."],
                },
            )
        ),
        "notes": list(source.get("notes") or ["Sporttery-only primary calibration input."]),
        "warnings": list(source.get("warnings") or []),
    }
    context = prematch_context or source.get("prematch_context")
    if isinstance(context, dict) and context:
        normalized["prematch_context"] = deepcopy(context)
    if source.get("intelligence"):
        normalized["intelligence"] = deepcopy(source["intelligence"])
    return normalized


def _stable_hash(value: Any) -> str:
    text = json.dumps(value or {}, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_sporttery_prediction_context_key(
    payload: dict[str, Any],
    *,
    prematch_context: dict[str, Any] | None = None,
) -> str:
    match = _as_mapping(payload.get("match"))
    markets = _as_mapping(payload.get("markets"))
    context = {
        "home_team": match.get("home_team"),
        "away_team": match.get("away_team"),
        "match_id": match.get("match_id"),
        "sporttery_payload_hash": _stable_hash(markets.get("sporttery") or {}),
        "prematch_context_hash": _stable_hash(
            prematch_context or payload.get("prematch_context") or {}
        ),
        "model_settings_hash": _stable_hash(payload.get("settings") or {}),
        "odds_movement_settings_hash": _stable_hash(
            payload.get("odds_movement_settings") or {}
        ),
    }
    return _stable_hash(context)


def get_canonical_top_score(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not result:
        return None
    v3 = result.get("v3") or {}
    top_scores = v3.get("top_scores") or result.get("top_scores") or []
    return top_scores[0] if top_scores else None


def run_sporttery_only_prediction(
    payload: dict[str, Any],
    *,
    match_overrides: dict[str, Any] | None = None,
    prematch_context: dict[str, Any] | None = None,
    settings_overrides: dict[str, Any] | None = None,
    movement_settings_overrides: dict[str, Any] | None = None,
    db_path: Any = None,
    save_history: bool = True,
) -> dict[str, Any]:
    normalized = normalize_sporttery_only_payload(
        payload,
        match_overrides=match_overrides,
        prematch_context=prematch_context,
        settings_overrides=settings_overrides,
        movement_settings_overrides=movement_settings_overrides,
    )
    match_input = match_input_from_dict(normalized)
    result = predict(match_input, dc_enabled=bool(match_input.settings.dc_enabled))
    context_key = build_sporttery_prediction_context_key(
        normalized,
        prematch_context=prematch_context,
    )
    history_record = None
    if save_history:
        history_record = save_prediction_history(
            result,
            normalized,
            normalized.get("settings") if isinstance(normalized.get("settings"), dict) else {},
            {
                "prediction_context_key": context_key,
                "app_mode": "sporttery_only",
                "source_mode": "sporttery_only",
                "international_payload": {},
                "sporttery_payload": (normalized.get("markets") or {}).get("sporttery") or {},
                "prematch_context": prematch_context or normalized.get("prematch_context") or {},
            },
            db_path=db_path,
        )
    return {
        "payload": normalized,
        "match_input": match_input,
        "result": result,
        "context_key": context_key,
        "history_record": history_record,
    }
