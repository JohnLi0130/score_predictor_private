from __future__ import annotations

import math
from typing import Any

from .implied import (
    build_market_probability_table,
    decimal_odds_to_raw_prob,
    normalize_fair_probs,
)


def compute_odds_movement(opening: dict[str, float], current: dict[str, float]) -> dict:
    outcomes = [outcome for outcome in opening if outcome in current]
    if not outcomes:
        raise ValueError("Opening and current odds must share at least one outcome.")

    opening_raw = {
        outcome: decimal_odds_to_raw_prob(float(opening[outcome]))
        for outcome in outcomes
    }
    current_raw = {
        outcome: decimal_odds_to_raw_prob(float(current[outcome]))
        for outcome in outcomes
    }
    opening_fair = normalize_fair_probs(opening_raw)
    current_fair = normalize_fair_probs(current_raw)

    rows: dict[str, dict[str, float | str]] = {}
    for outcome in outcomes:
        opening_odds = float(opening[outcome])
        current_odds = float(current[outcome])
        odds_change = current_odds - opening_odds
        if abs(odds_change) < 1e-12:
            direction = "unchanged"
            movement_direction = "flat"
        elif odds_change < 0:
            direction = "odds_shortened"
            movement_direction = "up"
        else:
            direction = "odds_drifted"
            movement_direction = "down"
        fair_delta = current_fair[outcome] - opening_fair[outcome]
        if abs(fair_delta) < 0.002:
            movement_direction = "flat"
        elif fair_delta > 0:
            movement_direction = "up"
        else:
            movement_direction = "down"
        strength_abs = abs(fair_delta)
        if strength_abs >= 0.025:
            movement_strength = "strong"
        elif strength_abs >= 0.010:
            movement_strength = "medium"
        else:
            movement_strength = "weak"

        rows[outcome] = {
            "opening_odds": opening_odds,
            "current_odds": current_odds,
            "open_odds": opening_odds,
            "latest_odds": current_odds,
            "odds_change": odds_change,
            "odds_change_pct": odds_change / opening_odds,
            "opening_raw_prob": opening_raw[outcome],
            "current_raw_prob": current_raw[outcome],
            "raw_prob_change": current_raw[outcome] - opening_raw[outcome],
            "opening_fair_prob": opening_fair[outcome],
            "current_fair_prob": current_fair[outcome],
            "open_devig_prob": opening_fair[outcome],
            "latest_devig_prob": current_fair[outcome],
            "fair_prob_change": fair_delta,
            "prob_delta": fair_delta,
            "logit_delta": _logit_delta(opening_fair[outcome], current_fair[outcome]),
            "direction": direction,
            "movement_direction": movement_direction,
            "movement_strength": movement_strength,
            "volatility": 0.0,
            "reversal_count": 0,
            "direction_consistency": 1.0,
        }

    return {
        "outcomes": rows,
        "opening": build_market_probability_table(
            {outcome: opening[outcome] for outcome in outcomes}
        ),
        "current": build_market_probability_table(
            {outcome: current[outcome] for outcome in outcomes}
        ),
    }


def _logit(value: float) -> float:
    bounded = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(bounded / (1.0 - bounded))


def _logit_delta(opening: float, current: float) -> float:
    return _logit(current) - _logit(opening)


def _setting(settings: Any, key: str, default: Any) -> Any:
    if isinstance(settings, dict):
        return settings.get(key, default)
    return getattr(settings, key, default)


def _movement_weights(settings: Any) -> dict[str, float]:
    weights = _setting(settings, "movement_weights", {}) or {}
    defaults = {
        "sporttery_1x2_movement": 0.08,
        "sporttery_handicap_3way_movement": 0.06,
        "sporttery_correct_score_movement": 0.05,
        "sporttery_total_goals_movement": 0.08,
        "sporttery_half_full_movement": 0.0,
    }
    merged = dict(defaults)
    if isinstance(weights, dict):
        merged.update({str(key): max(0.0, float(value)) for key, value in weights.items()})
    merged["sporttery_half_full_movement"] = 0.0
    return merged


def _timestamp(item: dict[str, Any]) -> Any:
    return item.get("timestamp") or item.get("published_at") or item.get("snapshot_time")


def _as_odds_mapping(value: Any, market_name: str = "") -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    source = value.get("odds") if isinstance(value.get("odds"), dict) else value
    if market_name in {"sporttery_correct_score", "correct_score"} and isinstance(
        value.get("scores"), dict
    ):
        source = value["scores"]
    aliases = {
        "home_win": "home",
        "away_win": "away",
        "win": "home",
        "loss": "away",
        "yes_odds": "yes",
        "no_odds": "no",
    }
    odds: dict[str, float] = {}
    for key, item in source.items():
        normalized_key = aliases.get(str(key), str(key))
        if normalized_key in {"history", "line", "handicap", "source", "provider", "weight", "type"}:
            continue
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if number > 1.0:
            odds[normalized_key] = number
    return odds


def _history_items(market: Any) -> list[dict[str, Any]]:
    if isinstance(market, dict) and isinstance(market.get("history"), list):
        return [item for item in market["history"] if isinstance(item, dict)]
    return []


def compute_history_movement(
    history: list[dict[str, Any]],
    *,
    market_name: str,
    current_market: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = list(history or [])
    if len(rows) == 1 and current_market:
        current_odds = _as_odds_mapping(current_market, market_name)
        if current_odds:
            rows = rows + [
                {
                    "snapshot_time": current_market.get("snapshot_time")
                    or current_market.get("published_at")
                    or current_market.get("timestamp")
                    or "latest",
                    **current_odds,
                }
            ]
    if len(rows) < 2:
        return {
            "market": market_name,
            "outcomes": {},
            "opening_timestamp": _timestamp(rows[0]) if rows else None,
            "latest_timestamp": _timestamp(rows[-1]) if rows else None,
            "warnings": ["insufficient_movement_history"],
        }

    snapshots: list[dict[str, Any]] = []
    warnings: list[str] = []
    for item in rows:
        if _timestamp(item) is None:
            warnings.append("movement_snapshot_missing_timestamp")
        odds = _as_odds_mapping(item, market_name)
        if odds:
            snapshots.append({"timestamp": _timestamp(item), "odds": odds})
    if len(snapshots) < 2:
        return {
            "market": market_name,
            "outcomes": {},
            "warnings": list(dict.fromkeys(warnings + ["insufficient_movement_history"])),
        }

    opening = snapshots[0]
    latest = snapshots[-1]
    common = [
        outcome
        for outcome in opening["odds"]
        if all(outcome in snapshot["odds"] for snapshot in snapshots)
    ]
    if not common:
        return {
            "market": market_name,
            "outcomes": {},
            "warnings": list(dict.fromkeys(warnings + ["movement_history_no_common_outcomes"])),
        }

    devig_series = []
    for snapshot in snapshots:
        fair = normalize_fair_probs(
            {outcome: decimal_odds_to_raw_prob(snapshot["odds"][outcome]) for outcome in common}
        )
        devig_series.append(fair)

    outcomes: dict[str, dict[str, Any]] = {}
    for outcome in common:
        series = [float(item[outcome]) for item in devig_series]
        deltas = [series[index] - series[index - 1] for index in range(1, len(series))]
        volatility = sum(abs(delta) for delta in deltas)
        meaningful = [delta for delta in deltas if abs(delta) >= 0.002]
        reversal_count = sum(
            1
            for index in range(1, len(meaningful))
            if meaningful[index] * meaningful[index - 1] < 0
        )
        direction_consistency = (
            abs(sum(deltas)) / volatility if volatility > 1e-12 else 1.0
        )
        prob_delta = series[-1] - series[0]
        if abs(prob_delta) < 0.002:
            movement_direction = "flat"
        elif prob_delta > 0:
            movement_direction = "up"
        else:
            movement_direction = "down"
        strength_abs = abs(prob_delta)
        if strength_abs >= 0.025:
            movement_strength = "strong"
        elif strength_abs >= 0.010:
            movement_strength = "medium"
        else:
            movement_strength = "weak"
        outcomes[outcome] = {
            "open_odds": float(opening["odds"][outcome]),
            "latest_odds": float(latest["odds"][outcome]),
            "open_devig_prob": series[0],
            "latest_devig_prob": series[-1],
            "prob_delta": prob_delta,
            "logit_delta": _logit_delta(series[0], series[-1]),
            "movement_direction": movement_direction,
            "movement_strength": movement_strength,
            "volatility": volatility,
            "reversal_count": reversal_count,
            "direction_consistency": direction_consistency,
        }

    return {
        "market": market_name,
        "opening_timestamp": opening["timestamp"],
        "latest_timestamp": latest["timestamp"],
        "snapshot_count": len(snapshots),
        "outcomes": outcomes,
        "warnings": list(dict.fromkeys(warnings)),
    }


def _market_from_sources(sources: list[dict[str, Any]], *names: str) -> dict[str, Any]:
    for source in sources:
        if not isinstance(source, dict):
            continue
        for name in names:
            market = source.get(name)
            if isinstance(market, dict):
                return market
    return {}


def _delta(market: dict[str, Any] | None, outcome: str) -> float:
    return float(((market or {}).get("outcomes") or {}).get(outcome, {}).get("prob_delta", 0.0))


def _sum_delta(market: dict[str, Any] | None, outcomes: set[str]) -> float:
    rows = (market or {}).get("outcomes") or {}
    return sum(float(rows.get(outcome, {}).get("prob_delta", 0.0)) for outcome in outcomes)


def build_odds_movement_summary(match_input: Any) -> dict[str, Any]:
    settings = getattr(match_input, "odds_movement_settings", {})
    if not bool(_setting(settings, "enabled", True)):
        return {
            "enabled": False,
            "affect_lambda": bool(_setting(settings, "affect_lambda", True)),
            "markets": {},
            "themes": [],
            "drivers": [],
            "warnings": [],
        }

    sources = [
        getattr(match_input, "sporttery_market", {}) or {},
        getattr(match_input, "value_comparison_market", {}) or {},
        getattr(match_input, "calibration_market", {}) or {},
    ]
    specs = {
        "sporttery_1x2": ("sporttery_1x2", "odds_1x2", "spf"),
        "sporttery_handicap_3way": ("sporttery_handicap_3way", "rqspf", "sporttery_handicap"),
        "sporttery_correct_score": (
            "sporttery_correct_score",
            "correct_score_odds",
            "correct_score",
        ),
        "sporttery_total_goals": (
            "sporttery_total_goals",
            "sporttery_total_goals_odds",
            "total_goals",
        ),
        "sporttery_half_full": (
            "sporttery_half_full",
            "half_full_time",
            "half_full_time_odds",
        ),
    }
    markets: dict[str, Any] = {}
    warnings: list[str] = []
    for canonical_name, aliases in specs.items():
        market = _market_from_sources(sources, *aliases)
        history = _history_items(market)
        if history:
            movement = compute_history_movement(
                history,
                market_name=canonical_name,
                current_market=market,
            )
            markets[canonical_name] = movement
            warnings.extend(movement.get("warnings") or [])

    snapshots = getattr(match_input, "market_snapshots", []) or []
    if snapshots:
        legacy = _summary_from_legacy_snapshots(snapshots)
        for key, value in legacy.get("markets", {}).items():
            markets.setdefault(key, value)
        warnings.extend(legacy.get("warnings") or [])

    if not markets:
        warnings.append("insufficient_movement_history")

    themes, drivers, conflict_level = interpret_movement_themes(markets)
    return {
        "enabled": True,
        "affect_lambda": bool(_setting(settings, "affect_lambda", True)),
        "markets": markets,
        "themes": themes,
        "drivers": drivers,
        "warnings": list(dict.fromkeys(warnings)),
        "conflict_level": conflict_level,
        "settings": {
            "max_lambda_adjustment": float(_setting(settings, "max_lambda_adjustment", 0.035)),
            "max_total_lambda_adjustment": float(
                _setting(settings, "max_total_lambda_adjustment", 0.045)
            ),
            "max_rho_adjustment": float(_setting(settings, "max_rho_adjustment", 0.02)),
            "movement_weights": _movement_weights(settings),
        },
    }


def _summary_from_legacy_snapshots(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    if len(snapshots) < 2:
        return {"markets": {}, "warnings": ["insufficient_movement_history"]}
    legacy_specs = {
        "sporttery_1x2": "spf",
        "sporttery_handicap_3way": "rqspf",
    }
    markets = {}
    warnings: list[str] = []
    for canonical, legacy_name in legacy_specs.items():
        history = []
        for snapshot in snapshots:
            market = snapshot.get(legacy_name)
            if isinstance(market, dict):
                history.append(
                    {
                        "timestamp": _timestamp(snapshot),
                        **market,
                    }
                )
        if history:
            movement = compute_history_movement(history, market_name=canonical)
            markets[canonical] = movement
            warnings.extend(movement.get("warnings") or [])
    return {"markets": markets, "warnings": list(dict.fromkeys(warnings))}


def interpret_movement_themes(markets: dict[str, Any]) -> tuple[list[str], list[str], str]:
    themes: list[str] = []
    drivers: list[str] = []
    one_x_two = markets.get("sporttery_1x2")
    handicap = markets.get("sporttery_handicap_3way")
    correct_score = markets.get("sporttery_correct_score")
    total_goals = markets.get("sporttery_total_goals")

    if _delta(one_x_two, "home") > 0.008:
        themes.append("home_advantage_strengthened")
        drivers.append("sporttery_1x2_home_devig_probability_up")
    if _delta(one_x_two, "home") > 0.008 and (
        _delta(handicap, "home") <= 0.002 or _delta(handicap, "draw") > 0.004
    ):
        themes.append("home_advantage_but_not_handicap_cover")
        drivers.append("handicap_market_limits_big_home_win_read")

    low_home = _sum_delta(correct_score, {"1-0", "2-0", "2-1"})
    draw_cluster = _sum_delta(correct_score, {"0-0", "1-1", "2-2"})
    low_mid = _sum_delta(correct_score, {"1-0", "2-0", "2-1", "1-1", "0-0"})
    if low_home > 0.008:
        themes.append("movement_supports_low_score_home_win")
        themes.append("low_to_mid_score_cluster")
        drivers.append("correct_score_low_home_win_cluster_up")
    if draw_cluster > 0.008:
        themes.append("draw_cluster_strengthened")
        themes.append("movement_supports_draw_cluster")
        drivers.append("correct_score_draw_cluster_up")
    if low_mid > 0.010:
        themes.append("low_to_mid_score_cluster")

    total_2_3 = _sum_delta(total_goals, {"2", "3"})
    high_total = _sum_delta(total_goals, {"4", "5", "6", "7+"})
    if total_2_3 > 0.008:
        themes.append("total_goals_2_3_cluster")
        drivers.append("total_goals_2_3_cluster_up")

    volatility_flags = []
    for market_name, market in markets.items():
        for outcome, row in (market.get("outcomes") or {}).items():
            if int(row.get("reversal_count", 0)) > 0:
                themes.append("late_reversal")
                drivers.append(f"{market_name}.{outcome}_late_reversal")
            if float(row.get("volatility", 0.0)) >= 0.035 or float(
                row.get("direction_consistency", 1.0)
            ) <= 0.45:
                volatility_flags.append(f"{market_name}.{outcome}")
    if volatility_flags:
        themes.append("movement_signal_weak_due_to_volatility")
        drivers.extend(f"{item}_volatile" for item in volatility_flags[:4])

    conflict_count = 0
    if _delta(one_x_two, "home") > 0.008 and _delta(correct_score, "0-1") > 0.008:
        conflict_count += 1
    if total_2_3 > 0.008 and high_total > 0.008:
        conflict_count += 1
    if _delta(one_x_two, "home") > 0.008 and _delta(one_x_two, "away") > 0.008:
        conflict_count += 1
    conflict_level = "strong_conflict" if conflict_count >= 2 else "mild_conflict" if conflict_count else "aligned"
    if conflict_count:
        themes.append("cross_market_movement_conflict")
        drivers.append("movement_cross_market_conflict_detected")

    return list(dict.fromkeys(themes)), list(dict.fromkeys(drivers)), conflict_level


def apply_movement_to_lambda(
    lambda_home: float,
    lambda_away: float,
    rho: float,
    movement_summary: dict[str, Any],
    settings: Any,
) -> dict[str, Any]:
    warnings = list(movement_summary.get("warnings") or [])
    drivers = list(movement_summary.get("drivers") or [])
    themes = set(movement_summary.get("themes") or [])
    before_total = max(float(lambda_home) + float(lambda_away), 1e-9)
    weights = _movement_weights(settings)
    max_lambda = float(_setting(settings, "max_lambda_adjustment", 0.035))
    max_total = float(_setting(settings, "max_total_lambda_adjustment", 0.045))
    max_rho = float(_setting(settings, "max_rho_adjustment", 0.02))

    if not movement_summary.get("enabled", True) or not bool(
        _setting(settings, "affect_lambda", True)
    ):
        return _movement_adjustment_result(
            lambda_home,
            lambda_away,
            rho,
            lambda_home,
            lambda_away,
            rho,
            False,
            drivers,
            warnings,
            False,
        )
    if movement_summary.get("conflict_level") == "strong_conflict":
        warnings.append("cross_market_movement_conflict")
        return _movement_adjustment_result(
            lambda_home,
            lambda_away,
            rho,
            lambda_home,
            lambda_away,
            rho,
            False,
            drivers,
            list(dict.fromkeys(warnings)),
            False,
        )

    markets = movement_summary.get("markets") or {}
    one_x_two = markets.get("sporttery_1x2")
    handicap = markets.get("sporttery_handicap_3way")
    correct_score = markets.get("sporttery_correct_score")
    total_goals = markets.get("sporttery_total_goals")

    share_signal = weights["sporttery_1x2_movement"] * (
        _delta(one_x_two, "home") - _delta(one_x_two, "away")
    )
    if _delta(handicap, "home") <= 0.002 and _delta(handicap, "draw") > 0.004:
        share_signal *= 0.55
    share_signal += weights["sporttery_correct_score_movement"] * (
        _sum_delta(correct_score, {"1-0", "2-0", "2-1"})
        - _sum_delta(correct_score, {"0-1", "0-2", "1-2"})
    )

    total_signal = weights["sporttery_total_goals_movement"] * (
        _sum_delta(total_goals, {"2", "3"})
        - _sum_delta(total_goals, {"0", "1", "5", "6", "7+"})
    )
    rho_signal = weights["sporttery_correct_score_movement"] * 0.35 * (
        _sum_delta(correct_score, {"0-0", "1-0", "1-1", "2-1"})
    )

    multiplier = _movement_reliability_multiplier(markets, themes)
    if multiplier < 0.75:
        warnings.append("movement_signal_weak_due_to_volatility")
    if "late_reversal" in themes:
        warnings.append("odds_movement_reversal")
        warnings.append("late_odds_movement_detected")

    share_pct = max(-max_lambda, min(max_lambda, share_signal * multiplier))
    total_pct = max(-max_total, min(max_total, total_signal * multiplier))
    rho_adjustment = max(-max_rho, min(max_rho, rho_signal * multiplier))

    home_after = float(lambda_home) * (1.0 + total_pct + share_pct)
    away_after = float(lambda_away) * (1.0 + total_pct - share_pct)
    rho_after = float(rho) + rho_adjustment

    home_after, home_clamped = _clamp_individual_lambda(
        float(lambda_home),
        home_after,
        max_lambda,
    )
    away_after, away_clamped = _clamp_individual_lambda(
        float(lambda_away),
        away_after,
        max_lambda,
    )
    home_after, away_after, total_clamped = _clamp_total_lambda(
        float(lambda_home),
        float(lambda_away),
        home_after,
        away_after,
        max_total,
    )
    rho_after, rho_clamped = _clamp_rho(float(rho), rho_after, max_rho)
    clamped = home_clamped or away_clamped or total_clamped or rho_clamped
    if clamped:
        warnings.append("odds_movement_adjustment_clamped")
    if abs(home_after - lambda_home) > 1e-10 or abs(away_after - lambda_away) > 1e-10 or abs(rho_after - rho) > 1e-10:
        warnings.append("odds_movement_lambda_adjusted")

    if "movement_supports_low_score_home_win" in themes:
        warnings.append("movement_supports_low_score_home_win")
    if "movement_supports_draw_cluster" in themes:
        warnings.append("movement_supports_draw_cluster")
    if movement_summary.get("conflict_level") == "mild_conflict":
        warnings.append("cross_market_movement_conflict")

    return _movement_adjustment_result(
        lambda_home,
        lambda_away,
        rho,
        home_after,
        away_after,
        rho_after,
        abs(home_after - lambda_home) > 1e-10
        or abs(away_after - lambda_away) > 1e-10
        or abs(rho_after - rho) > 1e-10,
        list(dict.fromkeys(drivers + list(themes))),
        list(dict.fromkeys(warnings)),
        clamped,
    )


def _movement_reliability_multiplier(markets: dict[str, Any], themes: set[str]) -> float:
    multiplier = 1.0
    if "late_reversal" in themes:
        multiplier = min(multiplier, 0.5)
    if "movement_signal_weak_due_to_volatility" in themes:
        multiplier = min(multiplier, 0.5)
    if "cross_market_movement_conflict" in themes:
        multiplier = min(multiplier, 0.5)
    rows = [
        row
        for market in markets.values()
        for row in (market.get("outcomes") or {}).values()
    ]
    if rows:
        avg_consistency = sum(float(row.get("direction_consistency", 1.0)) for row in rows) / len(rows)
        multiplier *= max(0.25, min(1.0, avg_consistency))
    return max(0.0, min(1.0, multiplier))


def _clamp_individual_lambda(before: float, after: float, limit: float) -> tuple[float, bool]:
    lower = before * (1.0 - limit)
    upper = before * (1.0 + limit)
    clamped = max(lower, min(upper, after))
    return clamped, abs(clamped - after) > 1e-12


def _clamp_total_lambda(
    home_before: float,
    away_before: float,
    home_after: float,
    away_after: float,
    limit: float,
) -> tuple[float, float, bool]:
    before_total = max(home_before + away_before, 1e-9)
    after_total = max(home_after + away_after, 1e-9)
    pct = after_total / before_total - 1.0
    if abs(pct) <= limit:
        return home_after, away_after, False
    target_total = before_total * (1.0 + (limit if pct > 0 else -limit))
    scale = target_total / after_total
    return home_after * scale, away_after * scale, True


def _clamp_rho(before: float, after: float, limit: float) -> tuple[float, bool]:
    lower = before - limit
    upper = before + limit
    clamped = max(lower, min(upper, after))
    return clamped, abs(clamped - after) > 1e-12


def _movement_adjustment_result(
    lambda_home_before: float,
    lambda_away_before: float,
    rho_before: float,
    lambda_home_after: float,
    lambda_away_after: float,
    rho_after: float,
    applied: bool,
    drivers: list[str],
    warnings: list[str],
    clamped: bool,
) -> dict[str, Any]:
    before_total = max(float(lambda_home_before) + float(lambda_away_before), 1e-9)
    after_total = float(lambda_home_after) + float(lambda_away_after)
    return {
        "lambda_home_before": float(lambda_home_before),
        "lambda_away_before": float(lambda_away_before),
        "rho_before": float(rho_before),
        "lambda_home_after": float(lambda_home_after),
        "lambda_away_after": float(lambda_away_after),
        "rho_after": float(rho_after),
        "home_adjustment_pct": float(lambda_home_after) / float(lambda_home_before) - 1.0,
        "away_adjustment_pct": float(lambda_away_after) / float(lambda_away_before) - 1.0,
        "total_adjustment_pct": after_total / before_total - 1.0,
        "rho_adjustment": float(rho_after) - float(rho_before),
        "applied": bool(applied),
        "clamped": bool(clamped),
        "drivers": list(dict.fromkeys(drivers)),
        "warnings": list(dict.fromkeys(warnings)),
    }


def compute_market_heat(movement: dict) -> dict:
    rows = movement.get("outcomes", {})
    warnings: list[str] = []
    drivers: list[str] = []
    if not rows:
        return {
            "heated_outcome": None,
            "heat_level": "low",
            "drivers": [],
            "warnings": ["movement_empty"],
        }

    heated_outcome = max(
        rows,
        key=lambda outcome: float(rows[outcome].get("fair_prob_change", 0.0)),
    )
    heated_row = rows[heated_outcome]
    fair_change = float(heated_row.get("fair_prob_change", 0.0))
    odds_change_pct = float(heated_row.get("odds_change_pct", 0.0))

    if fair_change <= 0:
        heated_outcome = None
        heat_level = "low"
        warnings.append("no_outcome_with_positive_fair_probability_change")
    elif fair_change >= 0.03 or odds_change_pct <= -0.08:
        heat_level = "high"
    elif fair_change >= 0.015 or odds_change_pct <= -0.04:
        heat_level = "medium"
    else:
        heat_level = "low"

    for outcome, row in rows.items():
        direction = row.get("direction")
        if direction == "odds_shortened":
            drivers.append(f"{outcome}_odds_shortened")
        elif direction == "odds_drifted":
            drivers.append(f"{outcome}_odds_drifted")

    if heated_outcome in {"draw", "away"}:
        drivers.append(f"{heated_outcome}_cold_or_draw_pressure_detected")
    if heated_outcome == "away":
        drivers.append("away_result_market_heat")

    return {
        "heated_outcome": heated_outcome,
        "heat_level": heat_level,
        "drivers": list(dict.fromkeys(drivers)),
        "warnings": warnings,
    }
