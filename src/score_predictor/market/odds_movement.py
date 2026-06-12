from __future__ import annotations

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
        elif odds_change < 0:
            direction = "odds_shortened"
        else:
            direction = "odds_drifted"

        rows[outcome] = {
            "opening_odds": opening_odds,
            "current_odds": current_odds,
            "odds_change": odds_change,
            "odds_change_pct": odds_change / opening_odds,
            "opening_raw_prob": opening_raw[outcome],
            "current_raw_prob": current_raw[outcome],
            "raw_prob_change": current_raw[outcome] - opening_raw[outcome],
            "opening_fair_prob": opening_fair[outcome],
            "current_fair_prob": current_fair[outcome],
            "fair_prob_change": current_fair[outcome] - opening_fair[outcome],
            "direction": direction,
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

