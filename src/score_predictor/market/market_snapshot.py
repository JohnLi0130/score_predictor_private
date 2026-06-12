from __future__ import annotations

from .odds_movement import compute_market_heat, compute_odds_movement


def odds_without_metadata(snapshot_market: dict) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in snapshot_market.items()
        if key != "handicap" and isinstance(value, (int, float))
    }


def compute_snapshot_movements(snapshots: list[dict]) -> dict:
    if len(snapshots) < 2:
        return {"markets": {}, "warnings": ["market_snapshots_less_than_two"]}

    opening = snapshots[0]
    current = snapshots[-1]
    markets: dict[str, dict] = {}
    warnings: list[str] = []
    for market_name in ["spf", "rqspf"]:
        if market_name not in opening or market_name not in current:
            continue
        opening_odds = odds_without_metadata(opening[market_name])
        current_odds = odds_without_metadata(current[market_name])
        if not opening_odds or not current_odds:
            warnings.append(f"{market_name}_snapshot_missing_odds")
            continue
        movement = compute_odds_movement(opening_odds, current_odds)
        heat = compute_market_heat(movement)
        market_payload = {
            "opening_timestamp": opening.get("timestamp"),
            "current_timestamp": current.get("timestamp"),
            "opening_source": opening.get("source"),
            "current_source": current.get("source"),
            "movement": movement,
            "heat": heat,
        }
        if market_name == "rqspf":
            market_payload["handicap"] = current[market_name].get(
                "handicap", opening[market_name].get("handicap")
            )
            if heat.get("heated_outcome") in {"draw", "away"}:
                market_payload.setdefault("notes", []).append(
                    "sporttery_handicap_one_goal_small_win_or_non-cover_pressure"
                )
        markets[market_name] = market_payload

    return {"markets": markets, "warnings": warnings}

