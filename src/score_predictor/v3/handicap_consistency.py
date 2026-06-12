from __future__ import annotations

from typing import Any

from score_predictor.odds import fair_1x2_probs, fair_two_way_probs
from score_predictor.schemas import AsianHandicapOdds, SportteryRqspfOdds


def _direction_from_probs(
    probabilities: dict[str, float],
    home_key: str = "home",
    away_key: str = "away",
    tolerance: float = 0.03,
) -> str:
    home = float(probabilities.get(home_key, 0.0))
    away = float(probabilities.get(away_key, 0.0))
    if abs(home - away) <= tolerance:
        return "balanced"
    return "home" if home > away else "away"


def _direction_from_asian_handicap(handicap: AsianHandicapOdds) -> str:
    if handicap.line < 0:
        return "home"
    if handicap.line > 0:
        return "away"
    fair = fair_two_way_probs(handicap.home_odds, handicap.away_odds)
    return _direction_from_probs(fair, home_key="first", away_key="second")


def _direction_from_rqspf(rqspf: SportteryRqspfOdds) -> str:
    if rqspf.handicap < 0:
        return "home"
    if rqspf.handicap > 0:
        return "away"
    fair = fair_1x2_probs(rqspf.home, rqspf.draw, rqspf.away)
    return _direction_from_probs(fair)


def _conflict_warning(name: str, directions: dict[str, str]) -> str | None:
    decisive = {
        source: direction
        for source, direction in directions.items()
        if direction in {"home", "away"}
    }
    if len(set(decisive.values())) <= 1:
        return None
    parts = ",".join(f"{source}:{direction}" for source, direction in decisive.items())
    return f"{name}:{parts}"


def check_handicap_consistency(
    fair_1x2: dict[str, float],
    model_1x2: dict[str, float],
    asian_handicap: AsianHandicapOdds | None = None,
    rqspf: SportteryRqspfOdds | None = None,
) -> dict[str, Any]:
    directions = {
        "one_x_two": _direction_from_probs(fair_1x2),
        "model": _direction_from_probs(model_1x2),
    }

    handicap_sources: dict[str, str] = {}
    if asian_handicap is not None:
        handicap_sources["asian_handicap"] = _direction_from_asian_handicap(asian_handicap)
    if rqspf is not None:
        handicap_sources["sporttery_rqspf_official"] = _direction_from_rqspf(rqspf)

    directions.update(handicap_sources)

    warnings: list[str] = []
    if asian_handicap is not None:
        warning = _conflict_warning(
            "asian_handicap_consistency_conflict",
            {
                "one_x_two": directions["one_x_two"],
                "asian_handicap": directions["asian_handicap"],
                "model": directions["model"],
            },
        )
        if warning:
            warnings.append(warning)

    if rqspf is not None:
        warnings.append("sporttery_rqspf_treated_as_official_handicap_win_draw_loss")
        warning = _conflict_warning(
            "sporttery_rqspf_consistency_conflict",
            {
                "one_x_two": directions["one_x_two"],
                "sporttery_rqspf_official": directions["sporttery_rqspf_official"],
                "model": directions["model"],
            },
        )
        if warning:
            warnings.append(warning)

    all_warning = _conflict_warning("market_direction_conflict", directions)
    if all_warning and all_warning not in warnings:
        warnings.append(all_warning)

    conflict_count = sum("conflict" in warning for warning in warnings)
    score = max(0.0, min(1.0, 1.0 - conflict_count * 0.25))

    return {
        "directions": directions,
        "warnings": warnings,
        "score": score,
        "handicap_consistency_weight": 0.0,
    }
