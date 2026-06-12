from __future__ import annotations

from collections.abc import Iterable


def _combined_factor(factors: Iterable[float]) -> float:
    result = 1.0
    for factor in factors:
        if factor <= 0:
            raise ValueError("Adjustment factors must be positive.")
        if factor < 0.5 or factor > 1.5:
            raise ValueError("Adjustment factors must stay between 0.5 and 1.5 in V0.")
        result *= factor
    return result


def apply_multiplicative_adjustments(
    lambda_home: float,
    lambda_away: float,
    home_factors: list[float] | None = None,
    away_factors: list[float] | None = None,
) -> tuple[float, float]:
    if lambda_home <= 0 or lambda_away <= 0:
        raise ValueError("Lambda values must be positive.")

    home_factor = _combined_factor(home_factors or [])
    away_factor = _combined_factor(away_factors or [])
    return float(lambda_home * home_factor), float(lambda_away * away_factor)
