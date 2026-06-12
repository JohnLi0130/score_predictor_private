from __future__ import annotations

import math


def blend_lambdas(
    market_home: float,
    market_away: float,
    internal_home: float,
    internal_away: float,
    market_weight: float = 0.65,
) -> tuple[float, float]:
    if not 0 <= market_weight <= 1:
        raise ValueError("market_weight must be between 0 and 1.")
    for value in (market_home, market_away, internal_home, internal_away):
        if value <= 0:
            raise ValueError("Lambda values must be positive.")

    internal_weight = 1.0 - market_weight
    final_home = math.exp(
        market_weight * math.log(market_home)
        + internal_weight * math.log(internal_home)
    )
    final_away = math.exp(
        market_weight * math.log(market_away)
        + internal_weight * math.log(internal_away)
    )

    return float(final_home), float(final_away)
