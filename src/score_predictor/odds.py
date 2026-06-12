from __future__ import annotations

from collections.abc import Mapping


def implied_prob(odds: float) -> float:
    if odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.")
    return 1.0 / odds


def normalize_probs(raw_probs: Mapping[str, float]) -> dict[str, float]:
    total = sum(raw_probs.values())
    if total <= 0:
        raise ValueError("Probability sum must be positive.")
    return {key: float(value / total) for key, value in raw_probs.items()}


def fair_1x2_probs(home_odds: float, draw_odds: float, away_odds: float) -> dict[str, float]:
    raw = {
        "home": implied_prob(home_odds),
        "draw": implied_prob(draw_odds),
        "away": implied_prob(away_odds),
    }
    return normalize_probs(raw)


def fair_two_way_probs(first_odds: float, second_odds: float) -> dict[str, float]:
    raw = {
        "first": implied_prob(first_odds),
        "second": implied_prob(second_odds),
    }
    return normalize_probs(raw)


def fair_over_under_probs(over_odds: float, under_odds: float) -> dict[str, float]:
    fair = fair_two_way_probs(over_odds, under_odds)
    return {"over": fair["first"], "under": fair["second"]}
