from __future__ import annotations


def decimal_odds_to_raw_prob(odds: float) -> float:
    if odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.")
    return 1.0 / float(odds)


def normalize_fair_probs(raw_probs: dict[str, float]) -> dict[str, float]:
    total = sum(float(value) for value in raw_probs.values())
    if total <= 0:
        raise ValueError("Probability sum must be positive.")
    return {key: float(value) / total for key, value in raw_probs.items()}


def compute_overround(raw_probs: dict[str, float]) -> float:
    return sum(float(value) for value in raw_probs.values()) - 1.0


def compute_payout_rate(raw_probs: dict[str, float]) -> float:
    total = sum(float(value) for value in raw_probs.values())
    if total <= 0:
        raise ValueError("Probability sum must be positive.")
    return 1.0 / total


def compute_bookmaker_margin(raw_probs: dict[str, float]) -> float:
    return 1.0 - compute_payout_rate(raw_probs)


def compute_hidden_multiplier(odds: float, fair_prob: float) -> float:
    if odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.")
    if fair_prob < 0:
        raise ValueError("Fair probability must be non-negative.")
    return float(odds) * float(fair_prob)


def build_market_probability_table(odds: dict[str, float]) -> dict:
    if not odds:
        raise ValueError("Odds mapping must not be empty.")

    raw_probs = {
        outcome: decimal_odds_to_raw_prob(float(decimal_odds))
        for outcome, decimal_odds in odds.items()
    }
    fair_probs = normalize_fair_probs(raw_probs)
    outcomes: dict[str, dict[str, float]] = {}
    for outcome, decimal_odds in odds.items():
        raw_prob = raw_probs[outcome]
        fair_prob = fair_probs[outcome]
        outcomes[outcome] = {
            "odds": float(decimal_odds),
            "raw_prob": raw_prob,
            "fair_prob": fair_prob,
            "raw_prob_pct": raw_prob * 100.0,
            "fair_prob_pct": fair_prob * 100.0,
            "hidden_multiplier": compute_hidden_multiplier(
                float(decimal_odds), fair_prob
            ),
        }

    raw_prob_sum = sum(raw_probs.values())
    return {
        "outcomes": outcomes,
        "raw_probs": raw_probs,
        "fair_probs": fair_probs,
        "raw_prob_sum": raw_prob_sum,
        "overround": compute_overround(raw_probs),
        "payout_rate": compute_payout_rate(raw_probs),
        "bookmaker_margin": compute_bookmaker_margin(raw_probs),
    }

