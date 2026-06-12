from __future__ import annotations

import pytest

from score_predictor.market.implied import (
    build_market_probability_table,
    compute_hidden_multiplier,
    compute_overround,
    compute_payout_rate,
    decimal_odds_to_raw_prob,
    normalize_fair_probs,
)


def test_decimal_odds_must_be_greater_than_one() -> None:
    with pytest.raises(ValueError):
        decimal_odds_to_raw_prob(1.0)


def test_market_probability_features() -> None:
    raw = {
        "home": decimal_odds_to_raw_prob(2.0),
        "draw": decimal_odds_to_raw_prob(4.0),
        "away": decimal_odds_to_raw_prob(4.0),
    }
    fair = normalize_fair_probs(raw)

    assert raw["home"] == pytest.approx(0.5)
    assert sum(fair.values()) == pytest.approx(1.0)
    assert compute_overround(raw) == pytest.approx(0.0)
    assert compute_payout_rate(raw) == pytest.approx(1.0)
    assert compute_hidden_multiplier(2.0, fair["home"]) == pytest.approx(1.0)


def test_build_market_probability_table_contains_all_features() -> None:
    table = build_market_probability_table({"home": 1.73, "draw": 3.2, "away": 4.18})

    assert table["raw_prob_sum"] > 1.0
    assert table["payout_rate"] == pytest.approx(1 / table["raw_prob_sum"])
    assert "hidden_multiplier" in table["outcomes"]["home"]
    assert sum(table["fair_probs"].values()) == pytest.approx(1.0)

