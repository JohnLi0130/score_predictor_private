from __future__ import annotations

import pytest

from score_predictor.odds import fair_1x2_probs, implied_prob


def test_decimal_odds_must_be_greater_than_one() -> None:
    with pytest.raises(ValueError):
        implied_prob(1.0)


def test_fair_1x2_probs_sum_to_one() -> None:
    probs = fair_1x2_probs(1.35, 4.8, 8.5)
    assert set(probs) == {"home", "draw", "away"}
    assert sum(probs.values()) == pytest.approx(1.0)
