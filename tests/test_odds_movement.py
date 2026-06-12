from __future__ import annotations

import pytest

from score_predictor.market.odds_movement import (
    compute_market_heat,
    compute_odds_movement,
)


def test_odds_shortened_and_drifted_are_detected() -> None:
    movement = compute_odds_movement(
        {"home": 2.0, "draw": 3.2, "away": 4.5},
        {"home": 1.8, "draw": 3.3, "away": 4.8},
    )

    assert movement["outcomes"]["home"]["direction"] == "odds_shortened"
    assert movement["outcomes"]["away"]["direction"] == "odds_drifted"
    assert movement["outcomes"]["home"]["fair_prob_change"] > 0


def test_fair_probability_change_matches_current_minus_opening() -> None:
    movement = compute_odds_movement(
        {"home": 2.0, "draw": 4.0, "away": 4.0},
        {"home": 1.8, "draw": 4.2, "away": 4.4},
    )
    row = movement["outcomes"]["home"]

    assert row["fair_prob_change"] == pytest.approx(
        row["current_fair_prob"] - row["opening_fair_prob"]
    )


def test_market_heat_identifies_heated_outcome() -> None:
    movement = compute_odds_movement(
        {"home": 2.2, "draw": 3.1, "away": 3.4},
        {"home": 1.8, "draw": 3.3, "away": 4.2},
    )

    heat = compute_market_heat(movement)

    assert heat["heated_outcome"] == "home"
    assert heat["heat_level"] in {"medium", "high"}

