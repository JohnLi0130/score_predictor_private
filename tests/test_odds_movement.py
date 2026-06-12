from __future__ import annotations

import pytest

from score_predictor.market.odds_movement import (
    apply_movement_to_lambda,
    build_odds_movement_summary,
    compute_market_heat,
    compute_history_movement,
    compute_odds_movement,
)
from score_predictor.predictor import match_input_from_dict, predict


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


def test_history_movement_uses_devig_probability_delta() -> None:
    movement = compute_history_movement(
        [
            {"timestamp": "2026-06-10T10:00:00Z", "home": 2.0, "draw": 4.0, "away": 4.0},
            {"timestamp": "2026-06-10T16:00:00Z", "home": 1.8, "draw": 4.2, "away": 4.4},
        ],
        market_name="sporttery_1x2",
    )

    row = movement["outcomes"]["home"]

    assert row["open_devig_prob"] == pytest.approx(0.5)
    assert row["latest_devig_prob"] > row["open_devig_prob"]
    assert row["prob_delta"] == pytest.approx(
        row["latest_devig_prob"] - row["open_devig_prob"]
    )
    assert row["movement_direction"] == "up"


def test_apply_movement_to_lambda_is_bounded() -> None:
    movement_summary = {
        "enabled": True,
        "affect_lambda": True,
        "markets": {
            "sporttery_1x2": {
                "outcomes": {
                    "home": {"prob_delta": 0.20, "direction_consistency": 1.0, "volatility": 0.20, "reversal_count": 0},
                    "away": {"prob_delta": -0.20, "direction_consistency": 1.0, "volatility": 0.20, "reversal_count": 0},
                }
            },
            "sporttery_total_goals": {
                "outcomes": {
                    "2": {"prob_delta": 0.30, "direction_consistency": 1.0, "volatility": 0.30, "reversal_count": 0},
                    "3": {"prob_delta": 0.30, "direction_consistency": 1.0, "volatility": 0.30, "reversal_count": 0},
                }
            },
        },
        "themes": ["home_advantage_strengthened", "total_goals_2_3_cluster"],
        "drivers": [],
        "warnings": [],
        "conflict_level": "aligned",
    }
    settings = {
        "affect_lambda": True,
        "max_lambda_adjustment": 0.035,
        "max_total_lambda_adjustment": 0.045,
        "max_rho_adjustment": 0.02,
        "movement_weights": {
            "sporttery_1x2_movement": 1.0,
            "sporttery_total_goals_movement": 1.0,
            "sporttery_correct_score_movement": 1.0,
            "sporttery_handicap_3way_movement": 1.0,
            "sporttery_half_full_movement": 0.0,
        },
    }

    adjusted = apply_movement_to_lambda(1.5, 1.0, 0.01, movement_summary, settings)

    assert adjusted["applied"] is True
    assert abs(adjusted["home_adjustment_pct"]) <= 0.035 + 1e-12
    assert abs(adjusted["away_adjustment_pct"]) <= 0.035 + 1e-12
    assert abs(adjusted["total_adjustment_pct"]) <= 0.045 + 1e-12
    assert abs(adjusted["rho_adjustment"]) <= 0.02 + 1e-12


def test_half_full_movement_does_not_affect_lambda() -> None:
    movement_summary = {
        "enabled": True,
        "markets": {
            "sporttery_half_full": {
                "outcomes": {
                    "HH": {"prob_delta": 0.20, "direction_consistency": 1.0, "volatility": 0.20, "reversal_count": 0}
                }
            }
        },
        "themes": [],
        "drivers": [],
        "warnings": [],
        "conflict_level": "aligned",
    }

    adjusted = apply_movement_to_lambda(1.4, 1.1, 0.0, movement_summary, {})

    assert adjusted["applied"] is False
    assert adjusted["lambda_home_after"] == pytest.approx(1.4)
    assert adjusted["lambda_away_after"] == pytest.approx(1.1)


def test_high_volatility_reduces_adjustment() -> None:
    base_summary = {
        "enabled": True,
        "markets": {
            "sporttery_1x2": {
                "outcomes": {
                    "home": {"prob_delta": 0.04, "direction_consistency": 1.0, "volatility": 0.04, "reversal_count": 0},
                    "away": {"prob_delta": -0.04, "direction_consistency": 1.0, "volatility": 0.04, "reversal_count": 0},
                }
            }
        },
        "themes": ["home_advantage_strengthened"],
        "drivers": [],
        "warnings": [],
        "conflict_level": "aligned",
    }
    volatile_summary = {
        **base_summary,
        "themes": ["home_advantage_strengthened", "late_reversal", "movement_signal_weak_due_to_volatility"],
    }

    normal = apply_movement_to_lambda(1.4, 1.1, 0.0, base_summary, {})
    volatile = apply_movement_to_lambda(1.4, 1.1, 0.0, volatile_summary, {})

    assert abs(volatile["home_adjustment_pct"]) < abs(normal["home_adjustment_pct"])
    assert "movement_signal_weak_due_to_volatility" in volatile["warnings"]


def test_strong_cross_market_conflict_does_not_raise_and_skips_adjustment() -> None:
    movement_summary = {
        "enabled": True,
        "markets": {},
        "themes": ["cross_market_movement_conflict"],
        "drivers": [],
        "warnings": [],
        "conflict_level": "strong_conflict",
    }

    adjusted = apply_movement_to_lambda(1.4, 1.1, 0.0, movement_summary, {})

    assert adjusted["applied"] is False
    assert "cross_market_movement_conflict" in adjusted["warnings"]


def test_prediction_applies_odds_movement_lambda_adjustment_from_history() -> None:
    payload = {
        "match": {
            "match_id": "movement-a",
            "home_team": "Home",
            "away_team": "Away",
            "kickoff_time": "2026-06-12 20:00",
            "venue": {"venue_type": "neutral"},
            "target": "90min_score",
        },
        "market": {
            "odds_1x2": {"home": 2.0, "draw": 3.4, "away": 3.8},
            "over_under": {"line": 2.5, "over_odds": 1.95, "under_odds": 1.90},
        },
        "markets": {
            "sporttery": {
                "source": "Sporttery",
                "sporttery_1x2": {
                    "home": 2.10,
                    "draw": 3.30,
                    "away": 3.60,
                    "history": [
                        {"timestamp": "2026-06-12T08:00:00Z", "home": 2.25, "draw": 3.25, "away": 3.25},
                        {"timestamp": "2026-06-12T14:00:00Z", "home": 2.10, "draw": 3.30, "away": 3.60},
                    ],
                },
            }
        },
        "settings": {"market_only_mode": True, "dc_enabled": True, "max_goals": 8},
        "internal_model": {"home_lambda": 1.2, "away_lambda": 1.0},
    }

    result = predict(match_input_from_dict(payload))
    movement = result["v3"]["movement_adjustment"]

    assert movement["applied"] is True
    assert movement["lambda_home_after"] != pytest.approx(movement["lambda_home_before"])
    assert result["v3"]["top_scores"]


def test_build_summary_reports_insufficient_history_for_single_snapshot() -> None:
    payload = {
        "match": {
            "match_id": "movement-b",
            "home_team": "Home",
            "away_team": "Away",
            "kickoff_time": "2026-06-12 20:00",
            "venue": {"venue_type": "neutral"},
            "target": "90min_score",
        },
        "market": {"odds_1x2": {"home": 2.0, "draw": 3.4, "away": 3.8}},
        "internal_model": {"home_lambda": 1.2, "away_lambda": 1.0},
        "settings": {"market_only_mode": True},
        "markets": {
            "sporttery": {
                "sporttery_1x2": {
                    "history": [
                        {"timestamp": "2026-06-12T08:00:00Z", "home": 2.25, "draw": 3.25, "away": 3.25}
                    ]
                }
            }
        },
    }

    summary = build_odds_movement_summary(match_input_from_dict(payload))

    assert "insufficient_movement_history" in summary["warnings"]
