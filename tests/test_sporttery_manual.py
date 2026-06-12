from __future__ import annotations

from score_predictor.connectors.sporttery_manual import normalize_sporttery_manual


def test_sporttery_manual_normalizes_market_and_features() -> None:
    result = normalize_sporttery_manual(
        {
            "spf": {"home": 1.73, "draw": 3.20, "away": 4.18},
            "rqspf": {"handicap": -1, "home": 3.82, "draw": 3.11, "away": 1.83},
            "total_goals": {"0": 11.0, "1": 4.5, "2": 3.4},
            "correct_score": {"0-0": 11.0, "1-0": 6.5, "1-1": 7.0},
        }
    )

    assert result["source"] == "manual_sporttery"
    assert result["market"]["odds_1x2"]["home"] == 1.73
    assert "rqspf" in result["market"]
    assert "asian_handicap" not in result["market"]
    assert "total_goals" in result["features"]


def test_total_goals_analysis_returns_top_totals() -> None:
    result = normalize_sporttery_manual(
        {
            "spf": {"home": 1.8, "draw": 3.4, "away": 4.2},
            "total_goals": {"0": 15.0, "1": 5.0, "2": 3.2, "3": 3.7},
        }
    )

    assert result["features"]["total_goals"]["top_total_goals"][0]["outcome"] == "2"

