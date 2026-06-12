from __future__ import annotations

from score_predictor.intelligence.adjustments_from_intel import build_intelligence_adjustments
from score_predictor.intelligence.schemas import IntelligenceInput, LineupInfo, NarrativeFlags, PlayerInfo


def _low_lineup() -> LineupInfo:
    return LineupInfo(
        formation="4-4-2",
        confirmed=True,
        starters=[
            PlayerInfo(name=f"Bench {idx}", position="UNKNOWN", is_regular_starter=False, role_importance="bench")
            for idx in range(11)
        ],
    )


def test_friendly_applies_total_goals_discount() -> None:
    result = build_intelligence_adjustments(IntelligenceInput(match_type="friendly"), 0.65)
    assert result["total_lambda_factor"] < 1.0
    assert "friendly_match_total_goals_discount" in result["warnings"]


def test_high_narrative_heat_lowers_market_weight() -> None:
    result = build_intelligence_adjustments(
        IntelligenceInput(
            narrative_flags=NarrativeFlags(
                coach_debut=True,
                player_milestone=True,
                public_hype_home=True,
            )
        ),
        0.65,
    )
    assert result["market_weight"] == 0.55
    assert "market_may_be_public_sentiment_polluted" in result["warnings"]


def test_low_home_lsi_lowers_home_lambda_factor() -> None:
    result = build_intelligence_adjustments(
        IntelligenceInput(
            official_lineups_available=True,
            lineups={"home": _low_lineup()},
        ),
        0.65,
    )
    assert result["home_lambda_factor"] < 1.0


def test_both_low_lsi_lowers_total_lambda_factor() -> None:
    result = build_intelligence_adjustments(
        IntelligenceInput(
            official_lineups_available=True,
            lineups={"home": _low_lineup(), "away": _low_lineup()},
        ),
        0.65,
    )
    assert result["total_lambda_factor"] < 1.0
    assert "both_teams_rotation_or_low_strength" in result["warnings"]


def test_factors_are_clamped() -> None:
    result = build_intelligence_adjustments(
        IntelligenceInput(
            match_type="club_friendly",
            official_lineups_available=True,
            lineups={"home": _low_lineup(), "away": _low_lineup()},
        ),
        0.10,
    )
    assert result["market_weight"] >= 0.40
    assert result["total_lambda_factor"] >= 0.70

