from __future__ import annotations

from score_predictor.intelligence.match_context import compute_match_intensity
from score_predictor.intelligence.schemas import IntelligenceInput, NarrativeFlags


def test_world_cup_returns_high_mii() -> None:
    result = compute_match_intensity(IntelligenceInput(match_type="world_cup"))
    assert result["level"] == "high"


def test_friendly_returns_low_mii() -> None:
    result = compute_match_intensity(IntelligenceInput(match_type="friendly"))
    assert result["level"] == "low"


def test_friendly_with_ceremony_lowers_mii() -> None:
    plain = compute_match_intensity(IntelligenceInput(match_type="friendly"))
    ceremony = compute_match_intensity(
        IntelligenceInput(
            match_type="friendly",
            narrative_flags=NarrativeFlags(ceremonial_match=True),
        )
    )
    assert ceremony["score"] < plain["score"]

