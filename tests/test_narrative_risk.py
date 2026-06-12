from __future__ import annotations

from score_predictor.intelligence.narrative_risk import compute_narrative_heat
from score_predictor.intelligence.schemas import NarrativeFlags


def test_heavy_narrative_returns_high() -> None:
    result = compute_narrative_heat(
        NarrativeFlags(coach_debut=True, player_milestone=True, public_hype_home=True)
    )
    assert result["level"] == "high"


def test_no_flags_returns_low() -> None:
    result = compute_narrative_heat(NarrativeFlags())
    assert result["level"] == "low"
    assert result["score"] == 0

