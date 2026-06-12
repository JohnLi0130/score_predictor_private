from __future__ import annotations

import pytest

from score_predictor.intelligence.adjustments_from_intel import (
    build_intelligence_adjustments,
)
from score_predictor.intelligence.prematch_context import (
    parse_prematch_context,
    prematch_context_to_intelligence,
)
from score_predictor.predictor import match_input_from_dict, predict
from score_predictor.ui.form_helpers import copy_default_form_state
from score_predictor.ui.yaml_io import (
    apply_prematch_context_to_form_state,
    build_yaml_from_form_state,
)


def _prematch_payload(
    *,
    confidence: str = "medium",
    injuries: list[str] | None = None,
    enabled: bool = True,
    max_total_adjustment: float = 0.15,
) -> dict:
    return {
        "schema_version": "prematch_context_v1",
        "match_id": "canada_vs_bosnia_herzegovina_20260612",
        "match": {
            "home_team": "Canada",
            "away_team": "Bosnia and Herzegovina",
            "competition": "FIFA World Cup 2026",
            "stage": "Group stage",
            "kickoff_time_local": "2026-06-13 03:00",
            "venue": {
                "name": "Toronto Stadium",
                "city": "Toronto",
                "country": "Canada",
                "altitude_type": "normal",
                "home_advantage": True,
                "neutral_site": False,
            },
        },
        "source_quality": {
            "overall_confidence": confidence,
            "source_type": "mixed_public_reports",
            "requires_official_confirmation": True,
        },
        "home_context": {
            "fifa_rank": 30,
            "injuries": injuries or [],
            "strengths": ["high press"],
            "weaknesses": [],
            "lineup_notes": ["official lineup not yet confirmed"],
            "tactical_profile": [],
            "recent_form_notes": [],
            "model_effect_hint": {"lambda_home": 9.0},
        },
        "away_context": {
            "fifa_rank": 64,
            "injuries": [],
            "strengths": [],
            "weaknesses": [],
            "lineup_notes": [],
            "tactical_profile": [],
            "recent_form_notes": [],
            "model_effect_hint": {},
        },
        "match_context": {
            "head_to_head": {},
            "motivation": {},
            "tactical_matchup": [],
            "risk_notes": [],
        },
        "model_adjustment_policy": {
            "enabled": enabled,
            "adjustment_type": "bounded_multiplicative",
            "max_single_fact_adjustment": 0.05,
            "max_major_fact_adjustment": 0.12,
            "max_total_adjustment": max_total_adjustment,
            "do_not_override_market": True,
            "confidence": confidence,
        },
    }


def test_prematch_context_v1_yaml_can_be_parsed() -> None:
    parsed = parse_prematch_context(_prematch_payload())

    assert parsed["schema_version"] == "prematch_context_v1"
    assert parsed["_audit"]["subjective_detected"] is False
    assert parsed["_safe"]["match"]["home_team"] == "Canada"


def test_prematch_context_auto_fills_ui_fields() -> None:
    state = apply_prematch_context_to_form_state(
        copy_default_form_state(),
        _prematch_payload(injuries=["star striker FW"]),
    )

    assert state["home_team"] == "Canada"
    assert state["away_team"] == "Bosnia and Herzegovina"
    assert state["home_fifa_rank"] == 30
    assert state["away_fifa_rank"] == 64
    assert state["home_key_players_missing"] == "star striker FW"
    assert state["prematch_source_type"] == "mixed_public_reports"
    assert state["prematch_adjustment_enabled"] is True


def test_subjective_opinion_does_not_enter_lambda_adjustment() -> None:
    intel = prematch_context_to_intelligence(
        _prematch_payload(injuries=["我觉得主队方向很稳"])
    )
    result = build_intelligence_adjustments(intel, 0.65)

    assert intel.injuries_suspensions["home"].absent == []
    assert result["home_lambda_factor"] == pytest.approx(1.0)
    assert result["away_lambda_factor"] == pytest.approx(1.0)
    assert result["total_lambda_factor"] == pytest.approx(1.0)
    assert "prematch_context_subjective_content_audit_only" in result["warnings"]


def test_prematch_context_adjustment_total_is_capped() -> None:
    intel = prematch_context_to_intelligence(
        _prematch_payload(
            confidence="high",
            injuries=["star striker FW", "playmaker AM"],
            max_total_adjustment=0.15,
        )
    )
    result = build_intelligence_adjustments(intel, 0.65)

    for key in ("home_lambda_factor", "away_lambda_factor", "total_lambda_factor"):
        assert 0.85 <= result[key] <= 1.15


def test_low_source_quality_reduces_adjustment_strength() -> None:
    payload = _prematch_payload(injuries=["star striker FW"])
    high = build_intelligence_adjustments(
        prematch_context_to_intelligence({**payload, "source_quality": {
            **payload["source_quality"],
            "overall_confidence": "high",
        }}),
        0.65,
    )
    low = build_intelligence_adjustments(
        prematch_context_to_intelligence({**payload, "source_quality": {
            **payload["source_quality"],
            "overall_confidence": "low",
        }}),
        0.65,
    )

    assert low["home_lambda_factor"] > high["home_lambda_factor"]
    assert "prematch_context_source_quality_reduced_adjustment" in low["warnings"]


def test_manual_input_flow_still_runs_without_prematch_context_yaml() -> None:
    payload = build_yaml_from_form_state(copy_default_form_state())

    assert "prematch_context" not in payload
    match_input = match_input_from_dict(payload)
    result = predict(match_input)

    assert result["top_scores"]
    assert match_input.intelligence is not None
