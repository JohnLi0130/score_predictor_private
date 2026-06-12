from __future__ import annotations

import json
from pathlib import Path

import pytest

from score_predictor.cli import main
from score_predictor.dixon_coles import apply_dixon_coles_adjustment
from score_predictor.poisson import score_matrix
from score_predictor.predictor import load_match_input, predict
from score_predictor.schemas import AsianHandicapOdds, MatchInput, Odds1X2, SportteryRqspfOdds
from score_predictor.v3.handicap_consistency import check_handicap_consistency
from score_predictor.v3.market_calibration import (
    build_market_probabilities,
    weighted_market_loss,
)
from score_predictor.v3.odds_channels import score_channel_consistency, score_market_quality
from score_predictor.v3.sensitivity import run_sensitivity_analysis


def _v3_match(
    *,
    btts: bool = True,
    correct_score: bool = True,
    asian_line: float = -0.75,
) -> MatchInput:
    payload = {
        "match": "Home FC vs Away FC",
        "kickoff_time": "2026-06-10 20:00",
        "timezone": "Asia/Shanghai",
        "target": "90min_score",
        "venue_type": "home",
        "prediction_time": "T-24h",
        "odds_1x2": {"home": 1.85, "draw": 3.45, "away": 4.40},
        "over_under": {"line": 2.5, "over_odds": 1.92, "under_odds": 1.92},
        "over_under_markets": [
            {"line": 1.5, "over_odds": 1.35, "under_odds": 3.05},
            {"line": 3.5, "over_odds": 3.10, "under_odds": 1.34},
            {"line": 4.5, "over_odds": 5.50, "under_odds": 1.12},
        ],
        "asian_handicap": {
            "line": asian_line,
            "home_odds": 1.95,
            "away_odds": 1.90,
        },
        "internal_model": {"home_lambda": 1.55, "away_lambda": 0.95},
        "settings": {"market_weight": 0.65, "max_goals": 7},
    }
    if btts:
        payload["btts"] = {"yes": 1.70, "no": 2.15}
    if correct_score:
        payload["correct_score_odds"] = {
            "1-0": 6.50,
            "1-1": 7.20,
            "2-0": 8.00,
            "2-1": 8.50,
            "0-0": 13.00,
            "0-1": 12.00,
        }
    return MatchInput(**payload)


def test_dixon_coles_matrix_is_renormalized() -> None:
    matrix = score_matrix(1.45, 0.90, max_goals=7)
    adjusted = apply_dixon_coles_adjustment(matrix, 1.45, 0.90, rho=0.08)

    assert adjusted["prob"].sum() == pytest.approx(1.0)
    base_00 = float(matrix.loc[matrix["score"] == "0-0", "prob"].iloc[0])
    adjusted_00 = float(adjusted.loc[adjusted["score"] == "0-0", "prob"].iloc[0])
    assert adjusted_00 != pytest.approx(base_00)


def test_btts_market_participates_in_loss() -> None:
    market_probs = build_market_probabilities(_v3_match(btts=True, correct_score=False))
    without_btts = {**market_probs, "btts": None}
    params = [1.20, 0.80, 0.0]

    assert weighted_market_loss(params, market_probs) != pytest.approx(
        weighted_market_loss(params, without_btts)
    )


def test_alternate_totals_participate_in_loss() -> None:
    market_probs = build_market_probabilities(_v3_match(btts=False, correct_score=False))
    without_alternate_totals = {
        **market_probs,
        "over_under": {
            line: values
            for line, values in market_probs["over_under"].items()
            if line == market_probs["primary_over_under_line"]
        },
    }
    params = [1.20, 0.80, 0.0]

    assert market_probs["weights"]["alternate_totals"] == pytest.approx(0.8)
    assert weighted_market_loss(params, market_probs) != pytest.approx(
        weighted_market_loss(params, without_alternate_totals)
    )


def test_spreads_market_participates_in_loss() -> None:
    market_probs = build_market_probabilities(_v3_match(btts=False, correct_score=False))
    without_spreads = {**market_probs, "spreads": None}
    params = [1.20, 0.80, 0.0]

    assert market_probs["weights"]["spreads"] == pytest.approx(0.5)
    assert weighted_market_loss(params, market_probs) != pytest.approx(
        weighted_market_loss(params, without_spreads)
    )


def test_correct_score_market_participates_in_loss() -> None:
    market_probs = build_market_probabilities(_v3_match(btts=False, correct_score=True))
    without_correct_score = {**market_probs, "correct_score": {}}
    params = [1.20, 0.80, 0.0]

    assert weighted_market_loss(params, market_probs) != pytest.approx(
        weighted_market_loss(params, without_correct_score)
    )


def test_sporttery_correct_score_soft_calibration_participates_in_loss() -> None:
    match_input = _v3_match(btts=False, correct_score=False)
    match_input.sporttery_correct_score_odds.update(
        {"0-0": 9.5, "1-0": 5.4, "1-1": 5.0, "2-0": 6.5, "2-1": 6.0}
    )
    market_probs = build_market_probabilities(match_input)
    without_sporttery_correct = {**market_probs, "correct_score": {}, "correct_score_sources": []}
    params = [1.20, 0.80, 0.0]

    assert market_probs["weights"]["sporttery_correct_score"] > 0
    assert any(source["channel"] == "sporttery" for source in market_probs["correct_score_sources"])
    assert weighted_market_loss(params, market_probs) != pytest.approx(
        weighted_market_loss(params, without_sporttery_correct)
    )


def test_sporttery_total_goals_soft_calibration_participates_in_loss() -> None:
    match_input = _v3_match(btts=False, correct_score=False)
    match_input.sporttery_total_goals_odds.update(
        {"0": 9.5, "1": 4.3, "2": 3.1, "3": 3.6, "4": 6.2, "5": 12.5, "6": 22.0, "7+": 35.0}
    )
    market_probs = build_market_probabilities(match_input)
    without_total_goals = {**market_probs, "sporttery_total_goals": {}}
    params = [1.20, 0.80, 0.0]

    assert market_probs["weights"]["sporttery_total_goals"] > 0
    assert weighted_market_loss(params, market_probs) != pytest.approx(
        weighted_market_loss(params, without_total_goals)
    )


def test_sporttery_1x2_and_handicap_3way_are_soft_constraints() -> None:
    match_input = _v3_match(btts=False, correct_score=False)
    match_input.sporttery_1x2 = Odds1X2(home=1.62, draw=3.32, away=4.75)
    match_input.sporttery_handicap_3way = SportteryRqspfOdds(
        handicap=-1,
        home=3.11,
        draw=3.20,
        away=2.02,
    )
    market_probs = build_market_probabilities(match_input)
    without_sporttery = {
        **market_probs,
        "one_x_two_sources": [
            source for source in market_probs["one_x_two_sources"] if source["channel"] != "sporttery"
        ],
        "sporttery_handicap_3way": None,
    }
    params = [1.20, 0.80, 0.0]

    assert market_probs["weights"]["sporttery_1x2"] > 0
    assert market_probs["weights"]["sporttery_handicap_3way"] > 0
    assert weighted_market_loss(params, market_probs) != pytest.approx(
        weighted_market_loss(params, without_sporttery)
    )


def test_sporttery_half_full_is_audit_only_and_does_not_affect_lambda() -> None:
    base = _v3_match(btts=False, correct_score=False)
    with_half_full = _v3_match(btts=False, correct_score=False)
    with_half_full.half_full_time_odds.update(
        {"HH": 2.51, "HD": 16.0, "HA": 36.0, "DH": 4.25, "DD": 4.8, "DA": 10.0, "AH": 25.0, "AD": 16.0, "AA": 8.4}
    )

    base_fit = predict(base)["v3"]["joint_fit"]
    half_full_result = predict(with_half_full)["v3"]
    half_full_fit = half_full_result["joint_fit"]

    assert half_full_fit["weights"]["sporttery_half_full"] == 0.0
    assert half_full_result["sporttery_market_status"]["sporttery_half_full"]["status"] == "audit_only"
    assert half_full_fit["lambda_home"] == pytest.approx(base_fit["lambda_home"])
    assert half_full_fit["lambda_away"] == pytest.approx(base_fit["lambda_away"])
    assert half_full_fit["rho"] == pytest.approx(base_fit["rho"])


def test_missing_correct_score_runs() -> None:
    result = predict(_v3_match(correct_score=False))

    assert result["v3"]["correct_score_fit_error"] is None
    assert "v3_correct_score_market_missing" in result["v3"]["risk_warnings"]


def test_sporttery_correct_score_changes_missing_correct_score_warning() -> None:
    match_input = _v3_match(btts=False, correct_score=False)
    match_input.sporttery_correct_score_odds.update(
        {"0-0": 9.5, "1-0": 5.4, "1-1": 5.0, "2-0": 6.5, "2-1": 6.0}
    )
    result = predict(match_input)

    assert "v3_correct_score_market_missing" not in result["v3"]["risk_warnings"]
    assert "sporttery_correct_score_supplemented_missing_international" in result["v3"]["risk_warnings"]
    assert "sporttery_correct_score_soft_constraint" in result["v3"]["risk_warnings"]


def test_missing_btts_runs() -> None:
    result = predict(_v3_match(btts=False))

    assert result["v3"]["btts_fit_error"] is None
    assert "v3_btts_market_missing" in result["v3"]["risk_warnings"]


def test_btts_present_does_not_emit_missing_btts_warning() -> None:
    result = predict(_v3_match(btts=True))

    assert "v3_btts_market_missing" not in result["v3"]["risk_warnings"]


def test_market_quality_caps_incomplete_correct_score_at_medium() -> None:
    quality = score_market_quality(
        {"type": "correct_score", "scores": {"0-0": 9.5, "1-0": 5.4, "1-1": 5.0}}
    )

    assert quality["level"] in {"medium", "low"}
    assert quality["score"] <= 0.70
    assert "correct_score_incomplete" in quality["warnings"]


def test_total_goals_complete_market_can_score_high() -> None:
    quality = score_market_quality(
        {
            "type": "total_goals",
            "odds": {"0": 9.5, "1": 4.3, "2": 3.1, "3": 3.6, "4": 6.2, "5": 12.5, "6": 22.0, "7+": 35.0},
        }
    )

    assert quality["level"] == "high"
    assert quality["score"] >= 0.80


def test_strong_channel_conflict_lowers_consistency_without_error() -> None:
    consistency = score_channel_consistency(
        {
            "one_x_two": {"home": 0.65, "draw": 0.20, "away": 0.15},
            "over_under": {"2.5": {"over": 0.60, "under": 0.40}},
            "btts": {"yes": 0.62, "no": 0.38},
        },
        {
            "handicap_3way": {"line": -1, "home": 0.20, "draw": 0.25, "away": 0.55},
            "total_goals": {"0": 0.20, "1": 0.35, "2": 0.25, "3": 0.10, "4": 0.05, "5": 0.03, "6": 0.01, "7+": 0.01},
            "correct_score": {"0-0": 0.20, "1-0": 0.20, "0-1": 0.10, "1-1": 0.15, "2-0": 0.15},
        },
    )

    assert consistency["level"] == "strong_conflict"
    assert consistency["score"] == pytest.approx(0.25)
    assert "odds_channel_conflict" in consistency["warnings"]


def test_handicap_consistency_warning_is_generated() -> None:
    result = check_handicap_consistency(
        {"home": 0.62, "draw": 0.24, "away": 0.14},
        {"home": 0.58, "draw": 0.25, "away": 0.17},
        asian_handicap=AsianHandicapOdds(line=1.0, home_odds=1.95, away_odds=1.90),
    )

    assert any("conflict" in warning for warning in result["warnings"])
    assert result["score"] < 1.0


def test_sensitivity_analysis_outputs_probability_ranges() -> None:
    result = run_sensitivity_analysis(1.45, 0.90, rho=0.05, dc_enabled=True)

    assert result["scenario_count"] == 27
    assert "home_win" in result["result_probability_ranges"]
    assert "over_2_5_probability_range" in result
    assert 0 <= result["stability_score"] <= 1


def test_final_confidence_score_is_bounded() -> None:
    result = predict(_v3_match(), dc_enabled=True)
    final_confidence = result["v3"]["confidence"]["final_confidence_score"]

    assert 0 <= final_confidence <= 1


def test_old_v0_input_still_runs_with_v3_section() -> None:
    project_root = Path(__file__).resolve().parents[1]
    match_input = load_match_input(project_root / "examples" / "match_input_example.yaml")

    result = predict(match_input)

    assert "final_lambda" in result
    assert result["v3"]["enabled"] is True
    assert result["v3"]["top_scores"]


def test_cli_accepts_dc_enabled_flag(capsys) -> None:
    project_root = Path(__file__).resolve().parents[1]
    example = project_root / "examples" / "match_input_example.yaml"

    exit_code = main(["predict", str(example), "--dc-enabled", "true", "--json-only"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["v3"]["joint_fit"]["dc_enabled"] is True


def test_v3_yaml_parses_unquoted_btts_yes_no_keys() -> None:
    project_root = Path(__file__).resolve().parents[1]
    match_input = load_match_input(project_root / "examples" / "match_v3_multi_market.yaml")

    assert match_input.btts is not None
    assert match_input.btts.yes == pytest.approx(1.70)
