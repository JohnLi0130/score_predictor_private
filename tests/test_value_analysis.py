from __future__ import annotations

from pathlib import Path

import pytest

from score_predictor.predictor import match_input_from_dict
from score_predictor.schemas import MatchInput
from score_predictor.ui.yaml_io import load_yaml_payload, merge_prediction_payload
from score_predictor.v3.value_analysis import (
    REFERENCE_WARNING,
    breakeven_probability,
    edge,
    expected_value,
    value_row,
    build_value_analysis,
)
from score_predictor.v3.market_calibration import build_market_probabilities


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _payload(relative_path: str) -> dict:
    return load_yaml_payload((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))


def _match_input(*, sporttery_in_calibration: bool = False) -> MatchInput:
    calibration_sources = ["international", "pinnacle"]
    if sporttery_in_calibration:
        calibration_sources.append("sporttery")
    return MatchInput(
        match="Home FC vs Away FC",
        kickoff_time="2026-06-10 20:00",
        timezone="Asia/Shanghai",
        target="90min_score",
        venue_type="home",
        prediction_time="T-24h",
        odds_1x2={"home": 2.20, "draw": 3.40, "away": 3.20},
        over_under={"line": 2.5, "over_odds": 2.05, "under_odds": 1.80},
        btts={"yes": 1.90, "no": 1.95},
        correct_score_odds={"1-0": 7.50, "1-1": 6.80, "2-0": 9.50},
        sporttery_total_goals_odds={"2": 3.30, "3": 3.80, "7+": 30.0},
        half_full_time_odds={"胜胜": 3.20},
        internal_model={"home_lambda": 1.35, "away_lambda": 1.05},
        market_roles={
            "calibration_sources": calibration_sources,
            "value_comparison_sources": ["sporttery"],
            "roles_configured": True,
        },
        settings={"max_goals": 7},
    )


def _v3_result() -> dict:
    return {
        "final_score_matrix": [
            {"home_goals": 1, "away_goals": 0, "score": "1-0", "prob": 0.16},
            {"home_goals": 1, "away_goals": 1, "score": "1-1", "prob": 0.14},
            {"home_goals": 2, "away_goals": 0, "score": "2-0", "prob": 0.10},
            {"home_goals": 2, "away_goals": 1, "score": "2-1", "prob": 0.08},
            {"home_goals": 3, "away_goals": 0, "score": "3-0", "prob": 0.03},
            {"home_goals": 4, "away_goals": 3, "score": "4-3", "prob": 0.01},
        ],
        "probabilities": {
            "one_x_two": {"home": 0.52, "draw": 0.25, "away": 0.23},
            "over_under": {"2.5": {"over": 0.42, "under": 0.58}},
            "btts": {"yes": 0.46, "no": 0.54},
        },
    }


def test_breakeven_probability_formula() -> None:
    assert breakeven_probability(2.5) == pytest.approx(0.4)


def test_expected_value_formula() -> None:
    assert expected_value(0.45, 2.5) == pytest.approx(0.125)


def test_edge_formula() -> None:
    assert edge(0.45, 2.5) == pytest.approx(0.05)


def test_value_row_positive_ev() -> None:
    row = value_row(
        market="胜平负",
        outcome="主胜",
        model_probability=0.55,
        market_odds=2.2,
    )

    assert row["expected_value"] > 0
    assert row["edge"] > 0
    assert row["value_reliability"] == "independent_comparison"


def test_used_in_calibration_reference_warning() -> None:
    row = value_row(
        market="比分固定奖金",
        outcome="1-0",
        model_probability=0.16,
        market_odds=7.5,
        used_in_calibration=True,
    )

    assert row["used_in_calibration"] is True
    assert row["value_reliability"] == "reference_only"
    assert row["warning"] == REFERENCE_WARNING


def test_value_rank_is_sorted_by_expected_value_descending() -> None:
    analysis = build_value_analysis(_match_input(), _v3_result())
    expected_values = [row["expected_value"] for row in analysis["value_rank"]]

    assert expected_values == sorted(expected_values, reverse=True)


def test_probability_rank_is_sorted_by_model_probability_descending() -> None:
    analysis = build_value_analysis(_match_input(), _v3_result())
    probabilities = [row["model_probability"] for row in analysis["probability_rank"]]

    assert probabilities == sorted(probabilities, reverse=True)


def test_sporttery_in_calibration_marks_reference_only_warning() -> None:
    analysis = build_value_analysis(
        _match_input(sporttery_in_calibration=True),
        _v3_result(),
    )

    assert analysis["value_rank"]
    assert all(row["used_in_calibration"] for row in analysis["value_rank"])
    assert all(row["value_reliability"] == "reference_only" for row in analysis["value_rank"])
    assert REFERENCE_WARNING in analysis["warnings"]


def test_correct_score_market_now_participates_in_v3_soft_calibration() -> None:
    match_input = _match_input(sporttery_in_calibration=False)
    market_probabilities = build_market_probabilities(match_input)
    analysis = build_value_analysis(match_input, _v3_result())

    assert market_probabilities["correct_score"]
    assert market_probabilities["correct_score_sources"]
    assert any(row["market"] == "比分固定奖金" for row in analysis["value_rank"])


def test_value_analysis_prefers_b_source_sporttery_1x2_for_canada_bosnia() -> None:
    a_payload = _payload("data/input/generated/canada_bosnia_the_odds_api.yaml")
    b_payload = _payload("竞猜赔率/小组赛_加拿大vs波黑/canada_vs_bosnia_sporttery_b_source.yaml")
    merged = merge_prediction_payload({}, a_payload, b_payload)
    match_input = match_input_from_dict(merged)

    analysis = build_value_analysis(match_input, _v3_result())
    one_x_two_rows = [row for row in analysis["value_rank"] if row["market"] == "胜平负"]
    odds_by_outcome = {row["outcome"]: row["market_odds"] for row in one_x_two_rows}

    assert odds_by_outcome["主胜"] == pytest.approx(1.62)
    assert odds_by_outcome["平局"] == pytest.approx(3.32)
    assert odds_by_outcome["客胜"] == pytest.approx(4.75)
    assert all(row["used_in_calibration"] is False for row in one_x_two_rows)
    assert all(row["value_reliability"] == "independent_comparison" for row in one_x_two_rows)
    assert analysis["source_check"]["value_comparison_source"] == "Sporttery"


def test_value_analysis_falls_back_to_a_source_as_reference_only_without_b_source() -> None:
    a_payload = _payload("data/input/generated/canada_bosnia_the_odds_api.yaml")
    merged = merge_prediction_payload({}, a_payload, None)
    match_input = match_input_from_dict(merged)

    analysis = build_value_analysis(match_input, _v3_result())
    one_x_two_rows = [row for row in analysis["value_rank"] if row["market"] == "胜平负"]
    odds_by_outcome = {row["outcome"]: row["market_odds"] for row in one_x_two_rows}

    assert odds_by_outcome["主胜"] == pytest.approx(1.85)
    assert odds_by_outcome["平局"] == pytest.approx(3.49)
    assert odds_by_outcome["客胜"] == pytest.approx(4.81)
    assert all(row["used_in_calibration"] is True for row in one_x_two_rows)
    assert all(row["value_reliability"] == "reference_only" for row in one_x_two_rows)
    assert analysis["source_check"]["used_calibration_fallback"] is True


def test_sporttery_correct_score_ev_uses_score_matrix_probability() -> None:
    a_payload = _payload("data/input/generated/canada_bosnia_the_odds_api.yaml")
    b_payload = _payload("竞猜赔率/小组赛_加拿大vs波黑/canada_vs_bosnia_sporttery_b_source.yaml")
    match_input = match_input_from_dict(merge_prediction_payload({}, a_payload, b_payload))

    analysis = build_value_analysis(match_input, _v3_result())
    row = next(row for row in analysis["value_rank"] if row["market"] == "比分固定奖金" and row["outcome"] == "1-0")

    assert row["market_odds"] == pytest.approx(5.40)
    assert row["model_probability"] == pytest.approx(0.16)
    assert row["expected_value"] == pytest.approx(0.16 * 5.40 - 1.0)
    assert any(item["outcome"] == "home_other" for item in analysis["audit_only"])


def test_sporttery_total_goals_ev_uses_total_goal_distribution() -> None:
    a_payload = _payload("data/input/generated/canada_bosnia_the_odds_api.yaml")
    b_payload = _payload("竞猜赔率/小组赛_加拿大vs波黑/canada_vs_bosnia_sporttery_b_source.yaml")
    match_input = match_input_from_dict(merge_prediction_payload({}, a_payload, b_payload))

    analysis = build_value_analysis(match_input, _v3_result())
    row = next(row for row in analysis["value_rank"] if row["market"] == "总进球" and row["outcome"] == "2")
    seven_plus = next(row for row in analysis["value_rank"] if row["market"] == "总进球" and row["outcome"] == "7+")

    assert row["market_odds"] == pytest.approx(3.10)
    assert row["model_probability"] == pytest.approx(0.24)
    assert row["expected_value"] == pytest.approx(0.24 * 3.10 - 1.0)
    assert seven_plus["model_probability"] == pytest.approx(0.01)
