from __future__ import annotations

from pathlib import Path

import pytest

from score_predictor.poisson import score_matrix
from score_predictor.predictor import load_match_input, match_input_from_dict, predict
from score_predictor.ui.charts import (
    btts_probabilities,
    over_under_probabilities,
    score_matrix_to_frame,
    total_goals_distribution,
)
from score_predictor.ui.form_helpers import copy_default_form_state
from score_predictor.ui.yaml_io import (
    build_yaml_from_form_state,
    dump_yaml,
    load_yaml_payload,
    merge_prediction_payload,
)
from score_predictor.ui.streamlit_app import _resolve_event_fetch_selection


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _payload(relative_path: str) -> dict:
    return load_yaml_payload((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))


def test_ui_yaml_builder_output_can_be_read_by_cli_loader(tmp_path) -> None:
    state = copy_default_form_state()
    state["correct_score_rows"] = [
        {"score": "0-0", "odds": 13.0},
        {"score": "1-0", "odds": 6.5},
        {"score": "1-1", "odds": 7.2},
    ]
    payload = build_yaml_from_form_state(state)
    path = tmp_path / "downloaded_match.yaml"
    path.write_text(dump_yaml(payload), encoding="utf-8")

    match_input = load_match_input(path)
    result = predict(match_input)

    assert match_input.settings.market_only_mode is True
    assert result["v3"]["enabled"] is True
    assert result["v3"]["top_scores"]


def test_market_only_mode_without_internal_lambda_runs() -> None:
    state = copy_default_form_state()
    payload = build_yaml_from_form_state(state)
    payload.pop("internal_model")

    match_input = match_input_from_dict(payload)
    result = predict(match_input)

    assert match_input.settings.market_only_mode is True
    assert match_input.internal_model.home_lambda > 0
    assert match_input.internal_model.away_lambda > 0
    assert result["v3"]["lambda_flow"]["final_lambda_home"] == pytest.approx(
        result["v3"]["lambda_flow"]["market_prior_lambda_home"]
    )


def test_chart_data_helpers_handle_score_matrix() -> None:
    matrix = score_matrix(1.35, 0.95, max_goals=8)
    frame = score_matrix_to_frame(matrix)
    total_goals = total_goals_distribution(frame)
    over_under = over_under_probabilities(frame)
    btts = btts_probabilities(frame)

    assert {"home_goals", "away_goals", "score", "prob"}.issubset(frame.columns)
    assert total_goals["probability"].sum() == pytest.approx(1.0)
    assert set(over_under["line"]) == {"1.5", "2.5", "3.5", "4.5"}
    assert btts["yes"] + btts["no"] == pytest.approx(1.0)


def test_manual_event_id_can_bypass_event_lookup() -> None:
    selection = _resolve_event_fetch_selection(
        manual_event_id="d1f4f946c70a0b4e81f5d43e9d32361c",
        selected_event_id="",
        sport_key_input="soccer_fifa_world_cup",
        selected_sport_key="",
    )

    assert selection["event_id"] == "d1f4f946c70a0b4e81f5d43e9d32361c"
    assert selection["sport_key"] == "soccer_fifa_world_cup"
    assert selection["used_manual_event_id"] is True
    assert selection["missing_sport_key"] is False


def test_manual_event_id_requires_sport_key() -> None:
    selection = _resolve_event_fetch_selection(
        manual_event_id="d1f4f946c70a0b4e81f5d43e9d32361c",
        selected_event_id="evt_from_lookup",
        sport_key_input="",
        selected_sport_key="",
    )

    assert selection["event_id"] == "d1f4f946c70a0b4e81f5d43e9d32361c"
    assert selection["missing_sport_key"] is True


def test_merge_prediction_payload_keeps_a_and_b_sources_separate() -> None:
    a_payload = _payload("data/input/generated/canada_bosnia_the_odds_api.yaml")
    b_payload = _payload("竞猜赔率/小组赛_加拿大vs波黑/canada_vs_bosnia_sporttery_b_source.yaml")
    base_payload = build_yaml_from_form_state(copy_default_form_state())

    merged = merge_prediction_payload(base_payload, a_payload, b_payload)

    assert merged["markets"]["international"]["source"] == "The Odds API"
    assert merged["markets"]["sporttery"]["source"] == "Sporttery"
    assert merged["market"]["odds_1x2"]["home"] == pytest.approx(1.85)
    assert merged["markets"]["sporttery"]["odds_1x2"]["home"] == pytest.approx(1.62)
    assert merged["odds_channels"]["international"]["role"] == "primary_calibration"
    assert merged["odds_channels"]["sporttery"]["role"] == "supplemental_calibration"


def test_b_source_payload_does_not_overwrite_a_source_payload() -> None:
    a_payload = _payload("data/input/generated/canada_bosnia_the_odds_api.yaml")
    b_payload = _payload("竞猜赔率/小组赛_加拿大vs波黑/canada_vs_bosnia_sporttery_b_source.yaml")

    merged = merge_prediction_payload({}, a_payload, b_payload)

    assert a_payload["market"]["odds_1x2"]["home"] == pytest.approx(1.85)
    assert b_payload["markets"]["value_comparison"]["odds_1x2"]["home"] == pytest.approx(1.62)
    assert merged["market"]["odds_1x2"]["home"] == pytest.approx(1.85)
    assert merged["markets"]["sporttery"]["odds_1x2"]["home"] == pytest.approx(1.62)


def test_sporttery_channel_soft_calibration_participates_in_prediction_lambda() -> None:
    a_payload = _payload("data/input/generated/canada_bosnia_the_odds_api.yaml")
    b_payload = _payload("竞猜赔率/小组赛_加拿大vs波黑/canada_vs_bosnia_sporttery_b_source.yaml")

    a_only = merge_prediction_payload({}, a_payload, None)
    with_b = merge_prediction_payload({}, a_payload, b_payload)

    a_only_fit = predict(match_input_from_dict(a_only))["v3"]["joint_fit"]
    with_b_fit = predict(match_input_from_dict(with_b))["v3"]["joint_fit"]

    assert with_b["markets"]["sporttery"]["source"] == "Sporttery"
    assert with_b_fit["weights"]["sporttery_correct_score"] > 0
    assert with_b_fit["weights"]["sporttery_total_goals"] > 0
    assert (
        a_only_fit["lambda_home"],
        a_only_fit["lambda_away"],
        a_only_fit["rho"],
    ) != pytest.approx(
        (
            with_b_fit["lambda_home"],
            with_b_fit["lambda_away"],
            with_b_fit["rho"],
        )
    )
