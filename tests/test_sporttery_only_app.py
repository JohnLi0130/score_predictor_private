from __future__ import annotations

from pathlib import Path

import pytest

from score_predictor.history.store import list_predictions
from score_predictor.predictor import match_input_from_dict, predict
from score_predictor.ui.sporttery_only_helpers import (
    build_sporttery_prediction_context_key,
    get_canonical_top_score,
    normalize_sporttery_payload,
    normalize_sporttery_only_payload,
    run_sporttery_only_prediction,
)
from score_predictor.v3.market_calibration import build_market_probabilities


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _legacy_payload(*, with_history: bool = False) -> dict:
    odds_1x2 = {"home": 1.62, "draw": 3.32, "away": 4.75}
    if with_history:
        odds_1x2["history"] = [
            {"timestamp": "2026-06-12T08:00:00Z", "home": 1.78, "draw": 3.22, "away": 4.35},
            {"timestamp": "2026-06-12T14:00:00Z", "home": 1.62, "draw": 3.32, "away": 4.75},
        ]
    return {
        "match": {
            "match_id": "sporttery-only-a",
            "home_team": "Canada",
            "away_team": "Bosnia and Herzegovina",
            "competition": "World Cup",
            "stage": "Group",
            "kickoff_time": "2026-06-12 20:00",
            "timezone": "Asia/Shanghai",
            "venue": {"venue_type": "neutral"},
            "target": "90min_score",
        },
        "market": {
            "odds_1x2": odds_1x2,
            "rqspf": {"handicap": -1, "home": 3.11, "draw": 3.20, "away": 2.02},
            "correct_score_odds": {
                "0-0": 9.50,
                "1-0": 5.40,
                "1-1": 5.00,
                "2-0": 6.50,
                "2-1": 6.00,
                "0-1": 12.00,
                "home_other": 18.00,
                "draw_other": 28.00,
                "away_other": 35.00,
            },
            "sporttery_total_goals": {
                "odds": {
                    "0": 9.50,
                    "1": 4.30,
                    "2": 3.10,
                    "3": 3.60,
                    "4": 6.20,
                    "5": 12.50,
                    "6": 22.00,
                    "7+": 35.00,
                }
            },
            "half_full_time": {
                "HH": 2.51,
                "HD": 16.00,
                "HA": 36.00,
                "DH": 4.25,
                "DD": 4.80,
                "DA": 10.00,
                "AH": 25.00,
                "AD": 16.00,
                "AA": 8.40,
            },
        },
    }


def _new_structure_payload() -> dict:
    legacy = _legacy_payload()
    return {
        "match": legacy["match"],
        "markets": {
            "sporttery": {
                "source": "yaml",
                "provider": "sporttery",
                "weight": 1.0,
                "sporttery_1x2": legacy["market"]["odds_1x2"],
                "sporttery_handicap_3way": {
                    "line": -1,
                    "home_win": 3.11,
                    "draw": 3.20,
                    "away_win": 2.02,
                },
                "sporttery_correct_score": {
                    "scores": legacy["market"]["correct_score_odds"],
                },
                "sporttery_total_goals": legacy["market"]["sporttery_total_goals"],
                "sporttery_half_full": legacy["market"]["half_full_time"],
            }
        },
    }


def _result(payload: dict, **kwargs):
    return run_sporttery_only_prediction(payload, save_history=False, **kwargs)


def test_sporttery_only_prediction_does_not_need_external_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)

    output = _result(_legacy_payload())

    assert output["result"]["v3"]["top_scores"]
    assert output["payload"]["odds_channels"]["sporttery"]["role"] == "primary_calibration"


def test_app_source_hides_external_and_legacy_terms() -> None:
    source = (PROJECT_ROOT / "src/score_predictor/ui/sporttery_only_app.py").read_text(
        encoding="utf-8"
    )

    forbidden = [
        "国际赔率通道",
        "The Odds API",
        "API key",
        "bookmaker",
        "regions",
        "event_id",
        "Value Analysis",
        "breakeven",
    ]
    for term in forbidden:
        assert term not in source
    assert " Edge " not in source
    assert " EV " not in source


def test_current_compatible_sporttery_yaml_can_predict() -> None:
    output = _result(_legacy_payload())

    assert output["match_input"].sporttery_1x2 is not None
    assert output["result"]["v3"]["top_scores"]


def test_new_sporttery_market_structure_can_predict() -> None:
    output = _result(_new_structure_payload())

    assert output["match_input"].sporttery_total_goals_odds
    assert output["result"]["v3"]["top_scores"]


def test_normalize_sporttery_payload_recognizes_universal_1x2() -> None:
    payload = {
        "match": {"home_team": "United States", "away_team": "Paraguay"},
        "markets": {
            "sporttery": {
                "sporttery_1x2": {
                    "odds": {"home": 1.79, "draw": 3.25, "away": 3.80},
                    "history": [
                        {"timestamp": "2026-06-12T08:00:00Z", "home": 1.83, "draw": 3.20, "away": 3.70}
                    ],
                }
            }
        },
    }

    normalized = normalize_sporttery_payload(payload)
    one_x_two = normalized["markets"]["sporttery"]["sporttery_1x2"]

    assert one_x_two["odds"] == {"home": 1.79, "draw": 3.25, "away": 3.80}
    assert one_x_two["source_path"] == "markets.sporttery.sporttery_1x2"
    assert one_x_two["history"][0]["home"] == 1.83


def test_normalize_sporttery_payload_does_not_float_history_lists() -> None:
    payload = _legacy_payload(with_history=True)
    payload["market"]["correct_score_odds"] = {
        "scores": {"0-0": 9.5, "1-0": 5.4},
        "history": [{"scores": {"0-0": 9.7, "1-0": 5.8}}],
    }
    payload["market"]["sporttery_total_goals"]["history"] = [
        {"odds": {"0": 9.8, "1": 4.4, "2": 3.2}}
    ]

    normalized = normalize_sporttery_payload(payload)

    assert normalized["markets"]["sporttery"]["sporttery_1x2"]["history"]
    assert normalized["markets"]["sporttery"]["sporttery_correct_score"]["scores"]["0-0"] == 9.5
    assert normalized["markets"]["sporttery"]["sporttery_total_goals"]["history"]


def test_sporttery_only_prediction_uses_loaded_payload_not_manual_sample() -> None:
    source = (PROJECT_ROOT / "src/score_predictor/ui/sporttery_only_app.py").read_text(
        encoding="utf-8"
    )
    payload = _legacy_payload()
    payload["match"]["home_team"] = "United States"
    payload["match"]["away_team"] = "Paraguay"

    normalized = normalize_sporttery_only_payload(payload)

    assert normalized["match"]["home_team"] == "United States"
    assert normalized["match"]["away_team"] == "Paraguay"
    assert "payload = _load_yaml_from_text(st.session_state.get(\"sporttery_yaml_text\") or SAMPLE_YAML)" not in source
    assert "sporttery_normalized_payload" in source


def test_sporttery_1x2_is_primary_calibration_source() -> None:
    output = _result(_legacy_payload())
    market_probs = build_market_probabilities(output["match_input"])

    assert output["match_input"].odds_channels.sporttery.role == "primary_calibration"
    assert market_probs["one_x_two_sources"][0]["channel"] == "sporttery"
    assert market_probs["sporttery_market_status"]["sporttery_1x2"]["status"] == "primary_calibration"
    assert market_probs["weights"]["sporttery_1x2"] > 0.5


def test_sporttery_total_goals_and_correct_score_participate_in_v3_loss() -> None:
    output = _result(_legacy_payload())
    market_probs = build_market_probabilities(output["match_input"])

    assert market_probs["weights"]["sporttery_total_goals"] > 0
    assert market_probs["weights"]["sporttery_correct_score"] > 0
    assert market_probs["sporttery_total_goals"]
    assert any(source["channel"] == "sporttery" for source in market_probs["correct_score_sources"])
    assert "sporttery_total_goals" in output["result"]["v3"]["market_fit_errors"]


def test_sporttery_handicap_3way_is_three_outcome_market() -> None:
    output = _result(_legacy_payload())
    market_probs = build_market_probabilities(output["match_input"])
    handicap = market_probs["sporttery_handicap_3way"]

    assert set(handicap) >= {"line", "home", "draw", "away"}
    assert market_probs["weights"]["sporttery_handicap_3way"] > 0
    assert "sporttery_handicap_3way" in output["result"]["v3"]["market_fit_errors"]


def test_sporttery_half_full_is_audit_only_and_does_not_change_lambda() -> None:
    with_half_full = _legacy_payload()
    without_half_full = _legacy_payload()
    without_half_full["market"].pop("half_full_time")

    result_with = _result(with_half_full)["result"]["v3"]
    result_without = _result(without_half_full)["result"]["v3"]

    assert result_with["sporttery_market_status"]["sporttery_half_full"]["status"] == "audit_only"
    assert result_with["joint_fit"]["weights"]["sporttery_half_full"] == 0.0
    assert result_with["joint_fit"]["lambda_home"] == pytest.approx(
        result_without["joint_fit"]["lambda_home"]
    )
    assert result_with["joint_fit"]["lambda_away"] == pytest.approx(
        result_without["joint_fit"]["lambda_away"]
    )


def test_odds_movement_uses_history_devig_probability_delta() -> None:
    output = _result(_legacy_payload(with_history=True))
    movement = output["result"]["v3"]["odds_movement"]["markets"]["sporttery_1x2"]

    assert movement["outcomes"]["home"]["latest_devig_prob"] > movement["outcomes"]["home"]["open_devig_prob"]
    assert movement["outcomes"]["home"]["prob_delta"] == pytest.approx(
        movement["outcomes"]["home"]["latest_devig_prob"]
        - movement["outcomes"]["home"]["open_devig_prob"]
    )


def test_odds_movement_affect_lambda_adjusts_within_clamp() -> None:
    output = _result(
        _legacy_payload(with_history=True),
        movement_settings_overrides={
            "affect_lambda": True,
            "max_lambda_adjustment": 0.01,
            "max_total_lambda_adjustment": 0.015,
            "max_rho_adjustment": 0.005,
            "movement_weights": {
                "sporttery_1x2_movement": 2.0,
                "sporttery_total_goals_movement": 2.0,
                "sporttery_correct_score_movement": 2.0,
                "sporttery_handicap_3way_movement": 2.0,
            },
        },
    )
    movement = output["result"]["v3"]["movement_adjustment"]

    assert movement["applied"] is True
    assert abs(movement["home_adjustment_pct"]) <= 0.01 + 1e-12
    assert abs(movement["away_adjustment_pct"]) <= 0.01 + 1e-12
    assert abs(movement["total_adjustment_pct"]) <= 0.015 + 1e-12
    assert abs(movement["rho_adjustment"]) <= 0.005 + 1e-12


def test_canonical_top_score_matches_v3_output() -> None:
    result = _result(_legacy_payload())["result"]

    assert get_canonical_top_score(result) == result["v3"]["top_scores"][0]


def test_prediction_history_upsert_uses_sporttery_context_key(tmp_path: Path) -> None:
    db_path = tmp_path / "predictions.sqlite"
    payload = _legacy_payload()
    first = run_sporttery_only_prediction(payload, db_path=db_path)
    second = run_sporttery_only_prediction(payload, db_path=db_path)
    changed = _legacy_payload()
    changed["market"]["odds_1x2"]["home"] = 1.72
    run_sporttery_only_prediction(changed, db_path=db_path)

    rows = list_predictions(db_path)

    assert first["context_key"] == second["context_key"]
    assert second["history_record"]["run_count"] == 2
    assert len(rows) == 2


def test_context_key_changes_when_prematch_or_settings_change() -> None:
    base = normalize_sporttery_only_payload(_legacy_payload())
    with_context = normalize_sporttery_only_payload(
        _legacy_payload(),
        prematch_context={"weather": {"temperature_c": 34}},
    )
    with_settings = normalize_sporttery_only_payload(
        _legacy_payload(),
        settings_overrides={"sporttery_total_goals_weight": 0.65},
    )

    assert build_sporttery_prediction_context_key(base) != build_sporttery_prediction_context_key(
        with_context,
        prematch_context=with_context.get("prematch_context"),
    )
    assert build_sporttery_prediction_context_key(base) != build_sporttery_prediction_context_key(with_settings)


def test_primary_sporttery_payload_reuses_v3_core() -> None:
    normalized = normalize_sporttery_only_payload(_legacy_payload())
    match_input = match_input_from_dict(normalized)
    result = predict(match_input)

    assert result["v3"]["enabled"] is True
    assert result["v3"]["joint_fit"]["weights"]["sporttery_1x2"] > 0
