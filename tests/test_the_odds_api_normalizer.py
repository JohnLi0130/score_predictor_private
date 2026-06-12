from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from score_predictor.connectors.odds_api_normalizer import (
    market_keys_for_mode,
    normalize_event_odds_to_v3_input,
    select_bookmaker,
    selectable_market_keys,
)
from score_predictor.predictor import match_input_from_dict, predict
from score_predictor.ui.yaml_io import load_yaml_to_form_state


def _event_odds() -> dict:
    return {
        "id": "evt_1",
        "home_team": "Korea Republic",
        "away_team": "Czech Republic",
        "commence_time": "2026-06-12T12:00:00Z",
        "bookmakers": [
            {
                "key": "bet365",
                "title": "Bet365",
                "last_update": "2026-06-11T12:00:00Z",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Korea Republic", "price": 2.3},
                            {"name": "Draw", "price": 3.2},
                            {"name": "Czech Republic", "price": 3.0},
                        ],
                    }
                ],
            },
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "last_update": "2026-06-11T12:05:00Z",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Korea Republic", "price": 2.25},
                            {"name": "Draw", "price": 3.25},
                            {"name": "Czech Republic", "price": 3.1},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": 1.95, "point": 2.5},
                            {"name": "Under", "price": 1.90, "point": 2.5},
                        ],
                    },
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Korea Republic", "price": 1.91, "point": -0.25},
                            {"name": "Czech Republic", "price": 1.93, "point": 0.25},
                        ],
                    },
                    {
                        "key": "alternate_totals",
                        "outcomes": [
                            {"name": "Over", "price": 1.72, "point": 1.5},
                            {"name": "Under", "price": 2.12, "point": 1.5},
                            {"name": "Over", "price": 2.35, "point": 3.5},
                            {"name": "Under", "price": 1.60, "point": 3.5},
                        ],
                    },
                    {
                        "key": "btts",
                        "outcomes": [
                            {"name": "Yes", "price": 1.80},
                            {"name": "No", "price": 2.02},
                        ],
                    },
                    {
                        "key": "alternate_spreads",
                        "outcomes": [
                            {"name": "Korea Republic", "price": 2.10, "point": -0.5},
                            {"name": "Czech Republic", "price": 1.78, "point": 0.5},
                        ],
                    },
                    {
                        "key": "draw_no_bet",
                        "outcomes": [
                            {"name": "Korea Republic", "price": 1.62},
                            {"name": "Czech Republic", "price": 2.28},
                        ],
                    },
                    {
                        "key": "team_totals",
                        "outcomes": [
                            {"name": "Over", "description": "Korea Republic", "price": 1.88, "point": 1.5},
                            {"name": "Under", "description": "Korea Republic", "price": 1.95, "point": 1.5},
                            {"name": "Over", "description": "Czech Republic", "price": 2.25, "point": 0.5},
                            {"name": "Under", "description": "Czech Republic", "price": 1.65, "point": 0.5},
                        ],
                    },
                ],
            },
        ],
    }


def test_bookmaker_auto_priority_prefers_pinnacle() -> None:
    selected, warnings = select_bookmaker(_event_odds(), bookmaker="auto")

    assert warnings == []
    assert selected["key"] == "pinnacle"


def test_normalizer_parses_h2h_totals_and_spreads() -> None:
    normalized = normalize_event_odds_to_v3_input(
        _event_odds(),
        sport_key="soccer_fifa_world_cup",
        event_id="evt_1",
        bookmaker="auto",
    )
    summary = normalized["summary"]

    assert summary["selected_1x2"] == {"home": 2.25, "draw": 3.25, "away": 3.1}
    assert summary["selected_over_under"][0]["line"] == 2.5
    assert summary["selected_over_under"][0]["over_odds"] == 1.95
    assert summary["selected_asian_handicap"][0]["line"] == -0.25
    assert summary["selected_asian_handicap"][0]["home_odds"] == 1.91


def test_available_markets_modes_include_primary_and_audit_only_keys() -> None:
    available = [
        "h2h",
        "spreads",
        "totals",
        "alternate_totals",
        "btts",
        "alternate_spreads",
        "draw_no_bet",
        "team_totals",
        "correct_score_exact",
        "totals_h1",
        "alternate_totals_corners",
    ]

    assert market_keys_for_mode("完整建模模式") == [
        "h2h",
        "h2h_3_way",
        "spreads",
        "totals",
        "alternate_totals",
        "btts",
    ]
    selectable = selectable_market_keys(available)
    assert "alternate_totals" in selectable
    assert "btts" in selectable
    assert "alternate_spreads" in selectable
    assert "correct_score_exact" in selectable
    assert "totals_h1" not in selectable
    assert "alternate_totals_corners" not in selectable


def test_normalizer_parses_btts_and_alternate_totals() -> None:
    normalized = normalize_event_odds_to_v3_input(
        _event_odds(),
        sport_key="soccer_fifa_world_cup",
        event_id="evt_1",
        bookmaker="pinnacle",
    )

    payload = normalized["payload"]
    summary = normalized["summary"]

    assert summary["selected_btts"] == {"yes": 1.80, "no": 2.02}
    assert len(summary["selected_alternate_totals"]) == 2
    assert payload["market"]["btts"] == {"yes": 1.80, "no": 2.02}
    assert {row["line"] for row in payload["market"]["over_under_markets"]} == {1.5, 2.5, 3.5}
    assert payload["markets"]["international"]["alternate_totals"][0]["bookmaker"] == "pinnacle"


def test_normalizer_puts_secondary_markets_in_audit_only_bucket() -> None:
    normalized = normalize_event_odds_to_v3_input(
        _event_odds(),
        sport_key="soccer_fifa_world_cup",
        event_id="evt_1",
        bookmaker="pinnacle",
    )

    audit = normalized["payload"]["markets"]["international"]["audit_markets"]

    assert audit["alternate_spreads"][0]["line"] == -0.5
    assert audit["draw_no_bet"]["home"] == 1.62
    assert {row["team"] for row in audit["team_totals"]} == {"home", "away"}


def test_secondary_audit_markets_do_not_change_lambda() -> None:
    payload = normalize_event_odds_to_v3_input(
        _event_odds(),
        sport_key="soccer_fifa_world_cup",
        event_id="evt_1",
        bookmaker="pinnacle",
    )["payload"]
    without_audit = deepcopy(payload)
    without_audit["markets"]["international"]["audit_markets"] = {}

    result_with_audit = predict(match_input_from_dict(payload))["v3"]["joint_fit"]
    result_without_audit = predict(match_input_from_dict(without_audit))["v3"]["joint_fit"]

    assert result_with_audit["lambda_home"] == pytest.approx(result_without_audit["lambda_home"])
    assert result_with_audit["lambda_away"] == pytest.approx(result_without_audit["lambda_away"])
    assert result_with_audit["rho"] == pytest.approx(result_without_audit["rho"])


def test_normalizer_parses_correct_score_only_when_market_returned() -> None:
    event = _event_odds()
    event["bookmakers"][1]["markets"].append(
        {
            "key": "correct_score_exact",
            "outcomes": [
                {"name": "1-0", "price": 7.5},
                {"name": "Draw 1:1", "price": 6.2},
                {"name": "Other", "price": 40.0},
            ],
        }
    )

    normalized = normalize_event_odds_to_v3_input(
        event,
        sport_key="soccer_fifa_world_cup",
        event_id="evt_1",
        bookmaker="auto",
    )

    assert normalized["summary"]["selected_correct_score"] == {"1-0": 7.5, "1-1": 6.2}
    assert normalized["payload"]["markets"]["international"]["correct_score_odds"] == {
        "1-0": 7.5,
        "1-1": 6.2,
    }


def test_selected_bookmaker_market_fallback_warning() -> None:
    normalized = normalize_event_odds_to_v3_input(
        _event_odds(),
        sport_key="soccer_fifa_world_cup",
        event_id="evt_1",
        bookmaker="bet365",
    )

    assert normalized["summary"]["selected_bookmaker"] == "bet365"
    assert "totals" in normalized["summary"]["fallback_used"]
    assert any("缺少 totals" in warning for warning in normalized["warnings"])


def test_output_yaml_can_be_read_by_v3_predictor() -> None:
    payload = normalize_event_odds_to_v3_input(
        _event_odds(),
        sport_key="soccer_fifa_world_cup",
        event_id="evt_1",
        bookmaker="auto",
    )["payload"]

    match_input = match_input_from_dict(payload)
    result = predict(match_input)

    assert match_input.settings.market_only_mode is True
    assert match_input.odds_channels.international.role == "primary_calibration"
    assert match_input.market_roles.calibration_sources == ["international", "the_odds_api"]
    assert result["v3"]["enabled"] is True
    assert result["v3"]["top_scores"]


def test_ui_form_conversion_accepts_generated_yaml() -> None:
    payload = normalize_event_odds_to_v3_input(
        _event_odds(),
        sport_key="soccer_fifa_world_cup",
        event_id="evt_1",
        bookmaker="auto",
    )["payload"]
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)

    state = load_yaml_to_form_state(text)

    assert state["home_team"] == "Korea Republic"
    assert state["away_team"] == "Czech Republic"
    assert "the_odds_api" in state["calibration_sources"]
