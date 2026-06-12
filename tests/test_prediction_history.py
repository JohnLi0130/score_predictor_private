from __future__ import annotations

import csv
import io
from pathlib import Path

from score_predictor.history.store import (
    build_prediction_record,
    clear_history,
    delete_prediction,
    export_predictions_csv,
    get_prediction_detail,
    list_latest_by_match,
    list_predictions,
    save_prediction_history,
)
from score_predictor.predictor import match_input_from_dict, predict
from score_predictor.ui.streamlit_app import build_prediction_context_key


def _payload(match_id: str, home: str, away: str, home_odds: float = 1.85) -> dict:
    return {
        "match": {
            "match_id": match_id,
            "home_team": home,
            "away_team": away,
            "competition": "World Cup",
            "stage": "Group",
            "kickoff_time": "2026-06-12 20:00",
            "venue": {"venue_type": "neutral"},
            "target": "90min_score",
        },
        "market": {
            "odds_1x2": {"home": home_odds, "draw": 3.45, "away": 4.40},
            "over_under": {"line": 2.5, "over_odds": 1.90, "under_odds": 1.95},
        },
        "settings": {"market_only_mode": True, "max_goals": 8, "dc_enabled": True},
        "internal_model": {"home_lambda": 1.2, "away_lambda": 1.0},
    }


def _prediction(payload: dict) -> dict:
    return predict(match_input_from_dict(payload))


def _context(payload: dict) -> dict:
    return {
        "prediction_context_key": build_prediction_context_key(payload),
        "app_mode": "dual_channel",
        "source_mode": "dual_channel",
    }


def test_successful_prediction_creates_history_record(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "predictions.sqlite"
    payload = _payload("m1", "Canada", "Bosnia")
    result = _prediction(payload)

    saved = save_prediction_history(result, payload, payload["settings"], _context(payload), db_path)

    assert db_path.exists()
    assert saved["match_id"] == "m1"
    assert saved["home_team"] == "Canada"
    assert saved["top_scores_json"]


def test_same_prediction_context_updates_and_increments_run_count(tmp_path: Path) -> None:
    db_path = tmp_path / "predictions.sqlite"
    payload = _payload("m1", "Canada", "Bosnia")
    result = _prediction(payload)
    context = _context(payload)

    first = save_prediction_history(result, payload, payload["settings"], context, db_path)
    second = save_prediction_history(result, payload, payload["settings"], context, db_path)
    rows = list_predictions(db_path)

    assert len(rows) == 1
    assert second["prediction_id"] == first["prediction_id"]
    assert second["created_at"] == first["created_at"]
    assert second["run_count"] == 2
    assert second["save_action"] == "updated"


def test_different_prediction_context_inserts_new_version(tmp_path: Path) -> None:
    db_path = tmp_path / "predictions.sqlite"
    payload_a = _payload("m1", "Canada", "Bosnia", home_odds=1.85)
    payload_b = _payload("m1", "Canada", "Bosnia", home_odds=1.95)

    save_prediction_history(_prediction(payload_a), payload_a, payload_a["settings"], _context(payload_a), db_path)
    save_prediction_history(_prediction(payload_b), payload_b, payload_b["settings"], _context(payload_b), db_path)

    assert len(list_predictions(db_path)) == 2


def test_list_predictions_orders_by_updated_at_desc(tmp_path: Path) -> None:
    db_path = tmp_path / "predictions.sqlite"
    payload_a = _payload("m1", "Canada", "Bosnia")
    payload_b = _payload("m2", "Mexico", "South Africa")
    record_a = build_prediction_record(_prediction(payload_a), payload_a, payload_a["settings"], _context(payload_a))
    record_b = build_prediction_record(_prediction(payload_b), payload_b, payload_b["settings"], _context(payload_b))

    from score_predictor.history.store import upsert_prediction

    upsert_prediction(record_a, db_path, now="2026-06-12T10:00:00+00:00")
    upsert_prediction(record_b, db_path, now="2026-06-12T11:00:00+00:00")

    rows = list_predictions(db_path)

    assert [row["match_id"] for row in rows] == ["m2", "m1"]


def test_list_latest_by_match_returns_only_latest_per_match(tmp_path: Path) -> None:
    db_path = tmp_path / "predictions.sqlite"
    payload_a = _payload("m1", "Canada", "Bosnia", home_odds=1.85)
    payload_b = _payload("m1", "Canada", "Bosnia", home_odds=1.95)
    payload_c = _payload("m2", "Mexico", "South Africa")

    save_prediction_history(_prediction(payload_a), payload_a, payload_a["settings"], _context(payload_a), db_path)
    save_prediction_history(_prediction(payload_b), payload_b, payload_b["settings"], _context(payload_b), db_path)
    save_prediction_history(_prediction(payload_c), payload_c, payload_c["settings"], _context(payload_c), db_path)

    rows = list_latest_by_match(db_path)

    assert len(rows) == 2
    assert {row["match_id"] for row in rows} == {"m1", "m2"}


def test_get_prediction_detail_returns_full_json_fields(tmp_path: Path) -> None:
    db_path = tmp_path / "predictions.sqlite"
    payload = _payload("m1", "Canada", "Bosnia")
    saved = save_prediction_history(_prediction(payload), payload, payload["settings"], _context(payload), db_path)

    detail = get_prediction_detail(saved["prediction_id"], db_path)

    assert detail["raw_result_json"]["v3"]["top_scores"]
    assert isinstance(detail["warnings_json"], list)


def test_history_record_saves_dashboard_snapshot_and_modes(tmp_path: Path) -> None:
    db_path = tmp_path / "predictions.sqlite"
    payload = _payload("m1", "Canada", "Bosnia")
    saved = save_prediction_history(_prediction(payload), payload, payload["settings"], _context(payload), db_path)
    detail = get_prediction_detail(saved["prediction_id"], db_path)

    assert detail["app_mode"] == "dual_channel"
    assert detail["source_mode"] == "dual_channel"
    assert detail["dashboard_snapshot_json"]["prediction_summary"]["top_scores"]
    assert detail["top_scores_json"]
    assert detail["score_matrix_json"]


def test_history_list_filters_by_app_mode(tmp_path: Path) -> None:
    db_path = tmp_path / "predictions.sqlite"
    payload = _payload("m1", "Canada", "Bosnia")
    sporttery_context = {
        **_context(payload),
        "prediction_context_key": "sporttery-context",
        "app_mode": "sporttery_only",
        "source_mode": "sporttery_only",
    }
    save_prediction_history(_prediction(payload), payload, payload["settings"], _context(payload), db_path)
    save_prediction_history(_prediction(payload), payload, payload["settings"], sporttery_context, db_path)

    assert len(list_predictions(db_path, app_mode="dual_channel")) == 1
    assert len(list_predictions(db_path, app_mode="sporttery_only")) == 1


def test_history_detail_upsert_returns_latest_snapshot(tmp_path: Path) -> None:
    db_path = tmp_path / "predictions.sqlite"
    payload = _payload("m1", "Canada", "Bosnia")
    context = _context(payload)
    result_a = _prediction(payload)
    result_b = _prediction(payload)
    result_b["v3"]["top_scores"][0]["score"] = "9-9"

    first = save_prediction_history(result_a, payload, payload["settings"], context, db_path)
    second = save_prediction_history(result_b, payload, payload["settings"], context, db_path)
    detail = get_prediction_detail(first["prediction_id"], db_path)

    assert second["run_count"] == 2
    assert detail["prediction_id"] == first["prediction_id"]
    assert detail["dashboard_snapshot_json"]["prediction_summary"]["top_scores"][0]["score"] == "9-9"


def test_history_saves_real_teams_from_result_not_default_payload() -> None:
    payload = _payload("m1", "Home FC", "Away FC")
    result = _prediction(_payload("m1", "United States", "Paraguay"))
    record = build_prediction_record(result, payload, payload["settings"], _context(payload))

    assert record["home_team"] == "United States"
    assert record["away_team"] == "Paraguay"


def test_history_detail_component_is_shared_by_both_apps() -> None:
    root = Path(__file__).resolve().parents[1]
    component = (root / "src/score_predictor/ui/history_components.py").read_text(encoding="utf-8")
    main_app = (root / "src/score_predictor/ui/streamlit_app.py").read_text(encoding="utf-8")
    sporttery_app = (root / "src/score_predictor/ui/sporttery_only_app.py").read_text(encoding="utf-8")

    for label in ("Top 5", "Top 10", "胜平负", "总进球", "BTTS", "lambda", "风险提示"):
        assert label in component
    assert "render_prediction_history_tab" in main_app
    assert "render_prediction_history_tab" in sporttery_app


def test_delete_prediction_removes_single_record(tmp_path: Path) -> None:
    db_path = tmp_path / "predictions.sqlite"
    payload = _payload("m1", "Canada", "Bosnia")
    saved = save_prediction_history(_prediction(payload), payload, payload["settings"], _context(payload), db_path)

    deleted = delete_prediction(saved["prediction_id"], db_path)

    assert deleted == 1
    assert list_predictions(db_path) == []


def test_clear_history_removes_all_records(tmp_path: Path) -> None:
    db_path = tmp_path / "predictions.sqlite"
    for payload in (
        _payload("m1", "Canada", "Bosnia"),
        _payload("m2", "Mexico", "South Africa"),
    ):
        save_prediction_history(_prediction(payload), payload, payload["settings"], _context(payload), db_path)

    cleared = clear_history(db_path)

    assert cleared == 2
    assert list_predictions(db_path) == []


def test_export_predictions_csv_has_required_fields(tmp_path: Path) -> None:
    db_path = tmp_path / "predictions.sqlite"
    payload = _payload("m1", "Canada", "Bosnia")
    save_prediction_history(_prediction(payload), payload, payload["settings"], _context(payload), db_path)

    csv_text = export_predictions_csv(list_predictions(db_path))
    row = next(csv.DictReader(io.StringIO(csv_text)))

    assert row["updated_at"]
    assert row["match_id"] == "m1"
    assert row["top_score"]
    assert row["prediction_context_key"]
    assert row["run_count"] == "1"


def test_two_different_matches_are_both_kept_in_history(tmp_path: Path) -> None:
    db_path = tmp_path / "predictions.sqlite"
    payload_a = _payload("m1", "Canada", "Bosnia")
    payload_b = _payload("m2", "Mexico", "South Africa")

    save_prediction_history(_prediction(payload_a), payload_a, payload_a["settings"], _context(payload_a), db_path)
    save_prediction_history(_prediction(payload_b), payload_b, payload_b["settings"], _context(payload_b), db_path)

    assert {row["match_id"] for row in list_predictions(db_path)} == {"m1", "m2"}


def test_same_context_twice_has_one_record_with_run_count_two(tmp_path: Path) -> None:
    db_path = tmp_path / "predictions.sqlite"
    payload = _payload("m1", "Canada", "Bosnia")
    result = _prediction(payload)
    context = _context(payload)

    save_prediction_history(result, payload, payload["settings"], context, db_path)
    save_prediction_history(result, payload, payload["settings"], context, db_path)

    rows = list_predictions(db_path)
    assert len(rows) == 1
    assert rows[0]["run_count"] == 2
