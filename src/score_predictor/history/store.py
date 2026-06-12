from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HISTORY_DB_PATH = PROJECT_ROOT / "data" / "prediction_history" / "predictions.sqlite"
APP_VERSION = "prediction_history_v1"


JSON_COLUMNS = {
    "data_sources_json",
    "settings_json",
    "input_summary_json",
    "probabilities_1x2_json",
    "top_scores_json",
    "total_goals_distribution_json",
    "over_under_probabilities_json",
    "btts_probabilities_json",
    "odds_movement_summary_json",
    "movement_adjustment_json",
    "market_quality_json",
    "warnings_json",
    "raw_result_json",
}


SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT NOT NULL,
    prediction_context_key TEXT NOT NULL UNIQUE,
    match_id TEXT,
    event_id TEXT,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    competition TEXT,
    stage TEXT,
    kickoff_time TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    run_count INTEGER NOT NULL DEFAULT 1,
    input_hash TEXT,
    international_payload_hash TEXT,
    sporttery_payload_hash TEXT,
    prematch_context_hash TEXT,
    model_settings_hash TEXT,
    data_sources_json TEXT,
    settings_json TEXT,
    input_summary_json TEXT,
    lambda_home REAL,
    lambda_away REAL,
    rho REAL,
    lambda_home_before_movement REAL,
    lambda_away_before_movement REAL,
    rho_before_movement REAL,
    probabilities_1x2_json TEXT,
    top_scores_json TEXT,
    total_goals_distribution_json TEXT,
    over_under_probabilities_json TEXT,
    btts_probabilities_json TEXT,
    confidence_score REAL,
    risk_level TEXT,
    data_completeness_score REAL,
    market_consistency_score REAL,
    odds_movement_summary_json TEXT,
    movement_adjustment_json TEXT,
    market_quality_json TEXT,
    warnings_json TEXT,
    raw_result_json TEXT,
    app_version TEXT,
    actual_home_goals_90 INTEGER,
    actual_away_goals_90 INTEGER,
    actual_result_90 TEXT,
    actual_total_goals_90 INTEGER,
    actual_btts INTEGER,
    settled_at TEXT,
    settlement_notes TEXT
);
"""


INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_predictions_match_id ON predictions(match_id)",
    "CREATE INDEX IF NOT EXISTS idx_predictions_updated_at ON predictions(updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_predictions_teams ON predictions(home_team, away_team)",
]


CSV_FIELDS = [
    "updated_at",
    "match_id",
    "home_team",
    "away_team",
    "kickoff_time",
    "top_score",
    "home_win_prob",
    "draw_prob",
    "away_win_prob",
    "lambda_home",
    "lambda_away",
    "rho",
    "confidence_score",
    "risk_level",
    "data_sources",
    "prediction_context_key",
    "run_count",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_default(value: Any) -> str:
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True, default=_json_default)


def _json_loads(value: Any) -> Any:
    if value in (None, ""):
        return {}
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return value


def _hash(value: Any) -> str:
    return hashlib.sha256(_json_dumps(value).encode("utf-8")).hexdigest()


def _connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path or DEFAULT_HISTORY_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(SCHEMA)
    for statement in INDEXES:
        conn.execute(statement)
    conn.commit()


def _split_match(match_name: Any) -> tuple[str, str]:
    text = str(match_name or "")
    if " vs " in text:
        home, away = text.split(" vs ", 1)
        return home or "Home", away or "Away"
    return "Home", "Away"


def _match_payload(input_payload: dict[str, Any]) -> dict[str, Any]:
    return input_payload.get("match") if isinstance(input_payload.get("match"), dict) else {}


def _payload_markets(input_payload: dict[str, Any]) -> dict[str, Any]:
    return input_payload.get("markets") if isinstance(input_payload.get("markets"), dict) else {}


def _extract_match_info(
    prediction_result: dict[str, Any],
    input_payload: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    match = _match_payload(input_payload)
    home_from_result, away_from_result = _split_match(prediction_result.get("match"))
    return {
        "match_id": match.get("match_id") or input_payload.get("match_id") or context.get("match_id"),
        "event_id": match.get("event_id") or input_payload.get("event_id") or context.get("event_id"),
        "home_team": match.get("home_team") or context.get("home_team") or home_from_result,
        "away_team": match.get("away_team") or context.get("away_team") or away_from_result,
        "competition": match.get("competition") or input_payload.get("competition"),
        "stage": match.get("stage") or input_payload.get("stage"),
        "kickoff_time": prediction_result.get("kickoff_time")
        or match.get("commence_time_utc")
        or match.get("kickoff_time_beijing")
        or match.get("commence_time")
        or match.get("kickoff_time")
        or match.get("date"),
    }


def _final_score_records(v3: dict[str, Any]) -> list[dict[str, Any]]:
    return list(v3.get("final_score_matrix") or [])


def _total_goals_distribution(score_records: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in score_records:
        try:
            total = int(row.get("home_goals", 0)) + int(row.get("away_goals", 0))
            probability = float(row.get("prob", row.get("probability", 0.0)))
        except (TypeError, ValueError):
            continue
        key = "7+" if total >= 7 else str(total)
        totals[key] = totals.get(key, 0.0) + probability
    return totals


def _risk_level(warnings: list[Any]) -> str:
    text = " ".join(str(item) for item in warnings)
    if "conflict" in text or "optimizer_fallback" in text or "sensitive" in text:
        return "high"
    if "missing" in text or "margin" in text or "not_stable" in text:
        return "medium"
    return "low"


def _data_sources(input_payload: dict[str, Any], prediction_result: dict[str, Any]) -> dict[str, Any]:
    markets = _payload_markets(input_payload)
    v3 = prediction_result.get("v3") or {}
    return {
        "odds_channels": input_payload.get("odds_channels") or {},
        "market_roles": input_payload.get("market_roles") or {},
        "international": markets.get("international") or {},
        "sporttery": markets.get("sporttery") or {},
        "sporttery_market_status": v3.get("sporttery_market_status") or {},
    }


def build_prediction_record(
    prediction_result: dict[str, Any],
    input_payload: dict[str, Any],
    settings: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = dict(context or {})
    settings = dict(settings or input_payload.get("settings") or {})
    v3 = prediction_result.get("v3") or {}
    flow = v3.get("lambda_flow") or {}
    fit = v3.get("joint_fit") or {}
    confidence = v3.get("confidence") or {}
    probabilities = v3.get("probabilities") or {}
    warnings = list(dict.fromkeys((prediction_result.get("warnings") or []) + (v3.get("risk_warnings") or [])))
    match_info = _extract_match_info(prediction_result, input_payload, context)
    markets = _payload_markets(input_payload)
    international_payload = (
        context.get("international_payload")
        or markets.get("international")
        or input_payload.get("market")
        or {}
    )
    sporttery_payload = context.get("sporttery_payload") or markets.get("sporttery") or {}
    prematch_context = context.get("prematch_context") or input_payload.get("prematch_context") or {}
    context_key = context.get("prediction_context_key") or context.get("context_key")
    if not context_key:
        context_key = _hash(
            {
                "home_team": match_info["home_team"],
                "away_team": match_info["away_team"],
                "match_id": match_info["match_id"],
                "event_id": match_info["event_id"],
                "international_payload_hash": _hash(international_payload),
                "sporttery_payload_hash": _hash(sporttery_payload),
                "prematch_context_hash": _hash(prematch_context),
                "model_settings_hash": _hash(settings),
            }
        )
    score_records = _final_score_records(v3)
    movement = v3.get("movement_adjustment") or {}
    market_quality = v3.get("market_quality") or {}
    return {
        "prediction_id": context.get("prediction_id") or f"pred_{uuid4().hex}",
        "prediction_context_key": context_key,
        **match_info,
        "input_hash": context.get("input_hash") or _hash(input_payload),
        "international_payload_hash": context.get("international_payload_hash") or _hash(international_payload),
        "sporttery_payload_hash": context.get("sporttery_payload_hash") or _hash(sporttery_payload),
        "prematch_context_hash": context.get("prematch_context_hash") or _hash(prematch_context),
        "model_settings_hash": context.get("model_settings_hash") or _hash(settings),
        "data_sources_json": _data_sources(input_payload, prediction_result),
        "settings_json": settings,
        "input_summary_json": {
            "match": _match_payload(input_payload),
            "prediction_time": input_payload.get("prediction_time"),
            "notes": input_payload.get("notes") or [],
        },
        "lambda_home": flow.get("final_lambda_home"),
        "lambda_away": flow.get("final_lambda_away"),
        "rho": fit.get("rho"),
        "lambda_home_before_movement": movement.get("lambda_home_before")
        or flow.get("lambda_home_before_movement"),
        "lambda_away_before_movement": movement.get("lambda_away_before")
        or flow.get("lambda_away_before_movement"),
        "rho_before_movement": movement.get("rho_before") or fit.get("market_rho_before_movement"),
        "probabilities_1x2_json": probabilities.get("one_x_two") or {},
        "top_scores_json": v3.get("top_scores") or prediction_result.get("top_scores") or [],
        "total_goals_distribution_json": _total_goals_distribution(score_records),
        "over_under_probabilities_json": probabilities.get("over_under") or {},
        "btts_probabilities_json": probabilities.get("btts") or {},
        "confidence_score": confidence.get("final_confidence_score"),
        "risk_level": _risk_level(warnings),
        "data_completeness_score": confidence.get("data_quality_score"),
        "market_consistency_score": confidence.get("market_consistency_score"),
        "odds_movement_summary_json": v3.get("odds_movement") or {},
        "movement_adjustment_json": movement,
        "market_quality_json": market_quality,
        "warnings_json": warnings,
        "raw_result_json": prediction_result,
        "app_version": APP_VERSION,
        "actual_home_goals_90": None,
        "actual_away_goals_90": None,
        "actual_result_90": None,
        "actual_total_goals_90": None,
        "actual_btts": None,
        "settled_at": None,
        "settlement_notes": None,
    }


def _serialize_record(record: dict[str, Any]) -> dict[str, Any]:
    serialized = dict(record)
    for column in JSON_COLUMNS:
        serialized[column] = _json_dumps(serialized.get(column))
    return serialized


def _row_to_dict(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    data = dict(row)
    for column in JSON_COLUMNS:
        data[column] = _json_loads(data.get(column))
    return data


def upsert_prediction(
    record: dict[str, Any],
    db_path: Path | str | None = None,
    *,
    now: str | None = None,
) -> dict[str, Any]:
    timestamp = now or _now_iso()
    serialized = _serialize_record(record)
    context_key = str(serialized["prediction_context_key"])
    with _connect(db_path) as conn:
        existing = conn.execute(
            "SELECT id, prediction_id, created_at, run_count FROM predictions WHERE prediction_context_key = ?",
            (context_key,),
        ).fetchone()
        columns = [
            "prediction_id",
            "prediction_context_key",
            "match_id",
            "event_id",
            "home_team",
            "away_team",
            "competition",
            "stage",
            "kickoff_time",
            "input_hash",
            "international_payload_hash",
            "sporttery_payload_hash",
            "prematch_context_hash",
            "model_settings_hash",
            "data_sources_json",
            "settings_json",
            "input_summary_json",
            "lambda_home",
            "lambda_away",
            "rho",
            "lambda_home_before_movement",
            "lambda_away_before_movement",
            "rho_before_movement",
            "probabilities_1x2_json",
            "top_scores_json",
            "total_goals_distribution_json",
            "over_under_probabilities_json",
            "btts_probabilities_json",
            "confidence_score",
            "risk_level",
            "data_completeness_score",
            "market_consistency_score",
            "odds_movement_summary_json",
            "movement_adjustment_json",
            "market_quality_json",
            "warnings_json",
            "raw_result_json",
            "app_version",
            "actual_home_goals_90",
            "actual_away_goals_90",
            "actual_result_90",
            "actual_total_goals_90",
            "actual_btts",
            "settled_at",
            "settlement_notes",
        ]
        if existing:
            serialized["prediction_id"] = existing["prediction_id"]
            update_columns = [column for column in columns if column != "prediction_context_key"]
            assignments = ", ".join(f"{column} = ?" for column in update_columns)
            values = [serialized.get(column) for column in update_columns]
            values.extend([timestamp, int(existing["run_count"]) + 1, context_key])
            conn.execute(
                f"UPDATE predictions SET {assignments}, updated_at = ?, run_count = ? WHERE prediction_context_key = ?",
                values,
            )
            action = "updated"
        else:
            insert_columns = columns + ["created_at", "updated_at", "run_count"]
            placeholders = ", ".join("?" for _ in insert_columns)
            values = [serialized.get(column) for column in columns] + [timestamp, timestamp, 1]
            conn.execute(
                f"INSERT INTO predictions ({', '.join(insert_columns)}) VALUES ({placeholders})",
                values,
            )
            action = "inserted"
        conn.commit()
        saved = get_prediction_detail(context_key, db_path=db_path, by_context_key=True)
    saved["save_action"] = action
    return saved


def save_prediction_history(
    prediction_result: dict[str, Any],
    input_payload: dict[str, Any],
    settings: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    record = build_prediction_record(prediction_result, input_payload, settings, context)
    return upsert_prediction(record, db_path=db_path)


def list_predictions(
    db_path: Path | str | None = None,
    *,
    search: str | None = None,
    match_id: str | None = None,
    limit: int = 100,
    latest_only: bool = False,
) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = [
            _row_to_dict(row)
            for row in conn.execute(
                "SELECT * FROM predictions ORDER BY updated_at DESC, id DESC"
            ).fetchall()
        ]
    if search:
        needle = search.casefold()
        rows = [
            row
            for row in rows
            if needle in str(row.get("home_team", "")).casefold()
            or needle in str(row.get("away_team", "")).casefold()
        ]
    if match_id:
        rows = [row for row in rows if str(row.get("match_id") or "") == str(match_id)]
    if latest_only:
        rows = _latest_rows_by_match(rows)
    return rows[: max(1, int(limit))]


def _latest_rows_by_match(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    latest: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("match_id") or f"{row.get('home_team')} vs {row.get('away_team')}")
        if key in seen:
            continue
        seen.add(key)
        latest.append(row)
    return latest


def list_latest_by_match(
    db_path: Path | str | None = None,
    *,
    search: str | None = None,
    match_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    return list_predictions(
        db_path,
        search=search,
        match_id=match_id,
        limit=limit,
        latest_only=True,
    )


def get_prediction_detail(
    prediction_id: str,
    db_path: Path | str | None = None,
    *,
    by_context_key: bool = False,
) -> dict[str, Any] | None:
    column = "prediction_context_key" if by_context_key else "prediction_id"
    with _connect(db_path) as conn:
        row = conn.execute(
            f"SELECT * FROM predictions WHERE {column} = ?",
            (prediction_id,),
        ).fetchone()
    return _row_to_dict(row) if row else None


def delete_prediction(prediction_id: str, db_path: Path | str | None = None) -> int:
    with _connect(db_path) as conn:
        cursor = conn.execute("DELETE FROM predictions WHERE prediction_id = ?", (prediction_id,))
        conn.commit()
        return int(cursor.rowcount)


def clear_history(db_path: Path | str | None = None) -> int:
    with _connect(db_path) as conn:
        count = int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
        conn.execute("DELETE FROM predictions")
        conn.commit()
        return count


def _top_score(record: dict[str, Any]) -> str:
    top_scores = record.get("top_scores_json") or []
    if isinstance(top_scores, list) and top_scores:
        return str(top_scores[0].get("score", ""))
    return ""


def _probabilities(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("probabilities_1x2_json") or {}


def _data_sources_label(record: dict[str, Any]) -> str:
    sources = record.get("data_sources_json") or {}
    channels = sources.get("odds_channels") if isinstance(sources, dict) else {}
    if not isinstance(channels, dict):
        return ""
    labels = []
    for name, payload in channels.items():
        if isinstance(payload, dict):
            labels.append(f"{name}:{payload.get('source', '')}")
    return "; ".join(labels)


def export_predictions_csv(records: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for record in records:
        probs = _probabilities(record)
        writer.writerow(
            {
                "updated_at": record.get("updated_at"),
                "match_id": record.get("match_id"),
                "home_team": record.get("home_team"),
                "away_team": record.get("away_team"),
                "kickoff_time": record.get("kickoff_time"),
                "top_score": _top_score(record),
                "home_win_prob": probs.get("home"),
                "draw_prob": probs.get("draw"),
                "away_win_prob": probs.get("away"),
                "lambda_home": record.get("lambda_home"),
                "lambda_away": record.get("lambda_away"),
                "rho": record.get("rho"),
                "confidence_score": record.get("confidence_score"),
                "risk_level": record.get("risk_level"),
                "data_sources": _data_sources_label(record),
                "prediction_context_key": record.get("prediction_context_key"),
                "run_count": record.get("run_count"),
            }
        )
    return output.getvalue()


def export_predictions_json(records: list[dict[str, Any]]) -> str:
    return json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
