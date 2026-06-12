from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from score_predictor.history.store import (
    clear_history,
    delete_prediction,
    export_predictions_csv,
    export_predictions_json,
    get_prediction_detail,
    list_latest_by_match,
    list_predictions,
)
from score_predictor.ui.components import (
    render_empty_state,
    render_metadata_grid,
    render_section_card,
    render_status_card,
    render_styled_table,
)


def _fmt_probability(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "未保存"


def _fmt_number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "未保存"


def history_mode_label(record_or_mode: dict[str, Any] | str | None) -> str:
    mode = record_or_mode
    if isinstance(record_or_mode, dict):
        mode = record_or_mode.get("source_mode") or record_or_mode.get("app_mode")
    if mode == "sporttery_only":
        return "体彩-only"
    if mode == "dual_channel":
        return "国际+体彩"
    return str(mode or "未记录")


def _top_score(record: dict[str, Any]) -> str:
    top_scores = record.get("top_scores_json") or []
    if isinstance(top_scores, list) and top_scores:
        return str(top_scores[0].get("score") or "")
    snapshot = record.get("dashboard_snapshot_json") or {}
    summary = snapshot.get("prediction_summary") if isinstance(snapshot, dict) else {}
    top_scores = summary.get("top_scores") if isinstance(summary, dict) else []
    if isinstance(top_scores, list) and top_scores:
        return str(top_scores[0].get("score") or "")
    return "未保存"


def _history_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        probs = record.get("probabilities_1x2_json") or {}
        expected_goals = None
        if record.get("lambda_home") is not None and record.get("lambda_away") is not None:
            expected_goals = float(record["lambda_home"]) + float(record["lambda_away"])
        rows.append(
            {
                "prediction_id": record.get("prediction_id"),
                "更新时间": record.get("updated_at"),
                "比赛": f"{record.get('home_team')} vs {record.get('away_team')}",
                "开赛时间": record.get("kickoff_time"),
                "模式": history_mode_label(record),
                "最可能比分": _top_score(record),
                "主胜": _fmt_probability(probs.get("home")),
                "平": _fmt_probability(probs.get("draw")),
                "客胜": _fmt_probability(probs.get("away")),
                "预计进球": _fmt_number(expected_goals),
                "置信度": _fmt_probability(record.get("confidence_score")),
                "风险等级": record.get("risk_level") or "未保存",
                "数据源": history_mode_label(record),
                "run_count": record.get("run_count"),
                "context短hash": str(record.get("prediction_context_key") or "")[:12],
            }
        )
    return pd.DataFrame(rows)


def _snapshot(detail: dict[str, Any]) -> dict[str, Any]:
    snapshot = detail.get("dashboard_snapshot_json")
    return snapshot if isinstance(snapshot, dict) else {}


def _snapshot_section(detail: dict[str, Any], key: str, fallback_key: str | None = None) -> Any:
    snapshot = _snapshot(detail)
    if key in snapshot:
        return snapshot.get(key)
    if fallback_key:
        return detail.get(fallback_key)
    return None


def _render_json_expander(title: str, value: Any) -> None:
    with st.expander(title, expanded=False):
        st.json(value if value not in (None, "") else {"message": "旧记录未保存该字段"})


def render_history_snapshot_summary(detail: dict[str, Any]) -> None:
    snapshot = _snapshot(detail)
    summary = snapshot.get("prediction_summary") if isinstance(snapshot, dict) else {}
    top_scores = detail.get("top_scores_json") or summary.get("top_scores") or []
    top_score = top_scores[0] if isinstance(top_scores, list) and top_scores else {}
    match = snapshot.get("match") if isinstance(snapshot.get("match"), dict) else {}
    render_metadata_grid(
        [
            ("prediction_id", detail.get("prediction_id")),
            ("context key", str(detail.get("prediction_context_key") or "")[:24]),
            ("比赛", f"{detail.get('home_team') or match.get('home_team')} vs {detail.get('away_team') or match.get('away_team')}"),
            ("模式", history_mode_label(detail)),
            ("保存时间", detail.get("updated_at")),
            ("run_count", detail.get("run_count")),
            ("最可能比分", top_score.get("score") if isinstance(top_score, dict) else "旧记录未保存该字段"),
            ("Top score 概率", _fmt_probability(top_score.get("prob") if isinstance(top_score, dict) else None)),
            ("置信度", _fmt_probability(detail.get("confidence_score"))),
            ("风险等级", detail.get("risk_level") or "未保存"),
        ]
    )


def render_history_score_tables(detail: dict[str, Any]) -> None:
    summary = _snapshot_section(detail, "prediction_summary") or {}
    top_scores = detail.get("top_scores_json") or (summary.get("top_scores") if isinstance(summary, dict) else []) or []
    if isinstance(top_scores, list) and top_scores:
        top_frame = pd.DataFrame(top_scores)
        render_styled_table(top_frame.head(5), "Top 5 比分", max_height=240)
        render_styled_table(top_frame.head(10), "Top 10 比分", max_height=360)
    else:
        render_empty_state("Top 5 / Top 10 比分", "旧记录未保存该字段")

    score_matrix = detail.get("score_matrix_json") or _snapshot_section(detail, "score_matrix", "score_matrix_json") or []
    if isinstance(score_matrix, list) and score_matrix:
        render_styled_table(pd.DataFrame(score_matrix), "比分矩阵 / 热力图数据", max_height=420)
    else:
        render_empty_state("比分矩阵 / 热力图数据", "旧记录未保存该字段")


def render_history_audit_sections(detail: dict[str, Any]) -> None:
    risk = detail.get("risk_diagnostics_json") or _snapshot_section(detail, "risk_diagnostics", "risk_diagnostics_json") or {}
    warnings = risk.get("warnings") if isinstance(risk, dict) else detail.get("warnings_json")
    if warnings:
        render_styled_table(pd.DataFrame([{"风险提示": item} for item in warnings]), "风险提示", max_height=260)
    else:
        render_empty_state("风险提示", "该历史记录未保存风险提示")

    market_quality = detail.get("market_quality_json") or _snapshot_section(detail, "market_quality", "market_quality_json") or {}
    _render_json_expander("市场质量评分", market_quality)
    _render_json_expander("odds movement 趋势摘要", detail.get("odds_movement_summary_json") or _snapshot_section(detail, "odds_movement_summary", "odds_movement_summary_json"))
    _render_json_expander("数据源", detail.get("data_sources_json") or _snapshot_section(detail, "data_sources", "data_sources_json"))


def render_prediction_history_detail(detail: dict[str, Any]) -> None:
    render_section_card("历史预测详情", "从 SQLite 中保存的 snapshot 读取，不依赖当前页面 session。")
    render_history_snapshot_summary(detail)

    render_section_card("比分概率", "Top 5 / Top 10 与比分矩阵数据。")
    render_history_score_tables(detail)
    one_x_two = detail.get("probabilities_1x2_json") or _snapshot_section(detail, "probabilities_1x2", "probabilities_1x2_json")
    if one_x_two:
        render_styled_table(pd.DataFrame([one_x_two]), "胜平负概率", max_height=180)
    else:
        render_empty_state("胜平负概率", "旧记录未保存该字段")

    render_section_card("总进球 / BTTS", "总进球分布、Over / Under 和 BTTS 概率。")
    total_goals = detail.get("total_goals_distribution_json") or _snapshot_section(detail, "total_goals_distribution", "total_goals_distribution_json")
    if total_goals:
        render_styled_table(pd.DataFrame([total_goals]), "总进球分布", max_height=180)
    else:
        render_empty_state("总进球分布", "旧记录未保存该字段")
    over_under = detail.get("over_under_probabilities_json") or _snapshot_section(detail, "over_under_probabilities", "over_under_probabilities_json")
    render_styled_table(pd.DataFrame([over_under or {"message": "旧记录未保存该字段"}]), "Over / Under 概率", max_height=180)
    btts = detail.get("btts_probabilities_json") or _snapshot_section(detail, "btts_probabilities", "btts_probabilities_json")
    if btts:
        render_styled_table(pd.DataFrame([btts]), "BTTS 概率", max_height=180)
    else:
        render_empty_state("BTTS 概率", "该历史记录未保存 BTTS")

    render_section_card("lambda 与模型参数", "lambda_home / lambda_away / rho 与保存时参数。")
    lambda_summary = detail.get("lambda_summary_json") or _snapshot_section(detail, "lambda_summary", "lambda_summary_json") or {}
    if not lambda_summary:
        lambda_summary = {
            "lambda_home": detail.get("lambda_home"),
            "lambda_away": detail.get("lambda_away"),
            "rho": detail.get("rho"),
            "lambda_home_before_movement": detail.get("lambda_home_before_movement"),
            "lambda_away_before_movement": detail.get("lambda_away_before_movement"),
            "rho_before_movement": detail.get("rho_before_movement"),
        }
    render_styled_table(pd.DataFrame([lambda_summary]), "lambda_home / lambda_away / rho", max_height=220)
    _render_json_expander("模型参数", detail.get("model_settings_json") or detail.get("settings_json") or {})

    render_section_card("赔率趋势", "movement 修正前后与趋势摘要。")
    movement = detail.get("movement_adjustment_json") or _snapshot_section(detail, "movement_adjustment", "movement_adjustment_json")
    render_styled_table(pd.DataFrame([movement or {"message": "旧记录未保存该字段"}]), "movement before / after", max_height=240)

    render_section_card("审计与风险", "盘口质量、风险提示、数据完整度与数据源。")
    render_history_audit_sections(detail)

    render_section_card("输入与原始 JSON", "input hash / context key / raw result snapshot。")
    _render_json_expander("输入摘要", detail.get("input_payload_summary_json") or detail.get("input_summary_json") or {})
    _render_json_expander("完整 dashboard snapshot", detail.get("dashboard_snapshot_json") or {"message": "旧记录未保存该字段"})
    _render_json_expander("原始 JSON", detail.get("raw_result_json") or {"message": "旧记录未保存该字段"})


def render_prediction_history_tab(
    *,
    app_mode: str | None = None,
    key_prefix: str = "prediction_history",
    title: str = "预测历史",
) -> None:
    render_section_card(title, "历史列表 → 选择历史记录查看详情；详情读取保存时的完整 snapshot。")
    c1, c2, c3 = st.columns([1.2, 1.2, 1.0])
    with c1:
        search = st.text_input("按球队搜索", value="", key=f"{key_prefix}_search")
    with c2:
        match_filter = st.text_input("按 match_id 搜索", value="", key=f"{key_prefix}_match")
    with c3:
        show_all = st.checkbox("显示全部版本", value=False, key=f"{key_prefix}_all")

    records = (
        list_predictions(search=search, match_id=match_filter or None, app_mode=app_mode, latest_only=False)
        if show_all
        else list_latest_by_match(search=search, match_id=match_filter or None, app_mode=app_mode)
    )
    if not records:
        render_empty_state("暂无预测历史", "完成一次预测后会自动保存。")
        return

    table = _history_table(records)
    render_styled_table(table.drop(columns=["prediction_id"]), "历史列表", max_height=420)
    options = {
        f"{row.get('updated_at')} | {row.get('home_team')} vs {row.get('away_team')} | {history_mode_label(row)} | {row.get('prediction_id')}": row.get("prediction_id")
        for row in records
    }
    selected_label = st.selectbox("选择历史记录查看详情", list(options.keys()), key=f"{key_prefix}_selected")
    selected_id = str(options[selected_label])
    detail = get_prediction_detail(selected_id)
    if detail:
        render_prediction_history_detail(detail)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.download_button(
            "导出历史 CSV",
            data=export_predictions_csv(records),
            file_name=f"{key_prefix}.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "导出选中记录 JSON",
            data=export_predictions_json([detail] if detail else []),
            file_name=f"{selected_id}.json",
            mime="application/json",
        )
    with c3:
        st.download_button(
            "导出全部历史 JSON",
            data=export_predictions_json(records),
            file_name=f"{key_prefix}.json",
            mime="application/json",
        )
    with c4:
        confirm_delete = st.checkbox("确认删除选中记录", key=f"{key_prefix}_confirm_delete")
        if st.button("删除选中记录", disabled=not confirm_delete, key=f"{key_prefix}_delete"):
            delete_prediction(selected_id)
            st.session_state.pop(f"{key_prefix}_selected", None)
            render_status_card("已删除选中历史记录。", "success")
            st.rerun()
    with c5:
        confirm_clear = st.checkbox("确认清空历史", key=f"{key_prefix}_confirm_clear")
        if st.button("清空历史", disabled=not confirm_clear, key=f"{key_prefix}_clear"):
            count = clear_history(app_mode=app_mode)
            st.session_state.pop(f"{key_prefix}_selected", None)
            render_status_card(f"已清空历史记录：{count} 条。", "success")
            st.rerun()
