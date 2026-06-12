from __future__ import annotations

import html
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
from score_predictor.ui.charts import (
    plot_1x2_pie,
    plot_btts_bar,
    plot_over_under_table,
    plot_score_heatmap,
    plot_top_scores,
    plot_total_goals_distribution,
    score_matrix_to_frame,
    total_goals_distribution,
)
from score_predictor.ui.components import (
    render_empty_state,
    render_metadata_grid,
    render_section_card,
    render_status_card,
    render_styled_table,
)
from score_predictor.ui.sporttery_only_helpers import (
    DISCLAIMER_ZH,
    ODDS_MOVEMENT_SETTINGS,
    SPORTTERY_MODEL_WEIGHTS,
    get_canonical_top_score,
    normalize_sporttery_only_payload,
    run_sporttery_only_prediction,
)
from score_predictor.ui.theme import inject_professional_theme_css
from score_predictor.ui.yaml_io import dump_yaml, load_yaml_payload


APP_TITLE = "体彩版世界杯比分预测引擎"
APP_SUBTITLE = "基于中国体彩固定奖金与赔率趋势的 90 分钟比分概率看板"

SAMPLE_YAML = """match:
  match_id: sample-sporttery-only
  home_team: Canada
  away_team: Bosnia and Herzegovina
  competition: World Cup
  stage: Group
  kickoff_time: "2026-06-12 20:00"
  timezone: Asia/Shanghai
  venue:
    venue_type: neutral
  target: 90min_score

market:
  odds_1x2:
    home: 1.62
    draw: 3.32
    away: 4.75
  rqspf:
    handicap: -1
    home: 3.11
    draw: 3.20
    away: 2.02
  correct_score_odds:
    "0-0": 9.50
    "1-0": 5.40
    "1-1": 5.00
    "2-0": 6.50
    "2-1": 6.00
    home_other: 18.00
    draw_other: 28.00
    away_other: 35.00
  sporttery_total_goals:
    odds:
      "0": 9.50
      "1": 4.30
      "2": 3.10
      "3": 3.60
      "4": 6.20
      "5": 12.50
      "6": 22.00
      "7+": 35.00
  half_full_time:
    HH: 2.51
    HD: 16.00
    HA: 36.00
    DH: 4.25
    DD: 4.80
    DA: 10.00
    AH: 25.00
    AD: 16.00
    AA: 8.40
"""

WARNING_LABELS = {
    "v3_over_under_markets_missing": "未录入传统大小球盘口；总进球分布主要由体彩总进球固定奖金和 1X2 共同约束。",
    "v3_btts_market_missing": "缺少双方进球盘口，1-1、2-1、1-0 等相邻比分区分稳定性较弱。",
    "v3_correct_score_market_missing": "缺少正确比分盘口，具体比分排序可能对 lambda/rho 较敏感。",
    "sporttery_correct_score_soft_constraint": "体彩比分固定奖金已作为补充 soft calibration，不会单独决定最大概率比分。",
    "sporttery_total_goals_used": "体彩总进球固定奖金已参与总进球分布校准。",
    "correct_score_other_not_used": "比分固定奖金 other 项当前仅用于审计，未进入 loss。",
    "correct_score_incomplete": "比分固定奖金录入不完整，已降低校准权重。",
    "total_goals_incomplete": "体彩总进球固定奖金录入不完整，已降低校准权重。",
    "sporttery_market_low_payout_rate": "体彩该市场返还率偏低或录入不完整，已自动降低校准权重。",
    "odds_channel_conflict": "体彩不同盘口之间存在方向分歧，模型置信度已下调。",
    "odds_movement_lambda_adjusted": "赔率趋势已对 lambda/rho 做低权重有界修正。",
    "odds_movement_adjustment_clamped": "赔率趋势修正触发 clamp 上限。",
    "insufficient_movement_history": "赔率趋势历史快照不足，未进行 movement 修正。",
    "movement_signal_weak_due_to_volatility": "赔率走势震荡较高，movement 修正强度已降低。",
    "cross_market_movement_conflict": "赔率趋势跨市场信号存在分歧，已降低或关闭 movement 修正。",
}


def _escape(value: Any) -> str:
    return html.escape("—" if value in (None, "") else str(value))


def _fmt_probability(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "待预测"


def _fmt_number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _translate_warning(warning: Any) -> str:
    text = str(warning)
    return WARNING_LABELS.get(text, text.replace("_", " "))


def _init_session() -> None:
    st.session_state.setdefault("sporttery_source_payload", None)
    st.session_state.setdefault("sporttery_prematch_context", None)
    st.session_state.setdefault("sporttery_prediction_result", None)
    st.session_state.setdefault("sporttery_prediction_payload", None)
    st.session_state.setdefault("sporttery_history_record", None)


def _current_source_payload() -> dict[str, Any] | None:
    payload = st.session_state.get("sporttery_source_payload")
    return payload if isinstance(payload, dict) else None


def _match_overrides_from_inputs() -> dict[str, Any]:
    return {
        "match_id": st.session_state.get("sporttery_match_id"),
        "home_team": st.session_state.get("sporttery_home_team"),
        "away_team": st.session_state.get("sporttery_away_team"),
        "competition": st.session_state.get("sporttery_competition"),
        "stage": st.session_state.get("sporttery_stage"),
        "kickoff_time": st.session_state.get("sporttery_kickoff"),
        "timezone": st.session_state.get("sporttery_timezone"),
        "neutral_site": st.session_state.get("sporttery_neutral_site"),
        "venue": {
            "venue_type": "neutral"
            if st.session_state.get("sporttery_neutral_site", True)
            else "home"
        },
    }


def _settings_overrides_from_inputs() -> dict[str, Any]:
    return {
        "market_only_mode": True,
        "dc_enabled": bool(st.session_state.get("sporttery_dc_enabled", True)),
        "max_goals": int(st.session_state.get("sporttery_max_goals", 8)),
        "market_weight": 1.0,
        "sporttery_1x2_weight": float(
            st.session_state.get(
                "sporttery_1x2_weight",
                SPORTTERY_MODEL_WEIGHTS["sporttery_1x2_weight"],
            )
        ),
        "sporttery_total_goals_weight": float(
            st.session_state.get(
                "sporttery_total_goals_weight",
                SPORTTERY_MODEL_WEIGHTS["sporttery_total_goals_weight"],
            )
        ),
        "sporttery_correct_score_weight": float(
            st.session_state.get(
                "sporttery_correct_score_weight",
                SPORTTERY_MODEL_WEIGHTS["sporttery_correct_score_weight"],
            )
        ),
        "sporttery_handicap_3way_weight": float(
            st.session_state.get(
                "sporttery_handicap_3way_weight",
                SPORTTERY_MODEL_WEIGHTS["sporttery_handicap_3way_weight"],
            )
        ),
        "sporttery_half_full_weight": 0.0,
        "team_adjustment_strength": float(
            st.session_state.get("sporttery_team_adjustment_strength", 1.0)
        ),
    }


def _movement_overrides_from_inputs() -> dict[str, Any]:
    weights = dict(ODDS_MOVEMENT_SETTINGS["movement_weights"])
    weights.update(
        {
            "sporttery_1x2_movement": float(
                st.session_state.get("sporttery_1x2_movement", weights["sporttery_1x2_movement"])
            ),
            "sporttery_handicap_3way_movement": float(
                st.session_state.get(
                    "sporttery_handicap_3way_movement",
                    weights["sporttery_handicap_3way_movement"],
                )
            ),
            "sporttery_correct_score_movement": float(
                st.session_state.get(
                    "sporttery_correct_score_movement",
                    weights["sporttery_correct_score_movement"],
                )
            ),
            "sporttery_total_goals_movement": float(
                st.session_state.get(
                    "sporttery_total_goals_movement",
                    weights["sporttery_total_goals_movement"],
                )
            ),
            "sporttery_half_full_movement": 0.0,
        }
    )
    return {
        "enabled": bool(st.session_state.get("sporttery_movement_enabled", True)),
        "affect_confidence": True,
        "affect_market_quality": True,
        "affect_lambda": bool(st.session_state.get("sporttery_movement_affect_lambda", True)),
        "late_window_hours": float(st.session_state.get("sporttery_late_window_hours", 6.0)),
        "max_lambda_adjustment": float(st.session_state.get("sporttery_max_lambda_adjustment", 0.05)),
        "max_total_lambda_adjustment": float(
            st.session_state.get("sporttery_max_total_lambda_adjustment", 0.06)
        ),
        "max_rho_adjustment": float(st.session_state.get("sporttery_max_rho_adjustment", 0.025)),
        "movement_weights": weights,
    }


def _load_yaml_from_text(text: str) -> dict[str, Any]:
    return load_yaml_payload(text)


def _render_hero(result: dict[str, Any] | None) -> None:
    top_score = get_canonical_top_score(result)
    v3 = (result or {}).get("v3") or {}
    probs = (v3.get("probabilities") or {}).get("one_x_two") or {}
    flow = v3.get("lambda_flow") or {}
    confidence = (v3.get("confidence") or {}).get("final_confidence_score")
    match_name = result.get("match") if result else "等待载入比赛"
    kickoff = result.get("kickoff_time") if result else "—"
    st.markdown(
        f"""
        <div class="hero-card">
          <div class="hero-content">
            <div>
              <div class="hero-title">{_escape(APP_TITLE)}</div>
              <div class="hero-subtitle">{_escape(APP_SUBTITLE)}</div>
              <div class="match-title">{_escape(match_name)}</div>
              <div class="hero-meta">
                <div class="meta-item"><div class="meta-label">开赛时间</div><div class="meta-value">{_escape(kickoff)}</div></div>
                <div class="meta-item"><div class="meta-label">数据源</div><div class="meta-value">中国体彩 YAML primary calibration</div></div>
                <div class="meta-item"><div class="meta-label">综合置信评分</div><div class="meta-value">{_fmt_probability(confidence)}</div></div>
                <div class="meta-item"><div class="meta-label">lambda_home / lambda_away</div><div class="meta-value">{_fmt_number(flow.get("final_lambda_home"))} / {_fmt_number(flow.get("final_lambda_away"))}</div></div>
              </div>
            </div>
            <div class="kpi-grid">
              <div class="kpi-card"><div class="kpi-label">最可能比分</div><div class="kpi-value">{_escape(top_score.get("score") if top_score else "待预测")}</div><div class="kpi-caption">{_fmt_probability(top_score.get("prob")) if top_score else "等待模型输出"}</div></div>
              <div class="kpi-card"><div class="kpi-label">胜平负概率</div><div class="kpi-value">{_fmt_probability(probs.get("home"))}</div><div class="kpi-caption">主胜 / 平 / 客胜：{_fmt_probability(probs.get("home"))} / {_fmt_probability(probs.get("draw"))} / {_fmt_probability(probs.get("away"))}</div></div>
              <div class="kpi-card"><div class="kpi-label">预计进球</div><div class="kpi-value">{_fmt_number((flow.get("final_lambda_home") or 0) + (flow.get("final_lambda_away") or 0))}</div><div class="kpi-caption">90 分钟总进球均值</div></div>
              <div class="kpi-card"><div class="kpi-label">rho</div><div class="kpi-value">{_fmt_number((v3.get("joint_fit") or {}).get("rho"))}</div><div class="kpi-caption">Dixon-Coles 低比分修正</div></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_conclusion_summary(result: dict[str, Any]) -> None:
    top_score = get_canonical_top_score(result)
    v3 = result.get("v3") or {}
    probs = (v3.get("probabilities") or {}).get("one_x_two") or {}
    btts = (v3.get("probabilities") or {}).get("btts") or {}
    flow = v3.get("lambda_flow") or {}
    warnings = list(v3.get("risk_warnings") or [])
    st.markdown(
        f"""
        <div class="section-card">
          <p class="section-title">结论摘要</p>
          <p class="section-copy">同一 canonical top score 会同时用于顶部 Hero 和本摘要。</p>
          <div class="summary-grid">
            <div class="summary-item"><div class="summary-label">最可能比分</div><div class="summary-value">{_escape(top_score.get("score") if top_score else "待预测")}</div></div>
            <div class="summary-item"><div class="summary-label">胜平负概率</div><div class="summary-value">主胜 {_fmt_probability(probs.get("home"))}<br>平 {_fmt_probability(probs.get("draw"))}<br>客胜 {_fmt_probability(probs.get("away"))}</div></div>
            <div class="summary-item"><div class="summary-label">BTTS 概率</div><div class="summary-value">是 {_fmt_probability(btts.get("yes"))}<br>否 {_fmt_probability(btts.get("no"))}</div></div>
            <div class="summary-item"><div class="summary-label">预计进球</div><div class="summary-value">{_fmt_number((flow.get("final_lambda_home") or 0) + (flow.get("final_lambda_away") or 0))}</div></div>
            <div class="summary-item"><div class="summary-label">风险提示</div><div class="summary-value">{_escape("；".join(_translate_warning(item) for item in warnings[:2]) or "暂无高优先级风险")}</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _top_scores_frame(top_scores: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"排名": index, "比分": row.get("score"), "概率": _fmt_probability(row.get("prob"))}
            for index, row in enumerate(top_scores[:10], start=1)
        ]
    )


def _lambda_frame(v3: dict[str, Any]) -> pd.DataFrame:
    flow = v3.get("lambda_flow") or {}
    fit = v3.get("joint_fit") or {}
    return pd.DataFrame(
        [
            {"指标": "lambda_home_before_movement", "数值": _fmt_number(flow.get("lambda_home_before_movement"))},
            {"指标": "lambda_away_before_movement", "数值": _fmt_number(flow.get("lambda_away_before_movement"))},
            {"指标": "lambda_home", "数值": _fmt_number(flow.get("final_lambda_home"))},
            {"指标": "lambda_away", "数值": _fmt_number(flow.get("final_lambda_away"))},
            {"指标": "rho_before_movement", "数值": _fmt_number(fit.get("market_rho_before_movement"))},
            {"指标": "rho", "数值": _fmt_number(fit.get("rho"))},
        ]
    )


def _movement_frame(v3: dict[str, Any]) -> pd.DataFrame:
    movement = v3.get("movement_adjustment") or {}
    if not movement:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {"指标": "lambda_home", "修正前": _fmt_number(movement.get("lambda_home_before")), "修正后": _fmt_number(movement.get("lambda_home_after")), "变化": _fmt_probability(movement.get("home_adjustment_pct"), 2)},
            {"指标": "lambda_away", "修正前": _fmt_number(movement.get("lambda_away_before")), "修正后": _fmt_number(movement.get("lambda_away_after")), "变化": _fmt_probability(movement.get("away_adjustment_pct"), 2)},
            {"指标": "rho", "修正前": _fmt_number(movement.get("rho_before")), "修正后": _fmt_number(movement.get("rho_after")), "变化": _fmt_number(movement.get("rho_adjustment"), 4)},
            {"指标": "total_lambda", "修正前": "—", "修正后": "—", "变化": _fmt_probability(movement.get("total_adjustment_pct"), 2)},
        ]
    )


def _market_status_frame(v3: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, item in (v3.get("sporttery_market_status") or {}).items():
        rows.append(
            {
                "市场": key,
                "状态": item.get("status"),
                "base_weight": _fmt_number(item.get("base_weight"), 2),
                "market_quality_score": _fmt_number(item.get("market_quality_score"), 2),
                "consistency_score": _fmt_number(item.get("consistency_score"), 2),
                "final_weight": _fmt_number(item.get("final_weight"), 3),
                "payout_rate": _fmt_probability(item.get("payout_rate")),
                "warnings": "；".join(_translate_warning(w) for w in (item.get("warnings") or [])),
            }
        )
    return pd.DataFrame(rows)


def _movement_summary_frame(v3: dict[str, Any]) -> pd.DataFrame:
    summary = v3.get("odds_movement") or {}
    rows = []
    for market_name, market in (summary.get("markets") or {}).items():
        for outcome, item in (market.get("outcomes") or {}).items():
            rows.append(
                {
                    "市场": market_name,
                    "结果": outcome,
                    "开盘去水概率": _fmt_probability(item.get("open_devig_prob"), 2),
                    "最新去水概率": _fmt_probability(item.get("latest_devig_prob"), 2),
                    "概率变化": _fmt_probability(item.get("prob_delta"), 2),
                    "方向": item.get("movement_direction"),
                    "强度": item.get("movement_strength"),
                }
            )
    return pd.DataFrame(rows)


def _risk_level(v3: dict[str, Any]) -> str:
    confidence = float((v3.get("confidence") or {}).get("final_confidence_score") or 0.0)
    warnings = " ".join(str(item) for item in (v3.get("risk_warnings") or []))
    if "conflict" in warnings or confidence < 0.35:
        return "高"
    if "missing" in warnings or confidence < 0.55:
        return "中"
    return "低"


def _render_prediction_result(result: dict[str, Any], payload: dict[str, Any]) -> None:
    v3 = result.get("v3") or {}
    score_matrix = v3.get("final_score_matrix") or []
    match_parts = str(result.get("match", "Home vs Away")).split(" vs ", 1)
    home_team = match_parts[0] if match_parts else "Home"
    away_team = match_parts[1] if len(match_parts) > 1 else "Away"
    _render_conclusion_summary(result)

    render_section_card("预测看板", "胜平负、比分矩阵、预计进球、BTTS、lambda/rho 与赔率趋势修正。", "V3")
    one_x_two = (v3.get("probabilities") or {}).get("one_x_two") or {}
    c1, c2 = st.columns([0.95, 1.05])
    with c1:
        st.plotly_chart(
            plot_1x2_pie(
                float(one_x_two.get("home", 0.0)),
                float(one_x_two.get("draw", 0.0)),
                float(one_x_two.get("away", 0.0)),
            ),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(plot_total_goals_distribution(score_matrix), use_container_width=True)

    st.plotly_chart(
        plot_score_heatmap(score_matrix, home_team, away_team),
        use_container_width=True,
    )
    c1, c2 = st.columns([0.82, 1.18])
    with c1:
        render_styled_table(_top_scores_frame(v3.get("top_scores") or []), "Top 10 比分", max_height=360)
    with c2:
        st.plotly_chart(plot_top_scores(v3.get("top_scores") or []), use_container_width=True)

    total_goals = total_goals_distribution(score_matrix)
    render_styled_table(
        pd.DataFrame(
            [
                {"总进球": row["total_goals"], "概率": _fmt_probability(row["probability"])}
                for row in total_goals.to_dict(orient="records")
            ]
        ),
        "总进球分布",
        max_height=300,
    )
    over_under = (v3.get("probabilities") or {}).get("over_under") or {}
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_over_under_table(over_under), use_container_width=True)
    with c2:
        btts = (v3.get("probabilities") or {}).get("btts") or {}
        st.plotly_chart(
            plot_btts_bar(float(btts.get("yes", 0.0)), float(btts.get("no", 0.0))),
            use_container_width=True,
        )

    render_styled_table(_lambda_frame(v3), "lambda_home / lambda_away / rho", max_height=260)
    render_styled_table(_movement_frame(v3), "odds movement 修正前后", max_height=260)
    render_styled_table(_market_status_frame(v3), "体彩市场质量评分", max_height=360)
    movement_rows = _movement_summary_frame(v3)
    if movement_rows.empty:
        render_empty_state("暂无体彩盘口趋势摘要", "上传包含 history 快照的体彩 YAML 后会显示去水概率变化。")
    else:
        render_styled_table(movement_rows, "体彩盘口趋势摘要", max_height=360)

    confidence = v3.get("confidence") or {}
    render_metadata_grid(
        [
            ("风险等级", _risk_level(v3)),
            ("数据完整度", _fmt_probability(confidence.get("data_quality_score"))),
            ("市场一致性", _fmt_probability(confidence.get("market_consistency_score"))),
            ("稳定性", _fmt_probability(confidence.get("sensitivity_stability_score"))),
        ]
    )
    st.download_button(
        "下载本次预测 YAML",
        data=dump_yaml(payload),
        file_name="sporttery_only_prediction.yaml",
        mime="text/yaml",
    )


def _render_match_tab() -> None:
    render_section_card("比赛信息", "用于生成预测上下文和历史记录 key；赔率仍以体彩 YAML 为唯一来源。")
    source = _current_source_payload() or {}
    match = source.get("match") if isinstance(source.get("match"), dict) else {}
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input("match_id", value=str(match.get("match_id") or ""), key="sporttery_match_id")
        st.text_input("主队", value=str(match.get("home_team") or "Canada"), key="sporttery_home_team")
    with c2:
        st.text_input("客队", value=str(match.get("away_team") or "Bosnia and Herzegovina"), key="sporttery_away_team")
        st.text_input("赛事", value=str(match.get("competition") or "World Cup"), key="sporttery_competition")
    with c3:
        st.text_input("阶段", value=str(match.get("stage") or "Group"), key="sporttery_stage")
        st.text_input("开赛时间", value=str(match.get("kickoff_time") or "2026-06-12 20:00"), key="sporttery_kickoff")
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("时区", value=str(match.get("timezone") or "Asia/Shanghai"), key="sporttery_timezone")
    with c2:
        st.checkbox("中立场", value=bool(match.get("neutral_site", True)), key="sporttery_neutral_site")


def _render_sporttery_tab() -> None:
    render_section_card("体彩赔率通道", "上传或手动输入中国体彩固定奖金 YAML；该通道作为 primary calibration。", "primary")
    uploaded = st.file_uploader("上传体彩 YAML", type=["yaml", "yml"], key="sporttery_yaml_upload")
    text = st.text_area("手动输入体彩 YAML", value=SAMPLE_YAML, height=430, key="sporttery_yaml_text")
    c1, c2 = st.columns([1, 3])
    with c1:
        load_clicked = st.button("加载体彩 YAML", type="primary", use_container_width=True)
    if uploaded is not None:
        try:
            payload = load_yaml_payload(uploaded.getvalue())
            st.session_state["sporttery_source_payload"] = payload
            render_status_card("已载入上传的体彩 YAML。", "success")
        except Exception as exc:
            render_status_card(f"YAML 解析失败：{exc}", "danger")
    elif load_clicked:
        try:
            payload = _load_yaml_from_text(text)
            st.session_state["sporttery_source_payload"] = payload
            render_status_card("已载入手动输入的体彩 YAML。", "success")
        except Exception as exc:
            render_status_card(f"YAML 解析失败：{exc}", "danger")

    payload = _current_source_payload()
    if not payload:
        render_empty_state("尚未载入体彩 YAML", "可以直接使用示例内容，也可以上传当前兼容结构或新结构 YAML。")
        return
    try:
        normalized = normalize_sporttery_only_payload(
            payload,
            match_overrides=_match_overrides_from_inputs(),
            prematch_context=st.session_state.get("sporttery_prematch_context"),
            settings_overrides=_settings_overrides_from_inputs(),
            movement_settings_overrides=_movement_overrides_from_inputs(),
        )
        sporttery = (normalized.get("markets") or {}).get("sporttery") or {}
        rows = []
        labels = {
            "sporttery_1x2": "胜平负",
            "sporttery_handicap_3way": "让球胜平负",
            "sporttery_correct_score": "比分固定奖金",
            "sporttery_total_goals": "总进球固定奖金",
            "sporttery_half_full": "半全场",
        }
        for key, label in labels.items():
            status = "audit-only" if key == "sporttery_half_full" and sporttery.get(key) else "参与 calibration" if sporttery.get(key) else "ignored"
            rows.append({"市场": label, "识别状态": "已识别" if sporttery.get(key) else "未识别", "模型状态": status})
        render_styled_table(pd.DataFrame(rows), "识别到的体彩市场", max_height=260)
    except Exception as exc:
        render_status_card(f"体彩 YAML 当前无法进入模型：{exc}", "danger")


def _render_context_tab() -> None:
    render_section_card("赛前情报", "上传事实型赛前情报 YAML，作为轻量修正和审计输入。")
    uploaded = st.file_uploader("上传赛前情报 YAML", type=["yaml", "yml"], key="sporttery_prematch_upload")
    if uploaded is not None:
        try:
            st.session_state["sporttery_prematch_context"] = load_yaml_payload(uploaded.getvalue())
            render_status_card("赛前情报 YAML 已载入。", "success")
        except Exception as exc:
            render_status_card(f"赛前情报解析失败：{exc}", "danger")
    context = st.session_state.get("sporttery_prematch_context")
    if context:
        st.code(dump_yaml(context), language="yaml")
    else:
        render_empty_state("尚未载入赛前情报", "没有赛前情报时，模型仍会仅基于体彩固定奖金运行。")


def _render_settings_tab() -> None:
    render_section_card("模型参数", "体彩市场权重默认按纯数据版设置；半全场保持 audit-only，不影响 lambda。")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.checkbox("启用 Dixon-Coles", value=True, key="sporttery_dc_enabled")
        st.slider("max_goals", 5, 12, 8, 1, key="sporttery_max_goals")
        st.slider("赛前情报修正强度", 0.0, 1.0, 1.0, 0.05, key="sporttery_team_adjustment_strength")
    with c2:
        st.slider("胜平负权重", 0.20, 1.50, SPORTTERY_MODEL_WEIGHTS["sporttery_1x2_weight"], 0.05, key="sporttery_1x2_weight")
        st.slider("总进球权重", 0.10, 1.20, SPORTTERY_MODEL_WEIGHTS["sporttery_total_goals_weight"], 0.05, key="sporttery_total_goals_weight")
        st.slider("比分固定奖金权重", 0.05, 0.60, SPORTTERY_MODEL_WEIGHTS["sporttery_correct_score_weight"], 0.05, key="sporttery_correct_score_weight")
    with c3:
        st.slider("让球胜平负权重", 0.05, 0.80, SPORTTERY_MODEL_WEIGHTS["sporttery_handicap_3way_weight"], 0.05, key="sporttery_handicap_3way_weight")
        st.number_input(
            "半全场权重",
            min_value=0.0,
            max_value=0.0,
            value=0.0,
            step=0.01,
            key="sporttery_half_full_weight",
            disabled=True,
        )

    render_section_card("赔率趋势参数", "history 快照会先去水，再以低权重、有上限方式修正 lambda/rho。")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.checkbox("启用 odds movement", value=True, key="sporttery_movement_enabled")
        st.checkbox("允许修正 lambda", value=True, key="sporttery_movement_affect_lambda")
        st.slider("临近窗口小时", 1.0, 24.0, 6.0, 1.0, key="sporttery_late_window_hours")
    with c2:
        st.slider("单队 lambda 上限", 0.00, 0.10, 0.05, 0.005, key="sporttery_max_lambda_adjustment")
        st.slider("总 lambda 上限", 0.00, 0.12, 0.06, 0.005, key="sporttery_max_total_lambda_adjustment")
        st.slider("rho 上限", 0.00, 0.05, 0.025, 0.005, key="sporttery_max_rho_adjustment")
    with c3:
        st.slider("1X2 趋势权重", 0.00, 0.30, 0.12, 0.01, key="sporttery_1x2_movement")
        st.slider("让球趋势权重", 0.00, 0.30, 0.10, 0.01, key="sporttery_handicap_3way_movement")
        st.slider("比分/总进球趋势权重", 0.00, 0.30, 0.10, 0.01, key="sporttery_correct_score_movement")
        st.slider("总进球趋势权重", 0.00, 0.30, 0.12, 0.01, key="sporttery_total_goals_movement")


def _run_prediction_from_ui() -> None:
    payload = _current_source_payload()
    if payload is None:
        payload = _load_yaml_from_text(st.session_state.get("sporttery_yaml_text") or SAMPLE_YAML)
        st.session_state["sporttery_source_payload"] = payload
    output = run_sporttery_only_prediction(
        payload,
        match_overrides=_match_overrides_from_inputs(),
        prematch_context=st.session_state.get("sporttery_prematch_context"),
        settings_overrides=_settings_overrides_from_inputs(),
        movement_settings_overrides=_movement_overrides_from_inputs(),
        save_history=True,
    )
    st.session_state["sporttery_prediction_result"] = output["result"]
    st.session_state["sporttery_prediction_payload"] = output["payload"]
    st.session_state["sporttery_history_record"] = output["history_record"]


def _render_dashboard_tab() -> None:
    render_section_card("预测看板", "点击开始预测后自动保存历史，同一 context key 会覆盖旧记录并让 run_count + 1。")
    if st.button("开始预测", type="primary", use_container_width=True):
        try:
            _run_prediction_from_ui()
            record = st.session_state.get("sporttery_history_record") or {}
            if record.get("save_action") == "updated":
                render_status_card("已更新同一预测记录，run_count + 1。", "success")
            else:
                render_status_card("预测已保存到历史记录。", "success")
        except Exception as exc:
            render_status_card(f"预测失败：{exc}", "danger")

    result = st.session_state.get("sporttery_prediction_result")
    payload = st.session_state.get("sporttery_prediction_payload")
    if result and payload:
        _render_prediction_result(result, payload)
    else:
        render_empty_state("尚未生成预测结果", "载入体彩 YAML 后点击开始预测。")


def _render_audit_tab() -> None:
    result = st.session_state.get("sporttery_prediction_result")
    if not result:
        render_empty_state("暂无审计结果", "完成一次预测后可查看市场质量、拟合误差和风险提示。")
        return
    v3 = result.get("v3") or {}
    render_section_card("审计与风险", "综合检查体彩市场质量、盘口一致性、走势修正和模型稳定性。")
    render_metadata_grid(
        [
            ("风险等级", _risk_level(v3)),
            ("数据完整度", _fmt_probability((v3.get("confidence") or {}).get("data_quality_score"))),
            ("市场一致性", _fmt_probability((v3.get("confidence") or {}).get("market_consistency_score"))),
            ("综合置信评分", _fmt_probability((v3.get("confidence") or {}).get("final_confidence_score"))),
        ]
    )
    warnings = list(v3.get("risk_warnings") or [])
    if warnings:
        render_styled_table(
            pd.DataFrame([{"风险提示": _translate_warning(warning)} for warning in warnings]),
            "风险提示",
            max_height=320,
        )
    render_styled_table(_market_status_frame(v3), "体彩市场质量评分", max_height=360)
    fit_errors = v3.get("market_fit_errors") or {}
    if fit_errors:
        render_styled_table(pd.DataFrame([{"市场": key, "详情": value} for key, value in fit_errors.items()]), "市场拟合误差", max_height=360)
    movement = v3.get("odds_movement") or {}
    render_styled_table(
        pd.DataFrame(
            [
                {"项目": "conflict_level", "内容": movement.get("conflict_level")},
                {"项目": "themes", "内容": "；".join(movement.get("themes") or [])},
                {"项目": "drivers", "内容": "；".join(movement.get("drivers") or [])},
            ]
        ),
        "体彩盘口趋势摘要",
        max_height=220,
    )


def _history_top_score(record: dict[str, Any]) -> str:
    top_scores = record.get("top_scores_json") or []
    if isinstance(top_scores, list) and top_scores:
        return str(top_scores[0].get("score", ""))
    return ""


def _history_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        probs = record.get("probabilities_1x2_json") or {}
        rows.append(
            {
                "prediction_id": record.get("prediction_id"),
                "更新时间": record.get("updated_at"),
                "match_id": record.get("match_id"),
                "主队": record.get("home_team"),
                "客队": record.get("away_team"),
                "最可能比分": _history_top_score(record),
                "主胜": _fmt_probability(probs.get("home")),
                "平": _fmt_probability(probs.get("draw")),
                "客胜": _fmt_probability(probs.get("away")),
                "run_count": record.get("run_count"),
            }
        )
    return pd.DataFrame(rows)


def _render_history_detail(record: dict[str, Any]) -> None:
    render_metadata_grid(
        [
            ("prediction_id", record.get("prediction_id")),
            ("context key", str(record.get("prediction_context_key") or "")[:20]),
            ("比赛", f"{record.get('home_team')} vs {record.get('away_team')}"),
            ("run_count", record.get("run_count")),
            ("风险等级", record.get("risk_level")),
            ("置信评分", _fmt_probability(record.get("confidence_score"))),
        ]
    )
    top_scores = pd.DataFrame(record.get("top_scores_json") or [])
    if not top_scores.empty:
        render_styled_table(top_scores.head(10), "Top 10 比分", max_height=320)
    render_styled_table(pd.DataFrame([record.get("probabilities_1x2_json") or {}]), "胜平负概率", max_height=160)
    render_styled_table(pd.DataFrame([record.get("movement_adjustment_json") or {}]), "odds movement 修正前后", max_height=220)


def _render_history_tab() -> None:
    render_section_card("预测历史", "每次成功预测后自动保存；同一 context key 重复预测会 upsert 覆盖旧记录。")
    c1, c2, c3 = st.columns([1.2, 1.2, 1.0])
    with c1:
        search = st.text_input("按球队搜索", value="", key="sporttery_history_search")
    with c2:
        match_filter = st.text_input("按 match_id 搜索", value="", key="sporttery_history_match")
    with c3:
        show_all = st.checkbox("显示全部版本", value=False, key="sporttery_history_all")

    records = (
        list_predictions(search=search, match_id=match_filter or None, latest_only=False)
        if show_all
        else list_latest_by_match(search=search, match_id=match_filter or None)
    )
    if not records:
        render_empty_state("暂无预测历史", "完成一次预测后会自动保存。")
        return
    table = _history_table(records)
    render_styled_table(table.drop(columns=["prediction_id"]), "历史列表", max_height=420)
    options = {
        f"{row.get('updated_at')} | {row.get('home_team')} vs {row.get('away_team')} | {row.get('prediction_id')}": row.get("prediction_id")
        for row in records
    }
    selected_label = st.selectbox("选择历史记录查看详情", list(options.keys()), key="sporttery_history_selected")
    selected_id = str(options[selected_label])
    detail = get_prediction_detail(selected_id)
    if detail:
        _render_history_detail(detail)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            "导出历史 CSV",
            data=export_predictions_csv(records),
            file_name="sporttery_prediction_history.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "导出历史 JSON",
            data=export_predictions_json(records),
            file_name="sporttery_prediction_history.json",
            mime="application/json",
        )
    with c3:
        confirm_delete = st.checkbox("确认删除选中记录", key="sporttery_history_confirm_delete")
        if st.button("删除选中记录", disabled=not confirm_delete, key="sporttery_history_delete"):
            delete_prediction(selected_id)
            render_status_card("已删除选中历史记录。", "success")
    with c4:
        confirm_clear = st.checkbox("确认清空历史", key="sporttery_history_confirm_clear")
        if st.button("清空历史", disabled=not confirm_clear, key="sporttery_history_clear"):
            count = clear_history()
            render_status_card(f"已清空历史记录：{count} 条。", "success")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_professional_theme_css()
    _init_session()
    _render_hero(st.session_state.get("sporttery_prediction_result"))
    render_status_card(DISCLAIMER_ZH, "info")

    (
        match_tab,
        sporttery_tab,
        context_tab,
        settings_tab,
        dashboard_tab,
        audit_tab,
        history_tab,
    ) = st.tabs(
        [
            "比赛信息",
            "体彩赔率通道",
            "赛前情报",
            "模型参数",
            "预测看板",
            "审计与风险",
            "预测历史",
        ]
    )

    with match_tab:
        _render_match_tab()
    with sporttery_tab:
        _render_sporttery_tab()
    with context_tab:
        _render_context_tab()
    with settings_tab:
        _render_settings_tab()
    with dashboard_tab:
        _render_dashboard_tab()
    with audit_tab:
        _render_audit_tab()
    with history_tab:
        _render_history_tab()


if __name__ == "__main__":
    main()
