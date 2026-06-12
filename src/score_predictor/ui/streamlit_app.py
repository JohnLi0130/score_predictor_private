from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from score_predictor.predictor import match_input_from_dict
from score_predictor.predictor import predict
from score_predictor.report import format_human_report, to_pretty_json
from score_predictor.connectors.odds_api_normalizer import (
    MARKET_MODE_KEYS,
    market_keys_for_mode,
    normalize_event_odds_to_v3_input,
    selectable_market_keys,
)
from score_predictor.connectors.the_odds_api import (
    MISSING_API_KEY_MESSAGE,
    DEFAULT_CONFIG,
    extract_available_market_keys,
    fetch_event_markets,
    fetch_event_odds,
    fetch_events,
    fetch_sports,
    find_world_cup_sport_key,
    load_provider_config,
    match_event_by_teams,
)

from score_predictor.ui.charts import (
    plot_1x2_pie,
    plot_btts_bar,
    plot_market_fit_errors,
    plot_over_under_table,
    plot_score_heatmap,
    plot_top_scores,
    plot_total_goals_distribution,
    score_matrix_to_frame,
)
from score_predictor.ui.components import (
    render_badge_row,
    render_empty_state,
    render_metadata_grid,
    render_section_card,
    render_status_card,
    render_styled_table,
    render_text_panel,
    status_badge,
)
from score_predictor.ui.form_helpers import (
    audit_only_score_warning,
    build_input_warnings,
    build_margin_warnings,
    copy_default_form_state,
    rows_from_table,
)
from score_predictor.ui.yaml_io import (
    build_yaml_from_form_state,
    dump_yaml,
    load_yaml_payload,
    load_yaml_to_form_state,
    match_input_from_form_state,
    merge_prediction_payload,
)
from score_predictor.ui.theme import inject_professional_theme_css
from score_predictor.ui.worldcup_config import (
    api_match_name,
    load_team_aliases,
    load_worldcup_groups,
    team_from_label,
    team_label,
)


DISCLAIMER_ZH = (
    "本工具仅用于足球比分概率建模和娱乐分析，不构成投注建议，也不提供任何投注推荐。"
    "最大概率比分不代表确定结果。"
)

FONT_FAMILY = (
    '"Microsoft YaHei", "Noto Sans SC", "PingFang SC", '
    '"Hiragino Sans GB", "SimHei", sans-serif'
)

METRIC_LABELS_ZH = {
    "market_lambda_home": "市场主队预期进球",
    "market_lambda_away": "市场客队预期进球",
    "team_adjusted_lambda_home": "情报修正后主队预期进球",
    "team_adjusted_lambda_away": "情报修正后客队预期进球",
    "final_lambda_home": "最终主队预期进球",
    "final_lambda_away": "最终客队预期进球",
    "rho": "Dixon-Coles 修正参数",
    "dc_enabled": "Dixon-Coles 状态",
    "market_consistency_score": "盘口一致性评分",
    "data_quality_score": "数据质量评分",
    "sensitivity_stability_score": "敏感性稳定评分",
    "result_confidence": "方向置信评分",
    "score_confidence": "比分置信评分",
    "final_confidence_score": "综合置信评分",
    "loss": "联合拟合损失",
    "optimizer_success": "优化器状态",
}

MARKET_LABELS_ZH = {
    "1X2": "胜平负",
    "one_x_two": "胜平负",
    "Over/Under": "大小球",
    "over_under": "大小球",
    "BTTS": "双方进球",
    "btts": "双方进球",
    "Correct Score": "比分固定奖金",
    "correct_score": "比分固定奖金",
}

WARNING_LABELS_ZH = {
    "missing_home_team": "主队名称缺失",
    "missing_away_team": "客队名称缺失",
    "missing_date": "比赛时间缺失",
    "missing_1x2_home_odds": "主胜固定奖金缺失",
    "missing_1x2_draw_odds": "平局固定奖金缺失",
    "missing_1x2_away_odds": "客胜固定奖金缺失",
    "missing_over_under_rows": "大小球盘口缺失",
    "missing_internal_lambda_home": "主队人工 lambda 缺失",
    "missing_internal_lambda_away": "客队人工 lambda 缺失",
    "other_score_odds_are_audit_only_unless_supported_by_calibration": (
        "其他比分固定奖金当前仅作审计参考"
    ),
    "market_only_mode_used_default_internal_lambda": (
        "盘口模式内部 lambda 采用默认兜底值"
    ),
    "asian_handicap_recorded_not_used_in_v0": "亚洲让球已记录，仅用于审计参考",
    "over_under_missing_market_total_less_reliable": (
        "缺少大小球市场，总进球判断可靠性下降"
    ),
    "dixon_coles_enabled_in_v3_section": "Dixon-Coles 已在 V3 中启用",
    "dixon_coles_not_enabled_poisson_only": (
        "Dixon-Coles 未启用，当前使用泊松比分矩阵"
    ),
    "intelligence_missing": "赛前事实情报缺失",
    "official_lineups_not_available": "官方首发尚不可用",
    "official_squads_not_available": "官方名单尚不可用",
    "friendly_match_total_goals_discount": "友谊赛总进球期望已做保守折减",
    "club_friendly_match_total_goals_discount": "俱乐部友谊赛总进球期望已做保守折减",
    "home_low_lineup_strength": "主队阵容强度偏低",
    "away_low_lineup_strength": "客队阵容强度偏低",
    "both_teams_rotation_or_low_strength": "双方轮换或阵容强度偏低",
    "home_key_striker_absent": "主队关键前锋缺阵",
    "away_key_striker_absent": "客队关键前锋缺阵",
    "home_key_playmaker_absent": "主队关键组织者缺阵",
    "away_key_playmaker_absent": "客队关键组织者缺阵",
    "heat_humidity_total_goals_discount": "高温高湿环境压低总进球期望",
    "rain_total_goals_discount": "降雨因素压低总进球期望",
    "market_may_be_public_sentiment_polluted": "市场可能受到舆论热度扰动",
    "v3_over_under_markets_missing": "V3 缺少大小球市场输入",
    "v3_btts_market_missing": "缺少双方进球盘口，1-1、2-1、1-0 等相邻比分区分稳定性较弱。",
    "v3_correct_score_market_missing": "缺少正确比分盘口，具体比分排序可能对 lambda/rho 较敏感。",
    "sporttery_correct_score_soft_constraint": "体彩比分固定奖金已作为补充 soft calibration，不会单独决定最大概率比分。",
    "sporttery_correct_score_supplemented_missing_international": "国际赔率通道未返回正确比分盘口；已使用体彩比分固定奖金作为补充校准源。",
    "sporttery_market_low_payout_rate": "体彩该市场返还率偏低，已自动降低校准权重。",
    "odds_channel_conflict": "国际赔率通道与体彩赔率通道存在方向分歧，模型置信度已下调。",
    "odds_channel_mild_conflict": "国际赔率通道与体彩赔率通道存在轻微分歧，已降低对应权重。",
    "correct_score_incomplete": "比分固定奖金录入不完整，仅作为弱校准信号。",
    "correct_score_other_not_used": "比分固定奖金 other 项当前仅用于审计，未进入 loss。",
    "sporttery_total_goals_used": "体彩总进球固定奖金已参与总进球分布校准。",
    "total_goals_incomplete": "体彩总进球固定奖金录入不完整，已降低校准权重。",
    "most_likely_score_sensitive_to_small_lambda_or_rho_changes": (
        "最可能比分对 lambda 或 rho 的小幅变化较敏感，请同时参考 Top5 比分簇。"
    ),
    "result_direction_sensitive_to_small_lambda_or_rho_changes": (
        "胜平负方向对 lambda 或 rho 的小幅变化较敏感"
    ),
    "top_5_scores_not_stable_under_small_perturbations": (
        "Top 5 比分在小扰动下不够稳定"
    ),
    "sporttery_rqspf_treated_as_official_handicap_win_draw_loss": (
        "让球胜平负按官方让球三项盘口处理"
    ),
}

DIRECTION_LABELS_ZH = {
    "home": "主队方向",
    "away": "客队方向",
    "draw": "平局方向",
    "balanced": "均衡",
    "model": "模型方向",
    "one_x_two": "胜平负市场",
    "asian_handicap": "亚洲让球",
    "sporttery_rqspf_official": "体彩让球胜平负",
}


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORLD_CUP_GROUPS_PATH = PROJECT_ROOT / "config" / "worldcup_2026_groups.yaml"
TEAM_ALIASES_PATH = PROJECT_ROOT / "config" / "team_aliases_worldcup_2026.yaml"
THE_ODDS_API_CONFIG_PATH = PROJECT_ROOT / "config" / "provider_the_odds_api.example.yaml"


def _load_worldcup_groups() -> dict[str, list[dict[str, Any]]]:
    return load_worldcup_groups(WORLD_CUP_GROUPS_PATH)


def _load_team_aliases() -> dict[str, str]:
    return load_team_aliases(TEAM_ALIASES_PATH)


def _api_key_status() -> str:
    return "已设置" if os.getenv("THE_ODDS_API_KEY") else "未设置"


def _quota_from_metadata(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return "暂无"
    return str(metadata.get("x_requests_remaining") or "暂无")


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0B1020;
            --panel: #111827;
            --panel-2: #172033;
            --gold: #F5C542;
            --sky: #38BDF8;
            --green: #22C55E;
            --red: #EF4444;
            --amber: #F59E0B;
            --text: #F8FAFC;
            --muted: #CBD5E1;
            --border: rgba(255,255,255,0.08);
            --font-cn: "Microsoft YaHei", "Noto Sans SC", "PingFang SC",
                "Hiragino Sans GB", "SimHei", sans-serif;
        }

        html, body, [class*="css"], .stApp {
            font-family: var(--font-cn);
        }

        .stApp {
            color: var(--text);
            background:
                radial-gradient(circle at 16% 10%, rgba(56,189,248,0.14), transparent 30%),
                radial-gradient(circle at 86% 4%, rgba(245,197,66,0.10), transparent 28%),
                linear-gradient(180deg, #0B1020 0%, #0A0F1C 52%, #080C16 100%);
        }

        [data-testid="stAppViewContainer"] > .main {
            background: transparent;
        }

        .block-container {
            max-width: 1580px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        #MainMenu, footer, header, [data-testid="stDecoration"] {
            visibility: hidden;
            height: 0;
        }

        h1 {
            font-size: 38px !important;
            line-height: 1.16 !important;
            letter-spacing: 0 !important;
            color: var(--text) !important;
            font-weight: 800 !important;
        }

        h2 {
            font-size: 22px !important;
            color: var(--text) !important;
            letter-spacing: 0 !important;
        }

        h3 {
            font-size: 18px !important;
            color: var(--text) !important;
            letter-spacing: 0 !important;
        }

        p, label, span, div {
            letter-spacing: 0 !important;
        }

        div[data-testid="stMarkdownContainer"] p {
            color: var(--muted);
            font-size: 15px;
            line-height: 1.75;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            padding: 8px;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: rgba(17, 24, 39, 0.72);
        }

        .stTabs [data-baseweb="tab"] {
            height: 44px;
            border-radius: 7px;
            padding: 0 18px;
            color: var(--muted);
            font-size: 15px;
            font-weight: 700;
            background: transparent;
        }

        .stTabs [aria-selected="true"] {
            color: #0B1020 !important;
            background: linear-gradient(135deg, var(--gold), #FFE08A);
            box-shadow: 0 10px 26px rgba(245,197,66,0.22);
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 8px;
            border: 1px solid rgba(245,197,66,0.28);
            background: rgba(23, 32, 51, 0.92);
            color: var(--text);
            min-height: 42px;
            font-weight: 700;
            font-size: 15px;
            transition: all 0.18s ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--gold);
            color: var(--gold);
            transform: translateY(-1px);
        }

        .stButton > button[kind="primary"] {
            border: 0;
            color: #0B1020;
            background: linear-gradient(135deg, var(--gold), #FFD875);
            box-shadow: 0 14px 34px rgba(245,197,66,0.24);
            font-size: 16px;
            min-height: 48px;
        }

        div[data-baseweb="input"],
        div[data-baseweb="textarea"],
        div[data-baseweb="select"] {
            border-radius: 8px;
            border: 1px solid var(--border);
            background: rgba(17,24,39,0.92);
        }

        input, textarea {
            color: var(--text) !important;
            caret-color: var(--gold);
            font-size: 15px !important;
        }

        div[data-baseweb="select"] *,
        div[data-baseweb="popover"] *,
        [role="listbox"] *,
        [role="option"] {
            color: var(--text) !important;
        }

        div[data-baseweb="popover"],
        [role="listbox"] {
            background: #111827 !important;
            border: 1px solid var(--border) !important;
        }

        div[data-baseweb="select"] svg {
            color: #CBD5E1 !important;
            fill: #CBD5E1 !important;
        }

        textarea {
            min-height: 110px;
        }

        div[data-testid="stNumberInput"] input {
            color: var(--text) !important;
        }

        div[data-testid="stFileUploader"] section {
            background: rgba(17,24,39,0.78);
            border: 1px dashed rgba(56,189,248,0.35);
            border-radius: 8px;
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stDataEditor"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            background: rgba(17,24,39,0.72);
        }

        .hero-card {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 28px 30px;
            margin-bottom: 18px;
            background:
                linear-gradient(135deg, rgba(17,24,39,0.96), rgba(23,32,51,0.88)),
                linear-gradient(90deg, rgba(56,189,248,0.10), rgba(245,197,66,0.08));
            box-shadow: 0 22px 60px rgba(0,0,0,0.28);
        }

        .hero-card::before {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            background:
                linear-gradient(90deg, rgba(245,197,66,0.15), transparent 32%),
                linear-gradient(180deg, rgba(255,255,255,0.04), transparent 38%);
        }

        .hero-content {
            position: relative;
            display: grid;
            grid-template-columns: minmax(0, 1.4fr) minmax(360px, 0.8fr);
            gap: 28px;
            align-items: start;
        }

        .hero-title {
            margin: 0 0 8px;
            font-size: 38px;
            line-height: 1.14;
            font-weight: 850;
            color: var(--text);
        }

        .hero-subtitle {
            margin: 0 0 24px;
            color: var(--muted);
            font-size: 16px;
            line-height: 1.7;
        }

        .match-title {
            margin-top: 6px;
            color: var(--text);
            font-size: 30px;
            line-height: 1.22;
            font-weight: 780;
        }

        .hero-meta {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            margin-top: 18px;
        }

        .meta-item {
            padding: 12px 14px;
            border-radius: 8px;
            border: 1px solid var(--border);
            background: rgba(11,16,32,0.42);
        }

        .meta-label {
            color: #94A3B8;
            font-size: 12px;
            margin-bottom: 4px;
        }

        .meta-value {
            color: var(--text);
            font-size: 15px;
            font-weight: 700;
            overflow-wrap: anywhere;
        }

        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            min-height: 28px;
            padding: 5px 10px;
            border-radius: 999px;
            border: 1px solid rgba(56,189,248,0.30);
            color: #BAE6FD;
            background: rgba(56,189,248,0.10);
            font-size: 13px;
            font-weight: 700;
        }

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
        }

        .kpi-card {
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px 16px 15px;
            background: rgba(11,16,32,0.54);
            min-height: 118px;
        }

        .kpi-label {
            color: var(--muted);
            font-size: 14px;
            margin-bottom: 10px;
        }

        .kpi-value {
            color: var(--gold);
            font-size: 34px;
            line-height: 1.05;
            font-weight: 850;
            overflow-wrap: anywhere;
        }

        .kpi-caption {
            color: #94A3B8;
            font-size: 13px;
            margin-top: 10px;
        }

        .section-card {
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 18px 20px;
            margin: 18px 0 16px;
            background: rgba(17,24,39,0.82);
            box-shadow: 0 16px 36px rgba(0,0,0,0.20);
        }

        .section-title {
            margin: 0;
            color: var(--text);
            font-size: 20px;
            font-weight: 780;
        }

        .section-copy {
            margin: 7px 0 0;
            color: var(--muted);
            font-size: 15px;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 12px;
            margin-top: 14px;
        }

        .summary-item {
            padding: 14px;
            border-left: 3px solid var(--gold);
            background: rgba(11,16,32,0.45);
            border-radius: 7px;
        }

        .summary-label {
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 6px;
        }

        .summary-value {
            color: var(--text);
            font-size: 20px;
            font-weight: 800;
            line-height: 1.25;
            overflow-wrap: anywhere;
        }

        .muted-text {
            color: var(--muted);
            font-size: 15px;
            line-height: 1.7;
        }

        .success-card, .warning-card, .danger-card {
            border-radius: 8px;
            padding: 15px 16px;
            margin: 12px 0;
            font-size: 15px;
            line-height: 1.65;
        }

        .success-card {
            border: 1px solid rgba(34,197,94,0.32);
            background: rgba(34,197,94,0.10);
            color: #BBF7D0;
        }

        .warning-card {
            border: 1px solid rgba(245,158,11,0.34);
            background: rgba(245,158,11,0.11);
            color: #FDE68A;
        }

        .danger-card {
            border: 1px solid rgba(239,68,68,0.36);
            background: rgba(239,68,68,0.12);
            color: #FECACA;
        }

        .small-note {
            color: #94A3B8;
            font-size: 13px;
            line-height: 1.7;
        }

        @media (max-width: 980px) {
            .hero-content,
            .summary-grid {
                grid-template-columns: 1fr;
            }
            .kpi-grid,
            .hero-meta {
                grid-template-columns: 1fr;
            }
            .hero-title {
                font-size: 32px;
            }
            .match-title {
                font-size: 25px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_session() -> None:
    if "ui_form_state" not in st.session_state:
        st.session_state["ui_form_state"] = copy_default_form_state()
    if "ui_form_revision" not in st.session_state:
        st.session_state["ui_form_revision"] = 0
    for key in (
        "a_source_payload",
        "b_source_payload",
        "applied_a_source_payload",
        "applied_b_source_payload",
    ):
        st.session_state.setdefault(key, None)


def _key(name: str) -> str:
    return f"score_ui_{st.session_state['ui_form_revision']}_{name}"


def _value(name: str) -> Any:
    defaults = copy_default_form_state()
    return st.session_state["ui_form_state"].get(name, defaults.get(name))


def _active_state_snapshot() -> dict[str, Any]:
    state = copy_default_form_state()
    state.update(st.session_state.get("ui_form_state", {}))
    for name in state:
        widget_key = _key(name)
        if widget_key in st.session_state:
            state[name] = st.session_state[widget_key]
    return state


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _float_value(name: str, fallback: float) -> float:
    value = _value(name)
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _escape(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return html.escape(str(value))


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


def _fmt_signed_probability(value: Any) -> str:
    try:
        return f"{float(value) * 100:+.2f}%"
    except (TypeError, ValueError):
        return "—"


def _translate_metric(name: str) -> str:
    return METRIC_LABELS_ZH.get(name, name)


def _translate_market(name: str) -> str:
    return MARKET_LABELS_ZH.get(str(name), str(name))


def _translate_warning(warning: Any) -> str:
    text = str(warning)
    if text in WARNING_LABELS_ZH:
        return WARNING_LABELS_ZH[text]
    if text.startswith("high_1x2_margin:"):
        return f"胜平负固定奖金水位边际偏高（{text.split(':', 1)[1]}）"
    if text.startswith("high_ou_") and "_margin:" in text:
        line = text.removeprefix("high_ou_").split("_margin:", 1)[0]
        value = text.split(":", 1)[1]
        return f"大小球 {line} 水位边际偏高（{value}）"
    if text.startswith("high_btts_margin:"):
        return f"双方进球市场水位边际偏高（{text.split(':', 1)[1]}）"
    if text.startswith("v3_market_calibration_optimizer_fallback:"):
        return "V3 市场校准优化器使用兜底结果"
    if text.startswith("asian_handicap_consistency_conflict"):
        return "亚洲让球与胜平负或模型方向存在冲突"
    if text.startswith("sporttery_rqspf_consistency_conflict"):
        return "体彩让球胜平负与主盘口或模型方向存在冲突"
    if text.startswith("market_direction_conflict"):
        return "多市场方向存在冲突"
    return "模型提示：" + text.replace("_", " ")


def _risk_severity(warning: str) -> str:
    text = str(warning)
    if "conflict" in text or "optimizer_fallback" in text or "sensitive" in text:
        return "high"
    if "missing" in text or "margin" in text or "not_stable" in text:
        return "medium"
    return "low"


def _input_completion_score(state: dict[str, Any]) -> float:
    warnings = build_input_warnings(state)
    return max(0.0, min(1.0, 1.0 - len(warnings) / 8.0))


def _load_uploaded_yaml(uploaded_file: Any) -> None:
    marker = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("loaded_yaml_marker") == marker:
        return
    raw = uploaded_file.getvalue()
    payload = load_yaml_payload(raw)
    loaded = load_yaml_to_form_state(payload)
    st.session_state["ui_form_state"] = loaded
    st.session_state["ui_form_revision"] += 1
    st.session_state["loaded_yaml_marker"] = marker
    markets = payload.get("markets") if isinstance(payload.get("markets"), dict) else {}
    market = payload.get("market") if isinstance(payload.get("market"), dict) else {}
    source_text = " ".join(
        str(value)
        for value in (
            (markets.get("sporttery") or markets.get("value_comparison") or {}).get("source")
            if isinstance(markets.get("sporttery"), dict)
            or isinstance(markets.get("value_comparison"), dict)
            else "",
            market.get("source"),
            market.get("odds_source"),
            (market.get("odds_1x2") or {}).get("source")
            if isinstance(market.get("odds_1x2"), dict)
            else "",
        )
    ).lower()
    if markets.get("sporttery") or markets.get("value_comparison") or "sporttery" in source_text or "体彩" in source_text:
        st.session_state["b_source_payload"] = payload
        st.session_state["applied_b_source_payload"] = payload


def _split_match_name(match_name: str, state: dict[str, Any]) -> tuple[str, str]:
    if " vs " in match_name:
        home_team, away_team = match_name.split(" vs ", 1)
        return home_team or "主队", away_team or "客队"
    return str(state.get("home_team") or "主队"), str(state.get("away_team") or "客队")


def _v3_result(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    return result.get("v3") or {}


def _one_x_two_probs(result: dict[str, Any] | None) -> dict[str, float]:
    if not result:
        return {}
    v3_probs = (_v3_result(result).get("probabilities") or {}).get("one_x_two") or {}
    if v3_probs:
        return {
            "home": float(v3_probs.get("home", 0.0)),
            "draw": float(v3_probs.get("draw", 0.0)),
            "away": float(v3_probs.get("away", 0.0)),
        }
    legacy = result.get("probabilities") or {}
    return {
        "home": float(legacy.get("home_win", 0.0)),
        "draw": float(legacy.get("draw", 0.0)),
        "away": float(legacy.get("away_win", 0.0)),
    }


def _top_score(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not result:
        return None
    top_scores = _v3_result(result).get("top_scores") or result.get("top_scores") or []
    return top_scores[0] if top_scores else None


def _final_lambdas(result: dict[str, Any] | None) -> tuple[float | None, float | None]:
    if not result:
        return None, None
    flow = _v3_result(result).get("lambda_flow") or {}
    if flow:
        return flow.get("final_lambda_home"), flow.get("final_lambda_away")
    final_lambda = result.get("final_lambda") or {}
    return final_lambda.get("home"), final_lambda.get("away")


def _confidence_score(result: dict[str, Any] | None) -> float | None:
    if not result:
        return None
    confidence = _v3_result(result).get("confidence") or {}
    value = confidence.get("final_confidence_score")
    return float(value) if value is not None else None


def _prediction_direction(result: dict[str, Any] | None) -> str:
    probs = _one_x_two_probs(result)
    if not probs:
        return "不明确"
    labels = {"home": "主胜", "draw": "平局", "away": "客胜"}
    ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) < 2 or ranked[0][1] - ranked[1][1] < 0.03:
        return "不明确"
    return labels[ranked[0][0]]


def _score_matrix(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not result:
        return []
    return _v3_result(result).get("final_score_matrix") or []


def _total_goals_tendency(result: dict[str, Any] | None) -> str:
    matrix = _score_matrix(result)
    if not matrix:
        return "待预测"
    frame = score_matrix_to_frame(matrix)
    totals = frame["home_goals"] + frame["away_goals"]
    buckets = {
        "0-1 球": float(frame.loc[totals <= 1, "prob"].sum()),
        "2-3 球": float(frame.loc[(totals >= 2) & (totals <= 3), "prob"].sum()),
        "4 球及以上": float(frame.loc[totals >= 4, "prob"].sum()),
    }
    bucket = max(buckets, key=buckets.get)
    return f"{bucket}最集中（{_fmt_probability(buckets[bucket])}）"


def _risk_tips(result: dict[str, Any] | None, state: dict[str, Any]) -> list[str]:
    tips: list[str] = []
    v3 = _v3_result(result)
    if v3.get("btts_fit_error") is not None and float(v3["btts_fit_error"]) >= 0.08:
        tips.append("双方进球（BTTS）拟合误差偏高")
    if (
        v3.get("correct_score_fit_error") is not None
        and float(v3["correct_score_fit_error"]) >= 0.08
    ):
        tips.append("比分固定奖金校准误差偏高")
    for warning in (v3.get("risk_warnings") or [])[:4]:
        tips.append(_translate_warning(warning))
    for warning in build_input_warnings(state)[:3]:
        tips.append(_translate_warning(warning))
    unique = list(dict.fromkeys(tips))
    return unique or ["未发现显著高风险提示"]


def _section_card(title: str, body: str, badge: str | None = None) -> None:
    render_section_card(title, body, badge)


def _notice_card(message: str, level: str = "warning") -> None:
    render_status_card(message, level)


def _render_hero(state: dict[str, Any], result: dict[str, Any] | None) -> None:
    match_name = str(result.get("match") if result else "") if result else ""
    home_team, away_team = _split_match_name(match_name, state)
    top_score = _top_score(result)
    home_lambda, away_lambda = _final_lambdas(result)
    confidence = _confidence_score(result)
    one_x_two = _one_x_two_probs(result)
    completion = _input_completion_score(state)
    consistency = ((_v3_result(result).get("confidence") or {}).get("market_consistency_score"))
    risk_tips = _risk_tips(result, state)
    has_high_risk = any(_risk_severity(tip) == "high" for tip in risk_tips)
    if has_high_risk or completion < 0.6:
        risk_label = "高风险"
        risk_tone = "danger"
    elif len(risk_tips) > 1 or completion < 0.85:
        risk_label = "中风险"
        risk_tone = "warning"
    else:
        risk_label = "低风险"
        risk_tone = "success"
    expected_goals = (
        f"{float(home_lambda):.2f} : {float(away_lambda):.2f}"
        if home_lambda is not None and away_lambda is not None
        else "待预测"
    )
    mode = "市场模式" if state.get("market_only_mode") else "混合模式"
    dc_status = "Dixon-Coles 开启" if state.get("dc_enabled") else "Dixon-Coles 关闭"
    max_goals = state.get("max_goals") or "—"
    source_label = (
        "国际赔率通道已应用 / 体彩赔率通道保留"
        if st.session_state.get("applied_a_source_payload") or st.session_state.get("the_odds_a_source_payload")
        else state.get("odds_source") or "手动输入 / 体彩赔率通道"
    )
    data_health = (
        f"盘口一致性 {_fmt_probability(consistency)}"
        if consistency is not None
        else f"输入完整度 {_fmt_probability(completion)}"
    )
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-content">
                <div>
                    <div class="hero-title">世界杯比分预测引擎</div>
                    <div class="hero-subtitle">基于国际盘口、体彩数据与多市场联合校准的 90 分钟比分概率看板</div>
                    <div class="match-title">{_escape(home_team)} vs {_escape(away_team)}</div>
                    <div class="hero-meta">
                        <div class="meta-item">
                            <div class="meta-label">比赛时间</div>
                            <div class="meta-value">{_escape(state.get("date"))}</div>
                        </div>
                        <div class="meta-item">
                            <div class="meta-label">当前数据源</div>
                            <div class="meta-value">{_escape(source_label)}</div>
                        </div>
                        <div class="meta-item">
                            <div class="meta-label">赛事阶段</div>
                            <div class="meta-value">{_escape(state.get("competition"))} · {_escape(state.get("stage"))}</div>
                        </div>
                        <div class="meta-item">
                            <div class="meta-label">快照时间</div>
                            <div class="meta-value">{_escape(state.get("snapshot_time"))}</div>
                        </div>
                    </div>
                    <div class="badge-row">
                        {status_badge(mode, "accent")}
                        {status_badge(dc_status, "info")}
                        {status_badge(f"最大进球矩阵：{max_goals}", "info")}
                    </div>
                </div>
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <div class="kpi-label">主胜概率</div>
                        <div class="kpi-value">{_fmt_probability(one_x_two.get("home")) if one_x_two else "待预测"}</div>
                        <div class="kpi-caption">V3 胜平负分布</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-label">最可能比分</div>
                        <div class="kpi-value">{_escape(top_score.get("score") if top_score else "待预测")}</div>
                        <div class="kpi-caption">{_fmt_probability(top_score.get("prob")) if top_score else "等待模型输出"}</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-label">预期进球</div>
                        <div class="kpi-value">{_escape(expected_goals)}</div>
                        <div class="kpi-caption">主队 : 客队</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-label">模型置信度 / 输入完整度</div>
                        <div class="kpi-value">{_fmt_probability(confidence) if confidence is not None else _fmt_probability(_input_completion_score(state))}</div>
                        <div class="kpi-caption">置信评分不是命中概率</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-label">风险等级</div>
                        <div class="kpi-value">{_escape(risk_label)}</div>
                        <div class="kpi-caption">{status_badge(risk_label, risk_tone)}</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-label">数据完整度 / 盘口一致性</div>
                        <div class="kpi-value">{_escape(data_health)}</div>
                        <div class="kpi-caption">输入质量与市场稳定性摘要</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _resolve_event_fetch_selection(
    *,
    manual_event_id: Any,
    selected_event_id: Any,
    sport_key_input: Any,
    selected_sport_key: Any,
) -> dict[str, Any]:
    manual_id = str(manual_event_id or "").strip()
    selected_id = str(selected_event_id or "").strip()
    sport_key = str(sport_key_input or selected_sport_key or "").strip()
    return {
        "event_id": manual_id or selected_id,
        "sport_key": sport_key,
        "used_manual_event_id": bool(manual_id),
        "missing_sport_key": bool(manual_id and not sport_key),
        "missing_event_id": not bool(manual_id or selected_id),
    }


EXTRA_THE_ODDS_MARKETS = selectable_market_keys(
    list(dict.fromkeys(sum(MARKET_MODE_KEYS.values(), []))) + ["correct_score"]
)


def _split_market_keys(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _render_international_odds_tab() -> None:
    config = load_provider_config(THE_ODDS_API_CONFIG_PATH)
    groups = _load_worldcup_groups()
    latest_metadata = st.session_state.get("the_odds_api_metadata")
    render_section_card(
        "国际赔率通道",
        "The Odds API 只作为国际盘数据源。API key 仅从环境变量读取，不写入代码、YAML、日志或报告。",
        "primary calibration",
    )
    api_key_ready = os.getenv("THE_ODDS_API_KEY") is not None
    render_badge_row(
        [
            (f"API key：{_api_key_status()}", "success" if api_key_ready else "warning"),
            (f"剩余额度：{_quota_from_metadata(latest_metadata)}", "info"),
            (f"默认 markets：{config.get('default_markets')}", "accent"),
        ]
    )

    group_keys = sorted(groups.keys()) or ["A"]
    c1, c2, c3 = st.columns(3)
    with c1:
        group_key = st.selectbox("世界杯小组", group_keys, key="the_odds_group")
    teams = groups.get(group_key, [])
    team_labels = [team_label(team) for team in teams]
    with c2:
        home_label = st.selectbox("主队", team_labels, key="the_odds_home_team")
    home_team = team_from_label(teams, home_label)
    away_options = [label for label in team_labels if label != home_label]
    with c3:
        away_label = st.selectbox("客队", away_options, key="the_odds_away_team")
    away_team = team_from_label(teams, away_label)

    with st.expander("高级 API 参数", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            sport_key = st.text_input(
                "sport_key",
                value=st.session_state.get("the_odds_sport_key", ""),
                placeholder="留空自动查找",
                key="the_odds_sport_key_input",
            )
            regions = st.text_input(
                "regions",
                value=str(st.session_state.get("the_odds_regions", "eu")),
                placeholder="eu",
                key="the_odds_regions",
            )
        with c2:
            market_mode = st.selectbox(
                "盘口模式",
                list(MARKET_MODE_KEYS.keys()),
                index=0,
                key="the_odds_market_mode",
                help="省额度模式只拉核心盘口；完整建模模式增加 btts 和 alternate_totals；扩展审计模式再增加二级审计盘口。",
            )
            markets = st.text_input(
                "markets",
                value=",".join(market_keys_for_mode(market_mode)),
                placeholder="h2h,spreads,totals",
                key="the_odds_markets",
            )
            odds_format = st.text_input(
                "odds_format",
                value=str(config.get("default_odds_format", "decimal")),
                placeholder="decimal",
                key="the_odds_odds_format",
            )
        with c3:
            bookmaker = st.selectbox(
                "bookmaker",
                ["auto", "pinnacle", "betfair_ex_eu", "betfair_exchange", "bet365", "betfair", "matchbook"],
                index=1,
                key="the_odds_bookmaker",
            )
            event_id_manual = st.text_input(
                "event_id（手动）",
                value=st.session_state.get("the_odds_event_id_manual", ""),
                placeholder="未查到赛事时可手动填写",
                key="the_odds_event_id_manual",
            )
            if event_id_manual:
                st.session_state["the_odds_manual_event_id"] = event_id_manual
                st.session_state["the_odds_selected_event_id"] = event_id_manual

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        find_event = st.button("查找赛事", key="the_odds_find_event", use_container_width=True)
    with c2:
        query_markets = st.button("查询可用盘口", key="the_odds_query_markets", use_container_width=True)
    with c3:
        fetch_odds = st.button("拉取赔率", key="the_odds_fetch_odds", use_container_width=True)
    with c4:
        apply_to_a_source = st.button("应用到国际赔率通道", key="the_odds_apply", use_container_width=True)
    with c5:
        run_from_api = st.button("开始预测", key="the_odds_run_prediction", type="primary", use_container_width=True)

    if find_event:
        try:
            with st.spinner("正在查找赛事..."):
                resolved_sport_key = sport_key.strip()
                if not resolved_sport_key:
                    sports = fetch_sports(config=config)
                    st.session_state["the_odds_api_metadata"] = sports.metadata
                    resolved_sport_key = find_world_cup_sport_key(sports) or ""
                    if not resolved_sport_key:
                        _notice_card("未自动找到世界杯 sport_key，请在“高级 API 参数”中手动输入。", "warning")
                if resolved_sport_key:
                    events = fetch_events(resolved_sport_key, config=config)
                    st.session_state["the_odds_api_metadata"] = events.metadata
                    match = match_event_by_teams(
                        events,
                        api_match_name(home_team),
                        api_match_name(away_team),
                        team_aliases=_load_team_aliases(),
                    )
                    st.session_state["the_odds_sport_key"] = resolved_sport_key
                    st.session_state["the_odds_events"] = events.data
                    st.session_state["the_odds_event_candidates"] = match["candidates"]
                    st.session_state["the_odds_event_warnings"] = match["warnings"]
                    if match["warnings"]:
                        for warning in match["warnings"]:
                            _notice_card(warning, "warning")
                    else:
                        _notice_card("已找到候选赛事。", "success")
        except RuntimeError as exc:
            if str(exc) == MISSING_API_KEY_MESSAGE:
                _notice_card("API key 缺失：请在环境变量 THE_ODDS_API_KEY 中配置。", "danger")
            else:
                raise
        except Exception as exc:
            _notice_card(f"The Odds API 查找赛事失败：{exc}", "danger")

    candidates = st.session_state.get("the_odds_event_candidates") or []
    selected_event = None
    if candidates:
        options = [
            (
                f"{event.get('event_id') or event.get('id')} | "
                f"{event.get('api_home_team') or event.get('home_team')} vs "
                f"{event.get('api_away_team') or event.get('away_team')} | "
                f"{event.get('commence_time')} | "
                f"{float(event.get('match_quality') or 0.0):.2f}"
            )
            for event in candidates
        ]
        selected_label = st.selectbox("候选赛事", options, key="the_odds_selected_event")
        selected_index = options.index(selected_label)
        selected_event = candidates[selected_index]
        st.session_state["the_odds_selected_event_id"] = selected_event.get("event_id") or selected_event.get("id")
        st.session_state["the_odds_selected_sport_key"] = st.session_state.get("the_odds_sport_key")
        st.session_state["the_odds_selected_api_home_team"] = selected_event.get("api_home_team") or selected_event.get("home_team")
        st.session_state["the_odds_selected_api_away_team"] = selected_event.get("api_away_team") or selected_event.get("away_team")
        render_metadata_grid(
            [
                ("selected_event_id", st.session_state["the_odds_selected_event_id"]),
                ("selected_sport_key", st.session_state.get("the_odds_selected_sport_key", "—")),
                ("API 主队", st.session_state.get("the_odds_selected_api_home_team", "—")),
                ("API 客队", st.session_state.get("the_odds_selected_api_away_team", "—")),
                ("match_quality", f"{float(selected_event.get('match_quality') or 0.0):.2f}"),
                ("主客反向", "是" if selected_event.get("reversed_home_away") else "否"),
            ]
        )
    else:
        if st.session_state.get("the_odds_event_warnings"):
            for warning in st.session_state["the_odds_event_warnings"]:
                _notice_card(str(warning), "warning")
        render_empty_state(
            "尚未选择 API 赛事",
            "The Odds API 当前没有找到该比赛赛事，可能是该场未开放盘口、队名不匹配或赛事尚未收录。也可以在“高级 API 参数”中手动填写 sport_key 和 event_id。",
        )

    if query_markets:
        manual_event_id = str(st.session_state.get("the_odds_event_id_manual") or "").strip()
        selection = _resolve_event_fetch_selection(
            manual_event_id=manual_event_id,
            selected_event_id=st.session_state.get("the_odds_selected_event_id"),
            sport_key_input=sport_key.strip(),
            selected_sport_key="" if manual_event_id else st.session_state.get("the_odds_sport_key", ""),
        )
        event_id = selection["event_id"]
        resolved_sport_key = selection["sport_key"]
        if selection["missing_sport_key"] or selection["missing_event_id"] or not resolved_sport_key:
            _notice_card("请先查找赛事，或手动填写 sport_key 和 event_id 后再查询可用盘口。", "warning")
        else:
            try:
                with st.spinner("正在查询可用盘口..."):
                    available = fetch_event_markets(
                        resolved_sport_key,
                        event_id,
                        regions=regions,
                        config=config,
                    )
                    st.session_state["the_odds_api_metadata"] = available.metadata
                    st.session_state["the_odds_available_market_keys"] = extract_available_market_keys(available)
                    _notice_card("可用盘口查询完成。", "success")
            except RuntimeError as exc:
                if str(exc) == MISSING_API_KEY_MESSAGE:
                    _notice_card("API key 缺失：请在环境变量 THE_ODDS_API_KEY 中配置。", "danger")
                else:
                    raise
            except Exception as exc:
                _notice_card(f"The Odds API 查询可用盘口失败：{exc}", "danger")

    available_market_keys = st.session_state.get("the_odds_available_market_keys") or []
    markets_for_fetch = markets
    if available_market_keys:
        allowed = selectable_market_keys(available_market_keys)
        default_keys = [
            key for key in market_keys_for_mode(market_mode)
            if key in allowed
        ] or ["h2h", "spreads", "totals"]
        selected_market_keys = st.multiselect(
            "本场可拉取 markets",
            options=allowed,
            default=[key for key in default_keys if key in allowed],
            key="the_odds_selected_market_keys",
            help="只展示一级建模盘口和二级审计盘口；球员、角球、牌类和半场盘口暂不进入默认流程。",
        )
        markets_for_fetch = ",".join(selected_market_keys or default_keys)
        render_badge_row([(f"可用 market：{key}", "info") for key in available_market_keys])
        if not any("correct_score" in str(key) for key in available_market_keys):
            _notice_card(
                "国际赔率通道未返回正确比分盘口；可上传体彩比分固定奖金作为补充校准源。",
                "warning",
            )

    if fetch_odds:
        manual_event_id = str(st.session_state.get("the_odds_event_id_manual") or "").strip()
        selection = _resolve_event_fetch_selection(
            manual_event_id=manual_event_id,
            selected_event_id=st.session_state.get("the_odds_selected_event_id"),
            sport_key_input=sport_key.strip(),
            selected_sport_key="" if manual_event_id else st.session_state.get("the_odds_sport_key", ""),
        )
        event_id = selection["event_id"]
        resolved_sport_key = selection["sport_key"]
        if selection["missing_sport_key"]:
            _notice_card("已填写手动 event_id，请同时在“高级 API 参数”中填写 sport_key。", "warning")
        elif selection["missing_event_id"] or not resolved_sport_key:
            _notice_card("请先查找赛事，或在“高级 API 参数”中手动填写 sport_key 和 event_id。", "warning")
        else:
            try:
                with st.spinner("正在拉取并标准化赔率..."):
                    odds = fetch_event_odds(
                        resolved_sport_key,
                        event_id,
                        markets=markets_for_fetch,
                        regions=regions,
                        bookmakers=None if bookmaker == "auto" else bookmaker,
                        odds_format=odds_format,
                        config=config,
                    )
                    st.session_state["the_odds_api_metadata"] = odds.metadata
                    event_odds = odds.data
                    if selected_event and not selection["used_manual_event_id"]:
                        event_odds = {**selected_event, **event_odds}
                    normalized = normalize_event_odds_to_v3_input(
                        event_odds,
                        sport_key=resolved_sport_key,
                        event_id=event_id,
                        bookmaker=bookmaker,
                        preferred_bookmakers=config.get("preferred_bookmakers"),
                        metadata=odds.metadata,
                    )
                    st.session_state["the_odds_generated_payload"] = normalized["payload"]
                    st.session_state["the_odds_summary"] = normalized["summary"]
                    if selection["used_manual_event_id"]:
                        _notice_card("已使用手动 event_id 拉取赔率。", "success")
                    else:
                        _notice_card("赔率拉取并标准化完成。", "success")
            except RuntimeError as exc:
                if str(exc) == MISSING_API_KEY_MESSAGE:
                    _notice_card("API key 缺失：请在环境变量 THE_ODDS_API_KEY 中配置。", "danger")
                else:
                    raise
            except Exception as exc:
                _notice_card(f"The Odds API 拉取赔率失败：{exc}", "danger")

    summary = st.session_state.get("the_odds_summary")
    if summary:
        _section_card("拉取结果", "已从 The Odds API 解析国际赔率通道，可应用到 V3 兼容输入。", "API summary")
        render_metadata_grid(
            [
                ("event_id", summary.get("event_id", "—")),
                ("commence_time", summary.get("commence_time", "—")),
                ("API 主队", summary.get("api_home_team", "—")),
                ("API 客队", summary.get("api_away_team", "—")),
                ("选中 bookmaker", summary.get("selected_bookmaker", "—")),
                ("markets_found", ", ".join(summary.get("markets_found") or [])),
            ]
        )
        render_text_panel(", ".join(summary.get("bookmakers") or []) or "暂无 bookmaker 列表")
        selected_1x2 = pd.DataFrame([summary.get("selected_1x2") or {}]).rename(
            columns={"home": "主胜", "draw": "平局", "away": "客胜"}
        )
        selected_ou = pd.DataFrame(summary.get("selected_over_under") or []).rename(
            columns={"line": "盘口线", "over_odds": "大球赔率", "under_odds": "小球赔率"}
        )
        selected_alt_ou = pd.DataFrame(summary.get("selected_alternate_totals") or []).rename(
            columns={"line": "盘口线", "over_odds": "大球赔率", "under_odds": "小球赔率"}
        )
        selected_ah = pd.DataFrame(summary.get("selected_asian_handicap") or []).rename(
            columns={"line": "让球线", "home_odds": "主队赔率", "away_odds": "客队赔率"}
        )
        render_styled_table(selected_1x2, "selected 1X2", max_height=150)
        render_styled_table(selected_ou, "selected totals", max_height=220)
        if not selected_alt_ou.empty:
            render_styled_table(selected_alt_ou, "selected alternate_totals", max_height=260)
        render_styled_table(selected_ah, "selected spreads", max_height=220)
        if summary.get("selected_btts"):
            btts_frame = pd.DataFrame([summary["selected_btts"]]).rename(
                columns={"yes": "是", "no": "否"}
            )
            render_styled_table(btts_frame, "selected btts", max_height=150)
        audit_markets = summary.get("audit_markets") or {}
        audit_rows = [
            {"market": key, "items": len(value) if isinstance(value, list) else int(bool(value))}
            for key, value in audit_markets.items()
            if value
        ]
        if audit_rows:
            render_styled_table(pd.DataFrame(audit_rows), "国际二级审计盘口", max_height=180)
        if summary.get("warnings"):
            for warning in summary["warnings"]:
                _notice_card(warning, "warning")
        if summary.get("quota_headers"):
            quota_frame = pd.DataFrame([summary["quota_headers"]]).rename(
                columns={
                    "x_requests_remaining": "剩余额度",
                    "x_requests_used": "已用额度",
                    "x_requests_last": "本次消耗",
                }
            )
            render_styled_table(quota_frame, "quota headers", max_height=150)

    if apply_to_a_source:
        payload = st.session_state.get("the_odds_generated_payload")
        if not payload:
            _notice_card("请先拉取赔率。", "warning")
        else:
            st.session_state["the_odds_a_source_payload"] = payload
            st.session_state["a_source_payload"] = payload
            st.session_state["applied_a_source_payload"] = payload
            _notice_card("已应用到国际赔率通道。体彩赔率通道手动输入不会被覆盖。", "success")

    if run_from_api:
        payload = st.session_state.get("the_odds_generated_payload")
        if not payload:
            _notice_card("请先拉取赔率。", "warning")
        else:
            try:
                st.session_state["a_source_payload"] = payload
                st.session_state["applied_a_source_payload"] = payload
                merged_payload = merge_prediction_payload(
                    payload,
                    payload,
                    st.session_state.get("applied_b_source_payload")
                    or st.session_state.get("b_source_payload"),
                )
                match_input = match_input_from_dict(merged_payload)
                result = predict(match_input, dc_enabled=bool(match_input.settings.dc_enabled))
                st.session_state["prediction_result"] = result
                st.session_state["prediction_payload"] = merged_payload
                _notice_card("预测完成。", "success")
            except Exception as exc:
                _notice_card(f"预测失败，请检查输入数据：{exc}", "danger")


def _fit_errors_frame(v3: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fit_errors = v3.get("market_fit_errors") or {}
    one_x_two = fit_errors.get("one_x_two")
    if one_x_two:
        rows.append({"盘口类型": "胜平负", "盘口线": "", "RMSE 误差": one_x_two.get("rmse")})
    for line, row in (fit_errors.get("over_under") or {}).items():
        rows.append({"盘口类型": "大小球", "盘口线": line, "RMSE 误差": row.get("rmse")})
    spreads = fit_errors.get("spreads")
    if spreads:
        rows.append(
            {
                "盘口类型": "亚洲让球",
                "盘口线": spreads.get("line", ""),
                "RMSE 误差": spreads.get("rmse"),
            }
        )
    if v3.get("btts_fit_error") is not None:
        rows.append({"盘口类型": "双方进球", "盘口线": "", "RMSE 误差": v3.get("btts_fit_error")})
    if v3.get("correct_score_fit_error") is not None:
        rows.append(
            {
                "盘口类型": "比分固定奖金",
                "盘口线": "",
                "RMSE 误差": v3.get("correct_score_fit_error"),
            }
        )
    sporttery_total = fit_errors.get("sporttery_total_goals")
    if sporttery_total:
        rows.append({"盘口类型": "体彩总进球固定奖金", "盘口线": "0-6/7+", "RMSE 误差": sporttery_total.get("rmse")})
    sporttery_handicap = fit_errors.get("sporttery_handicap_3way")
    if sporttery_handicap:
        rows.append(
            {
                "盘口类型": "体彩让球胜平负",
                "盘口线": sporttery_handicap.get("line", ""),
                "RMSE 误差": sporttery_handicap.get("rmse"),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["RMSE 误差"] = frame["RMSE 误差"].astype(float)
    return frame


def _lambda_flow_frame(v3: dict[str, Any]) -> pd.DataFrame:
    flow = v3.get("lambda_flow") or {}
    fit = v3.get("joint_fit") or {}
    rows = [
        {"指标": _translate_metric("market_lambda_home"), "数值": _fmt_number(flow.get("market_prior_lambda_home"))},
        {"指标": _translate_metric("market_lambda_away"), "数值": _fmt_number(flow.get("market_prior_lambda_away"))},
        {"指标": _translate_metric("team_adjusted_lambda_home"), "数值": _fmt_number(flow.get("team_adjusted_lambda_home"))},
        {"指标": _translate_metric("team_adjusted_lambda_away"), "数值": _fmt_number(flow.get("team_adjusted_lambda_away"))},
        {"指标": _translate_metric("final_lambda_home"), "数值": _fmt_number(flow.get("final_lambda_home"))},
        {"指标": _translate_metric("final_lambda_away"), "数值": _fmt_number(flow.get("final_lambda_away"))},
        {"指标": _translate_metric("rho"), "数值": _fmt_number(fit.get("rho"))},
        {
            "指标": _translate_metric("dc_enabled"),
            "数值": "已开启" if fit.get("dc_enabled") else "已关闭",
        },
    ]
    return pd.DataFrame(rows)


def _confidence_frame(v3: dict[str, Any]) -> pd.DataFrame:
    confidence = v3.get("confidence") or {}
    rows = []
    for key in (
        "result_confidence",
        "score_confidence",
        "market_consistency_score",
        "data_quality_score",
        "sensitivity_stability_score",
        "final_confidence_score",
    ):
        rows.append({"指标": _translate_metric(key), "数值": _fmt_probability(confidence.get(key))})
    return pd.DataFrame(rows)


def _correct_score_calibration_frame(v3: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for row in v3.get("correct_score_calibration") or []:
        rows.append(
            {
                "通道": row.get("channel", "—"),
                "比分": row.get("score"),
                "市场概率": _fmt_probability(row.get("market_probability"), digits=2),
                "模型概率": _fmt_probability(row.get("model_probability"), digits=2),
                "误差": _fmt_signed_probability(row.get("difference")),
                "权重": _fmt_number(row.get("weight"), digits=3),
            }
        )
    return pd.DataFrame(rows)


def _sporttery_market_status_frame(v3: dict[str, Any]) -> pd.DataFrame:
    labels = {
        "sporttery_1x2": "胜平负",
        "sporttery_handicap_3way": "让球胜平负",
        "sporttery_correct_score": "比分固定奖金",
        "sporttery_total_goals": "总进球固定奖金",
        "sporttery_half_full": "半全场",
    }
    status_labels = {
        "soft_calibration": "参与 soft calibration",
        "audit_only": "audit-only",
        "ignored": "ignored",
    }
    rows = []
    for key, item in (v3.get("sporttery_market_status") or {}).items():
        rows.append(
            {
                "市场": labels.get(key, key),
                "状态": status_labels.get(str(item.get("status")), item.get("status")),
                "base_weight": _fmt_number(item.get("base_weight"), digits=2),
                "market_quality_score": _fmt_number(item.get("market_quality_score"), digits=2),
                "consistency_score": _fmt_number(item.get("consistency_score"), digits=2),
                "final_weight": _fmt_number(item.get("final_weight"), digits=3),
                "payout_rate": _fmt_probability(item.get("payout_rate")),
                "warnings": "，".join(str(warning) for warning in item.get("warnings") or []),
            }
        )
    return pd.DataFrame(rows)


def _channel_consistency_frame(v3: dict[str, Any]) -> pd.DataFrame:
    consistency = v3.get("channel_consistency") or {}
    return pd.DataFrame(
        [
            {"指标": "level", "数值": consistency.get("level", "—")},
            {"指标": "score", "数值": _fmt_probability(consistency.get("score"))},
            {
                "指标": "conflicts",
                "数值": "，".join(str(item) for item in consistency.get("conflicts") or []) or "—",
            },
            {
                "指标": "warnings",
                "数值": "，".join(str(item) for item in consistency.get("warnings") or []) or "—",
            },
        ]
    )


def _sensitivity_frame(v3: dict[str, Any]) -> pd.DataFrame:
    sensitivity = v3.get("sensitivity") or {}
    probability_ranges = sensitivity.get("result_probability_ranges") or {}
    rows = [
        {"指标": "稳定性评分", "数值": _fmt_probability(sensitivity.get("stability_score"))},
        {"指标": "场景数量", "数值": str(sensitivity.get("scenario_count", "—"))},
        {"指标": "Top 5 比分稳定", "数值": "是" if sensitivity.get("top_5_stable") else "否"},
        {"指标": "Top 5 最小重合度", "数值": _fmt_probability(sensitivity.get("top_5_overlap_min"))},
        {
            "指标": "最可能比分是否变化",
            "数值": "是" if sensitivity.get("most_likely_score_changed") else "否",
        },
        {
            "指标": "胜平负方向是否变化",
            "数值": "是" if sensitivity.get("result_direction_changed") else "否",
        },
    ]
    for key, label in (("home_win", "主胜概率范围"), ("draw", "平局概率范围"), ("away_win", "客胜概率范围")):
        value = probability_ranges.get(key) or {}
        rows.append(
            {
                "指标": label,
                "数值": f"{_fmt_probability(value.get('min'))} - {_fmt_probability(value.get('max'))}",
            }
        )
    over_range = sensitivity.get("over_2_5_probability_range") or {}
    rows.append(
        {
            "指标": "大 2.5 概率范围",
            "数值": f"{_fmt_probability(over_range.get('min'))} - {_fmt_probability(over_range.get('max'))}",
        }
    )
    return pd.DataFrame(rows)


def _top_scores_display_frame(top_scores: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "排名": index,
                "比分": row.get("score"),
                "概率": _fmt_probability(row.get("prob")),
            }
            for index, row in enumerate(top_scores[:5], start=1)
        ]
    )


def _over_under_display_frame(over_under: dict[str, dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "盘口线": line,
                "大": _fmt_probability(values.get("over")),
                "小": _fmt_probability(values.get("under")),
            }
            for line, values in over_under.items()
        ]
    )


def _render_downloads(
    result: dict[str, Any],
    payload: dict[str, Any],
    key_prefix: str = "results",
) -> None:
    v3 = result.get("v3") or {}
    score_frame = score_matrix_to_frame(v3.get("final_score_matrix") or [])
    try:
        report = format_human_report(result)
    except Exception as exc:  # pragma: no cover - defensive UI fallback
        report = f"分析报告生成失败：{exc}"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            "下载输入 YAML",
            data=dump_yaml(payload),
            file_name="downloaded_match.yaml",
            mime="text/yaml",
            key=f"{key_prefix}_download_input_yaml",
        )
    with c2:
        st.download_button(
            "下载预测 JSON",
            data=to_pretty_json(result),
            file_name="prediction.json",
            mime="application/json",
            key=f"{key_prefix}_download_prediction_json",
        )
    with c3:
        st.download_button(
            "下载比分矩阵 CSV",
            data=score_frame.to_csv(index=False),
            file_name="score_matrix.csv",
            mime="text/csv",
            key=f"{key_prefix}_download_score_matrix_csv",
        )
    with c4:
        st.download_button(
            "下载分析报告 Markdown",
            data=report,
            file_name="prediction_report.md",
            mime="text/markdown",
            key=f"{key_prefix}_download_markdown_report",
        )


def _render_conclusion_summary(result: dict[str, Any], state: dict[str, Any]) -> None:
    probs = _one_x_two_probs(result)
    top_score = _top_score(result)
    risk_tips = _risk_tips(result, state)
    st.markdown(
        f"""
        <div class="section-card">
            <p class="section-title">结论摘要</p>
            <p class="section-copy">优先展示模型结论，再进入图表和高级诊断。</p>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-label">预测倾向</div>
                    <div class="summary-value">{_escape(_prediction_direction(result))}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">最可能比分</div>
                    <div class="summary-value">{_escape(top_score.get("score") if top_score else "待预测")}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">胜平负概率</div>
                    <div class="summary-value">主胜 {_fmt_probability(probs.get("home"))}<br>平局 {_fmt_probability(probs.get("draw"))}<br>客胜 {_fmt_probability(probs.get("away"))}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">总进球倾向</div>
                    <div class="summary-value">{_escape(_total_goals_tendency(result))}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">风险提示</div>
                    <div class="summary-value">{_escape("；".join(risk_tips[:3]))}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_result(result: dict[str, Any], payload: dict[str, Any], state: dict[str, Any]) -> None:
    v3 = result.get("v3") or {}
    score_matrix = v3.get("final_score_matrix") or []
    match_name = str(result.get("match", "主队 vs 客队"))
    home_team, away_team = _split_match_name(match_name, state)

    _render_conclusion_summary(result, state)

    if state.get("market_only_mode"):
        _notice_card("当前为仅盘口模式：不使用人工预期进球（internal lambda），预期进球由盘口自动反推。", "success")

    _section_card("概率图表", "比分矩阵、胜平负、总进球、大小球和双方进球的核心可视化。", "V3 看板")
    c1, c2 = st.columns([0.95, 1.05])
    with c1:
        one_x_two = _one_x_two_probs(result)
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

    top_scores = list(v3.get("top_scores") or [])[:5]
    c1, c2 = st.columns([0.82, 1.18])
    with c1:
        render_styled_table(
            _top_scores_display_frame(top_scores),
            "最可能比分 Top 5",
            max_height=260,
        )
    with c2:
        st.plotly_chart(plot_top_scores(top_scores), use_container_width=True)

    over_under = (v3.get("probabilities") or {}).get("over_under") or {}
    c1, c2 = st.columns([1.08, 0.92])
    with c1:
        st.plotly_chart(plot_over_under_table(over_under), use_container_width=True)
    with c2:
        render_styled_table(
            _over_under_display_frame(over_under),
            "大小球概率表",
            max_height=260,
        )

    btts = (v3.get("probabilities") or {}).get("btts") or {}
    st.plotly_chart(
        plot_btts_bar(float(btts.get("yes", 0.0)), float(btts.get("no", 0.0))),
        use_container_width=True,
    )

    with st.expander("高级模型诊断", expanded=False):
        st.markdown("#### Lambda 过程")
        render_styled_table(_lambda_flow_frame(v3), "Lambda 过程", max_height=330)

        st.markdown("#### Dixon-Coles rho")
        fit = v3.get("joint_fit") or {}
        dc_rows = pd.DataFrame(
            [
                {"指标": _translate_metric("rho"), "数值": _fmt_number(fit.get("rho"))},
                {
                    "指标": _translate_metric("dc_enabled"),
                    "数值": "已开启" if fit.get("dc_enabled") else "已关闭",
                },
                {"指标": _translate_metric("loss"), "数值": _fmt_number(fit.get("loss"), digits=5)},
                {
                    "指标": _translate_metric("optimizer_success"),
                    "数值": "成功" if fit.get("optimizer_success") else "使用兜底",
                },
            ]
        )
        render_styled_table(dc_rows, "Dixon-Coles rho", max_height=220)

        fit_errors = _fit_errors_frame(v3)
        st.markdown("#### 盘口拟合误差")
        if fit_errors.empty:
            render_empty_state("暂无盘口拟合误差", "当前预测没有返回盘口拟合误差表。")
        else:
            render_styled_table(fit_errors, "盘口拟合误差", max_height=280)
            st.plotly_chart(plot_market_fit_errors(fit_errors), use_container_width=True)

        st.markdown("#### 体彩赔率通道状态")
        sporttery_status = _sporttery_market_status_frame(v3)
        if sporttery_status.empty:
            render_empty_state("暂无体彩赔率通道状态", "当前预测没有识别到体彩 YAML 市场。")
        else:
            render_styled_table(sporttery_status, "体彩赔率通道状态", max_height=360)

        st.markdown("#### 盘口源一致性")
        render_styled_table(_channel_consistency_frame(v3), "盘口源一致性", max_height=220)

        st.markdown("#### 敏感性分析")
        render_styled_table(_sensitivity_frame(v3), "敏感性分析", max_height=360)

        st.markdown("#### 比分固定奖金校准（Correct Score calibration）")
        correct_score = _correct_score_calibration_frame(v3)
        if correct_score.empty:
            render_empty_state("暂无比分固定奖金校准表", "当前输入没有可展示的 Correct Score calibration 数据。")
        else:
            render_styled_table(correct_score, "比分固定奖金校准", max_height=360)

        st.markdown("#### 综合置信评分拆解")
        render_styled_table(_confidence_frame(v3), "综合置信评分拆解", max_height=280)
        st.caption("综合置信评分不是命中概率，只代表输入质量、盘口一致性和模型稳定性。")

        st.markdown("#### 原始 JSON / YAML 下载")
        _render_downloads(result, payload, key_prefix="diagnostics")


def _warning_frame(warnings: list[Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "风险级别": {"low": "低风险", "medium": "中风险", "high": "高风险"}[
                    _risk_severity(str(warning))
                ],
                "提示": _translate_warning(warning),
            }
            for warning in warnings
        ]
    )


def _render_audit(result: dict[str, Any] | None, state: dict[str, Any]) -> None:
    input_warnings = build_input_warnings(state)
    margin_warnings = build_margin_warnings(state)
    audit_only_warnings = audit_only_score_warning(state)
    all_local_warnings = input_warnings + margin_warnings + audit_only_warnings

    _section_card(
        "审计摘要",
        "综合检查输入完整性、盘口一致性、市场拟合误差和模型稳定性。综合置信评分不是命中概率。",
        "risk audit",
    )
    completion = _input_completion_score(state)
    level = "low" if completion >= 0.85 else "medium" if completion >= 0.6 else "high"
    render_badge_row(
        [
            (f"输入完整度：{_fmt_probability(completion)}", "success" if level == "low" else "warning" if level == "medium" else "danger"),
            ("低风险" if level == "low" else "中风险" if level == "medium" else "高风险", "success" if level == "low" else "warning" if level == "medium" else "danger"),
        ]
    )

    _section_card("输入完整性", "检查比赛基础信息、胜平负、总进球、模式参数是否完整。")
    _notice_card(f"输入完整度：{_fmt_probability(completion)}", level)
    if all_local_warnings:
        render_styled_table(_warning_frame(all_local_warnings), "输入完整性提示", max_height=320)
    else:
        _notice_card("输入项完整，未发现本地表单缺失。", "success")

    if result is None:
        _section_card("盘口一致性", "完成预测后展示多市场方向一致性。")
        render_empty_state("尚未生成预测", "盘口一致性和模型拟合误差将在预测完成后显示。")
        _section_card("免责声明", DISCLAIMER_ZH)
        return

    v3 = result.get("v3") or {}
    consistency = v3.get("handicap_consistency") or {}
    confidence = v3.get("confidence") or {}

    _section_card("盘口一致性", "对胜平负、让球胜平负、亚洲让球和模型方向做一致性审计。")
    consistency_score = float(consistency.get("score", 0.0))
    consistency_level = "low" if consistency_score >= 0.8 else "medium" if consistency_score >= 0.55 else "high"
    _notice_card(f"盘口一致性评分：{_fmt_probability(consistency_score)}", consistency_level)
    directions = consistency.get("directions") or {}
    if directions:
        render_styled_table(
            pd.DataFrame(
                [
                    {
                        "来源": DIRECTION_LABELS_ZH.get(source, source),
                        "方向": DIRECTION_LABELS_ZH.get(direction, direction),
                    }
                    for source, direction in directions.items()
                ]
            ),
            "盘口方向一致性",
            max_height=260,
        )

    channel_consistency = v3.get("channel_consistency") or {}
    if channel_consistency:
        _section_card("盘口源一致性", "检查国际赔率通道与体彩赔率通道是否存在方向分歧。")
        render_styled_table(_channel_consistency_frame(v3), "盘口源一致性", max_height=220)

    sporttery_status = _sporttery_market_status_frame(v3)
    if not sporttery_status.empty:
        _section_card("体彩赔率通道市场质量", "展示每个体彩市场的参与状态、返还率、质量评分和最终权重。")
        render_styled_table(sporttery_status, "体彩赔率通道市场质量", max_height=360)

    _section_card("市场拟合误差", "RMSE 越高，代表模型分布与对应盘口隐含概率偏离越大。")
    fit_errors = _fit_errors_frame(v3)
    if fit_errors.empty:
        render_empty_state("暂无市场拟合误差", "当前预测没有返回市场拟合误差。")
    else:
        render_styled_table(fit_errors, "市场拟合误差", max_height=280)
        st.plotly_chart(plot_market_fit_errors(fit_errors), use_container_width=True)

    _section_card("高风险提示", "聚合输入、盘口一致性、V3 校准和敏感性分析产生的风险提示。")
    all_warnings = []
    all_warnings.extend(result.get("warnings") or [])
    all_warnings.extend(v3.get("risk_warnings") or [])
    all_warnings.extend(all_local_warnings)
    all_warnings = list(dict.fromkeys(all_warnings))
    if all_warnings:
        for warning in all_warnings:
            severity = _risk_severity(str(warning))
            label = {"low": "低风险", "medium": "中风险", "high": "高风险"}[severity]
            _notice_card(f"{label}：{_translate_warning(warning)}", severity)
    else:
        _notice_card("低风险：未发现显著风险提示。", "low")

    _section_card("模型解释", "综合置信评分用于衡量输入质量、盘口一致性和模型稳定性。")
    explanation_rows = [
        {
            "指标": _translate_metric("data_quality_score"),
            "数值": _fmt_probability(confidence.get("data_quality_score")),
        },
        {
            "指标": _translate_metric("market_consistency_score"),
            "数值": _fmt_probability(confidence.get("market_consistency_score")),
        },
        {
            "指标": _translate_metric("sensitivity_stability_score"),
            "数值": _fmt_probability(confidence.get("sensitivity_stability_score")),
        },
        {
            "指标": _translate_metric("final_confidence_score"),
            "数值": _fmt_probability(confidence.get("final_confidence_score")),
        },
    ]
    render_styled_table(pd.DataFrame(explanation_rows), "综合置信评分拆解", max_height=260)
    _notice_card("综合置信评分不是命中概率，只代表输入质量、盘口一致性和模型稳定性。", "warning")

    _section_card("免责声明", DISCLAIMER_ZH)


def main() -> None:
    st.set_page_config(
        page_title="世界杯比分预测引擎",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_professional_theme_css()
    _init_session()

    hero_state = _active_state_snapshot()
    _render_hero(hero_state, st.session_state.get("prediction_result"))

    (
        match_tab,
        international_tab,
        markets_tab,
        context_tab,
        settings_tab,
        results_tab,
        audit_tab,
    ) = st.tabs(
        [
            "比赛信息",
            "国际赔率通道",
            "体彩赔率通道",
            "赛前情报",
            "模型参数",
            "预测看板",
            "审计与风险",
        ]
    )

    with match_tab:
        _section_card("比赛信息", "上传 YAML 后会自动填充表单，也可以在下方手动维护比赛基础信息。", "YAML")
        uploaded = st.file_uploader(
            "上传 YAML 输入文件",
            type=["yaml", "yml"],
            help="保留手动 YAML 上传能力，用于复用示例或历史盘口输入。",
        )
        if uploaded is not None:
            try:
                _load_uploaded_yaml(uploaded)
                _notice_card("YAML 已加载，表单已自动填充。", "success")
            except Exception as exc:
                _notice_card(f"YAML 加载失败：{exc}", "danger")

        c1, c2, c3 = st.columns(3)
        with c1:
            match_id = st.text_input("比赛 ID", value=_text(_value("match_id")), key=_key("match_id"))
            date = st.text_input("比赛时间", value=_text(_value("date")), key=_key("date"))
            timezone = st.text_input("时区", value=_text(_value("timezone")), key=_key("timezone"))
        with c2:
            home_team = st.text_input("主队", value=_text(_value("home_team")), key=_key("home_team"))
            away_team = st.text_input("客队", value=_text(_value("away_team")), key=_key("away_team"))
            competition = st.text_input("赛事", value=_text(_value("competition")), key=_key("competition"))
        with c3:
            stage = st.text_input("阶段", value=_text(_value("stage")), key=_key("stage"))
            neutral_site = st.checkbox("中立场", value=bool(_value("neutral_site")), key=_key("neutral_site"))
            odds_source = st.text_input("数据源", value=_text(_value("odds_source")), key=_key("odds_source"))
            snapshot_time = st.text_input("盘口快照时间", value=_text(_value("snapshot_time")), key=_key("snapshot_time"))

    with international_tab:
        _render_international_odds_tab()

    with markets_tab:
        _section_card("胜平负固定奖金", "胜平负固定奖金用于约束比赛方向。", "核心盘口")
        c1, c2, c3 = st.columns(3)
        with c1:
            odds_home_win = st.number_input(
                "主胜固定奖金",
                min_value=1.01,
                value=_float_value("odds_home_win", 1.85),
                step=0.01,
                key=_key("odds_home_win"),
            )
        with c2:
            odds_draw = st.number_input(
                "平局固定奖金",
                min_value=1.01,
                value=_float_value("odds_draw", 3.45),
                step=0.01,
                key=_key("odds_draw"),
            )
        with c3:
            odds_away_win = st.number_input(
                "客胜固定奖金",
                min_value=1.01,
                value=_float_value("odds_away_win", 4.40),
                step=0.01,
                key=_key("odds_away_win"),
            )

        _section_card("让球胜平负", "让球胜平负用于审计盘口方向一致性；当前不直接进入主模型 lambda 优化。")
        c1, c2, c3 = st.columns(3)
        with c1:
            rqspf_handicap = st.text_input("体彩让球数", value=_text(_value("rqspf_handicap")), key=_key("rqspf_handicap"))
            asian_handicap_line = st.text_input("亚洲让球线", value=_text(_value("asian_handicap_line")), key=_key("asian_handicap_line"))
        with c2:
            rqspf_home_odds = st.text_input("让球主胜固定奖金", value=_text(_value("rqspf_home_odds")), key=_key("rqspf_home_odds"))
            rqspf_draw_odds = st.text_input("让球平局固定奖金", value=_text(_value("rqspf_draw_odds")), key=_key("rqspf_draw_odds"))
            asian_handicap_home_odds = st.text_input("亚洲让球主队奖金", value=_text(_value("asian_handicap_home_odds")), key=_key("asian_handicap_home_odds"))
        with c3:
            rqspf_away_odds = st.text_input("让球客胜固定奖金", value=_text(_value("rqspf_away_odds")), key=_key("rqspf_away_odds"))
            asian_handicap_away_odds = st.text_input("亚洲让球客队奖金", value=_text(_value("asian_handicap_away_odds")), key=_key("asian_handicap_away_odds"))

        _section_card("比分固定奖金", "比分固定奖金用于低权重校准具体比分。", "比分校准")
        correct_score_rows = st.data_editor(
            pd.DataFrame(rows_from_table(_value("correct_score_rows"))),
            num_rows="dynamic",
            use_container_width=True,
            height=440,
            key=_key("correct_score_rows"),
            column_config={
                "score": st.column_config.TextColumn("比分"),
                "odds": st.column_config.NumberColumn("固定奖金", min_value=1.01, step=0.01),
            },
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            home_other = st.text_input("主胜其他比分", value=_text(_value("home_other")), key=_key("home_other"))
        with c2:
            draw_other = st.text_input("平局其他比分", value=_text(_value("draw_other")), key=_key("draw_other"))
        with c3:
            away_other = st.text_input("客胜其他比分", value=_text(_value("away_other")), key=_key("away_other"))

        _section_card("总进球固定奖金", "总进球固定奖金用于约束总进球分布。", "大小球")
        ou_rows = st.data_editor(
            pd.DataFrame(rows_from_table(_value("ou_rows"))),
            num_rows="dynamic",
            use_container_width=True,
            height=240,
            key=_key("ou_rows"),
            column_config={
                "line": st.column_config.NumberColumn("盘口线", step=0.5),
                "over_odds": st.column_config.NumberColumn("大球固定奖金", min_value=1.01, step=0.01),
                "under_odds": st.column_config.NumberColumn("小球固定奖金", min_value=1.01, step=0.01),
            },
        )

        _section_card("双方进球固定奖金", "双方进球市场用于辅助校准双方是否都有进球的概率。", "双方进球")
        c1, c2 = st.columns(2)
        with c1:
            btts_yes_odds = st.number_input(
                "双方进球：是",
                min_value=1.01,
                value=_float_value("btts_yes_odds", 1.70),
                step=0.01,
                key=_key("btts_yes_odds"),
            )
        with c2:
            btts_no_odds = st.number_input(
                "双方进球：否",
                min_value=1.01,
                value=_float_value("btts_no_odds", 2.15),
                step=0.01,
                key=_key("btts_no_odds"),
            )

        _section_card("半全场", "半全场当前仅做审计参考，不参与主模型。")
        render_empty_state("半全场审计入口", "当前 UI 未接入半全场字段；保留该模块作为体彩盘口审计入口。")

    with context_tab:
        _section_card(
            "赛前事实情报",
            "只录入事实信息：官方首发、伤停、停赛、主力轮换、休息天数、天气、场地、比赛性质和小组出线形势。不要录入专家推荐、媒体预测、投注建议、盘口解读或网友观点。",
            "人工输入",
        )
        _notice_card(
            "赛前情报通过轻量乘法修正影响 lambda_home / lambda_away / total_goals，修正幅度受 team_adjustment_strength 限制；单个事实通常 1%-5%，重大阵容事实最多 8%-12%，总修正建议限制在 ±15%，默认不应推翻市场盘口。",
            "warning",
        )
        c1, c2 = st.columns(2)
        with c1:
            home_elo = st.text_input("主队 Elo", value=_text(_value("home_elo")), key=_key("home_elo"))
            home_fifa_rank = st.text_input("主队 FIFA 排名", value=_text(_value("home_fifa_rank")), key=_key("home_fifa_rank"))
            home_rest_days = st.text_input("主队休息天数", value=_text(_value("home_rest_days")), key=_key("home_rest_days"))
            home_key_players_missing = st.text_area(
                "主队关键缺阵",
                value=_text(_value("home_key_players_missing")),
                key=_key("home_key_players_missing"),
            )
            home_lineup_strength = st.text_input("主队阵容强度", value=_text(_value("home_lineup_strength")), key=_key("home_lineup_strength"))
        with c2:
            away_elo = st.text_input("客队 Elo", value=_text(_value("away_elo")), key=_key("away_elo"))
            away_fifa_rank = st.text_input("客队 FIFA 排名", value=_text(_value("away_fifa_rank")), key=_key("away_fifa_rank"))
            away_rest_days = st.text_input("客队休息天数", value=_text(_value("away_rest_days")), key=_key("away_rest_days"))
            away_key_players_missing = st.text_area(
                "客队关键缺阵",
                value=_text(_value("away_key_players_missing")),
                key=_key("away_key_players_missing"),
            )
            away_lineup_strength = st.text_input("客队阵容强度", value=_text(_value("away_lineup_strength")), key=_key("away_lineup_strength"))

        weather_note = st.text_area("天气说明", value=_text(_value("weather_note")), key=_key("weather_note"))
        injury_notes = st.text_area("伤停说明", value=_text(_value("injury_notes")), key=_key("injury_notes"))
        lineup_notes = st.text_area("首发 / 阵容说明", value=_text(_value("lineup_notes")), key=_key("lineup_notes"))
        schedule_notes = st.text_area("赛程说明", value=_text(_value("schedule_notes")), key=_key("schedule_notes"))
        motivation_notes = st.text_area("战意 / 重要性说明", value=_text(_value("motivation_notes")), key=_key("motivation_notes"))
        source_notes = st.text_area("信息来源备注", value=_text(_value("source_notes")), key=_key("source_notes"))

    with settings_tab:
        _section_card("模型参数", "参数只影响 UI 调用时传入的设置，不修改 V3 核心模型逻辑。", "V3")
        _section_card(
            "双赔率通道",
            "国际赔率通道用于 primary calibration；体彩 YAML 用于 supplemental soft calibration。体彩固定奖金以低权重参与多市场联合校准，比分固定奖金和总进球固定奖金会先去水，再按市场质量与盘口一致性自动降权。",
            "odds_channels",
        )
        c1, c2 = st.columns(2)
        with c1:
            calibration_sources = st.text_area(
                "国际校准来源",
                value=_text(_value("calibration_sources")),
                key=_key("calibration_sources"),
                help="每行一个源，例如 international、pinnacle、betfair、bet365。",
            )
        with c2:
            value_comparison_sources = st.text_area(
                "体彩补充校准来源",
                value=_text(_value("value_comparison_sources")),
                key=_key("value_comparison_sources"),
                help="每行一个源，例如 sporttery。体彩通道是低权重 soft calibration。",
            )
        c1, c2, c3 = st.columns(3)
        with c1:
            dc_enabled = st.checkbox(
                "启用 Dixon-Coles 低比分修正",
                value=bool(_value("dc_enabled")),
                key=_key("dc_enabled"),
                help="是否启用 Dixon-Coles 低比分修正，主要影响 0-0、1-0、0-1、1-1。",
            )
            market_only_mode = st.checkbox(
                "仅盘口模式",
                value=bool(_value("market_only_mode")),
                key=_key("market_only_mode"),
                help="对应 YAML key: market_only_mode。不使用人工预期进球（internal lambda），完全由盘口自动反推预期进球。",
            )
            max_goals = st.number_input(
                "最大进球矩阵",
                min_value=3,
                max_value=20,
                value=int(_value("max_goals") or 8),
                step=1,
                key=_key("max_goals"),
                help="比分矩阵最大进球数，一般设为 8。",
            )
        with c2:
            market_weight = st.number_input(
                "市场盘口权重",
                min_value=0.0,
                max_value=1.0,
                value=_float_value("market_weight", 1.0),
                step=0.05,
                key=_key("market_weight"),
                help="市场盘口权重。仅盘口模式建议 1.00。",
            )
            correct_score_weight = st.number_input(
                "比分固定奖金校准权重",
                min_value=0.0,
                max_value=5.0,
                value=_float_value("correct_score_weight", 0.35),
                step=0.05,
                key=_key("correct_score_weight"),
                help="比分市场水位高，建议 0.25-0.35。",
            )
            btts_weight = st.number_input(
                "双方进球市场权重",
                min_value=0.0,
                max_value=5.0,
                value=_float_value("btts_weight", 0.6),
                step=0.05,
                key=_key("btts_weight"),
                help="双方进球市场权重。",
            )
        with c3:
            ou_weight = st.number_input(
                "大小球 / 总进球市场权重",
                min_value=0.0,
                max_value=5.0,
                value=_float_value("ou_weight", 1.0),
                step=0.05,
                key=_key("ou_weight"),
                help="决定总进球期望。",
            )
            x1x2_weight = st.number_input(
                "胜平负市场权重",
                min_value=0.0,
                max_value=5.0,
                value=_float_value("x1x2_weight", 1.0),
                step=0.05,
                key=_key("x1x2_weight"),
                help="决定比赛方向。",
            )
            team_adjustment_strength = st.number_input(
                "赛前情报修正强度",
                min_value=0.0,
                max_value=5.0,
                value=_float_value("team_adjustment_strength", 1.0),
                step=0.05,
                key=_key("team_adjustment_strength"),
                help="没有可靠情报时不要过度调高。",
            )

        _section_card("国际一级盘口权重", "一级盘口参与 V3 loss；二级盘口只进入审计，不影响 lambda。")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            h2h_weight = st.number_input(
                "h2h 权重",
                min_value=0.0,
                max_value=5.0,
                value=_float_value("h2h_weight", 1.0),
                step=0.05,
                key=_key("h2h_weight"),
            )
        with c2:
            totals_weight = st.number_input(
                "totals 权重",
                min_value=0.0,
                max_value=5.0,
                value=_float_value("totals_weight", 1.0),
                step=0.05,
                key=_key("totals_weight"),
            )
        with c3:
            alternate_totals_weight = st.number_input(
                "alternate_totals 权重",
                min_value=0.0,
                max_value=5.0,
                value=_float_value("alternate_totals_weight", 0.8),
                step=0.05,
                key=_key("alternate_totals_weight"),
            )
        with c4:
            spreads_weight = st.number_input(
                "spreads 权重",
                min_value=0.0,
                max_value=5.0,
                value=_float_value("spreads_weight", 0.5),
                step=0.05,
                key=_key("spreads_weight"),
            )

        _section_card("体彩补充校准权重", "体彩通道是 supplemental soft calibration，最终权重会继续按市场质量和盘口一致性自动修正。")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            sporttery_1x2_weight = st.number_input(
                "体彩胜平负权重",
                min_value=0.05,
                max_value=0.35,
                value=_float_value("sporttery_1x2_weight", 0.15),
                step=0.01,
                key=_key("sporttery_1x2_weight"),
            )
        with c2:
            sporttery_handicap_3way_weight = st.number_input(
                "体彩让球胜平负权重",
                min_value=0.05,
                max_value=0.30,
                value=_float_value("sporttery_handicap_3way_weight", 0.15),
                step=0.01,
                key=_key("sporttery_handicap_3way_weight"),
            )
        with c3:
            sporttery_total_goals_weight = st.number_input(
                "体彩总进球权重",
                min_value=0.10,
                max_value=0.45,
                value=_float_value("sporttery_total_goals_weight", 0.30),
                step=0.01,
                key=_key("sporttery_total_goals_weight"),
            )
        with c4:
            sporttery_correct_score_weight = st.number_input(
                "体彩比分权重",
                min_value=0.10,
                max_value=0.35,
                value=_float_value("sporttery_correct_score_weight", 0.20),
                step=0.01,
                key=_key("sporttery_correct_score_weight"),
            )
        with c5:
            sporttery_half_full_weight = st.number_input(
                "半全场权重",
                min_value=0.0,
                max_value=0.0,
                value=0.0,
                step=0.01,
                key=_key("sporttery_half_full_weight"),
                disabled=True,
            )

        internal_lambda_home = None
        internal_lambda_away = None
        if not market_only_mode:
            _section_card("人工预期进球（internal lambda）", "混合模式下需要输入人工预期进球；仅盘口模式会隐藏该组参数。")
            c1, c2 = st.columns(2)
            with c1:
                internal_lambda_home = st.number_input(
                    "主队人工预期进球",
                    min_value=0.05,
                    max_value=5.0,
                    value=_float_value("internal_lambda_home", 1.4),
                    step=0.05,
                    key=_key("internal_lambda_home"),
                )
            with c2:
                internal_lambda_away = st.number_input(
                    "客队人工预期进球",
                    min_value=0.05,
                    max_value=5.0,
                    value=_float_value("internal_lambda_away", 1.0),
                    step=0.05,
                    key=_key("internal_lambda_away"),
                )

    current_state = {
        "match_id": match_id,
        "date": date,
        "home_team": home_team,
        "away_team": away_team,
        "competition": competition,
        "stage": stage,
        "neutral_site": neutral_site,
        "odds_source": odds_source,
        "snapshot_time": snapshot_time,
        "timezone": timezone,
        "odds_home_win": odds_home_win,
        "odds_draw": odds_draw,
        "odds_away_win": odds_away_win,
        "ou_rows": rows_from_table(ou_rows),
        "btts_yes_odds": btts_yes_odds,
        "btts_no_odds": btts_no_odds,
        "correct_score_rows": rows_from_table(correct_score_rows),
        "home_other": home_other,
        "draw_other": draw_other,
        "away_other": away_other,
        "asian_handicap_line": asian_handicap_line,
        "asian_handicap_home_odds": asian_handicap_home_odds,
        "asian_handicap_away_odds": asian_handicap_away_odds,
        "rqspf_handicap": rqspf_handicap,
        "rqspf_home_odds": rqspf_home_odds,
        "rqspf_draw_odds": rqspf_draw_odds,
        "rqspf_away_odds": rqspf_away_odds,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "home_fifa_rank": home_fifa_rank,
        "away_fifa_rank": away_fifa_rank,
        "home_rest_days": home_rest_days,
        "away_rest_days": away_rest_days,
        "home_key_players_missing": home_key_players_missing,
        "away_key_players_missing": away_key_players_missing,
        "home_lineup_strength": home_lineup_strength,
        "away_lineup_strength": away_lineup_strength,
        "weather_note": weather_note,
        "injury_notes": injury_notes,
        "lineup_notes": lineup_notes,
        "schedule_notes": schedule_notes,
        "motivation_notes": motivation_notes,
        "source_notes": source_notes,
        "dc_enabled": dc_enabled,
        "max_goals": max_goals,
        "market_weight": market_weight,
        "correct_score_weight": correct_score_weight,
        "btts_weight": btts_weight,
        "ou_weight": ou_weight,
        "x1x2_weight": x1x2_weight,
        "h2h_weight": h2h_weight,
        "totals_weight": totals_weight,
        "alternate_totals_weight": alternate_totals_weight,
        "spreads_weight": spreads_weight,
        "sporttery_1x2_weight": sporttery_1x2_weight,
        "sporttery_handicap_3way_weight": sporttery_handicap_3way_weight,
        "sporttery_total_goals_weight": sporttery_total_goals_weight,
        "sporttery_correct_score_weight": sporttery_correct_score_weight,
        "sporttery_half_full_weight": sporttery_half_full_weight,
        "team_adjustment_strength": team_adjustment_strength,
        "market_only_mode": market_only_mode,
        "internal_lambda_home": internal_lambda_home,
        "internal_lambda_away": internal_lambda_away,
        "calibration_sources": calibration_sources,
        "value_comparison_sources": value_comparison_sources,
    }
    st.session_state["ui_form_state"] = current_state

    with results_tab:
        _section_card("预测看板", "点击开始预测后，先展示结论摘要，再展示图表和高级诊断。")
        run_prediction = st.button("开始预测", type="primary", use_container_width=True)
        try:
            base_payload = build_yaml_from_form_state(current_state)
            a_source_payload = (
                st.session_state.get("applied_a_source_payload")
                or st.session_state.get("a_source_payload")
                or st.session_state.get("the_odds_a_source_payload")
            )
            b_source_payload = (
                st.session_state.get("applied_b_source_payload")
                or st.session_state.get("b_source_payload")
            )
            current_payload = merge_prediction_payload(
                base_payload,
                a_source_payload,
                b_source_payload,
            )
            if a_source_payload:
                _notice_card("当前预测将优先使用已应用的国际赔率通道 payload。", "info")
            st.download_button(
                "下载输入 YAML",
                data=dump_yaml(current_payload),
                file_name="downloaded_match.yaml",
                mime="text/yaml",
                key="results_current_input_yaml",
            )
        except Exception as exc:
            current_payload = {}
            _notice_card(f"输入 YAML 构建失败：{exc}", "danger")

        if run_prediction:
            try:
                match_input = match_input_from_dict(current_payload)
                dc_enabled_for_run = bool(match_input.settings.dc_enabled)
                result = predict(match_input, dc_enabled=dc_enabled_for_run)
                st.session_state["prediction_result"] = result
                st.session_state["prediction_payload"] = current_payload
                st.session_state["applied_a_source_payload"] = a_source_payload
                st.session_state["applied_b_source_payload"] = b_source_payload
                _notice_card("预测完成。", "success")
            except Exception as exc:
                st.session_state["prediction_result"] = None
                _notice_card(f"预测失败，请检查输入数据：{exc}", "danger")

        result = st.session_state.get("prediction_result")
        if result:
            _render_result(
                result,
                st.session_state.get("prediction_payload") or current_payload,
                current_state,
            )
        else:
            render_empty_state(
                "尚未生成预测结果",
                "请检查比赛信息、国际赔率通道或体彩赔率通道输入后点击“开始预测”。",
            )

    with audit_tab:
        _render_audit(st.session_state.get("prediction_result"), current_state)


if __name__ == "__main__":
    main()
