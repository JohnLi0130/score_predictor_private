from __future__ import annotations

from typing import Final

import streamlit as st


FONT_FAMILY: Final[str] = (
    '"Microsoft YaHei", "Noto Sans SC", "PingFang SC", '
    '"Hiragino Sans GB", "SimHei", sans-serif'
)

TOKENS: Final[dict[str, str]] = {
    "page_bg": "#070B14",
    "page_bg_2": "#0B1220",
    "card_bg": "#111827",
    "card_bg_high": "#141D2E",
    "card_bg_hover": "#182235",
    "input_bg": "#0F172A",
    "border_weak": "rgba(148, 163, 184, 0.18)",
    "border_strong": "rgba(148, 163, 184, 0.34)",
    "text": "#F8FAFC",
    "text_secondary": "#CBD5E1",
    "text_muted": "#94A3B8",
    "placeholder": "#7C879A",
    "accent": "#D6A84F",
    "accent_hover": "#E4BD6A",
    "cyan": "#38BDF8",
    "info": "#60A5FA",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "risk": "#EF4444",
    "danger_bg": "rgba(239, 68, 68, 0.12)",
    "warning_bg": "rgba(245, 158, 11, 0.12)",
    "success_bg": "rgba(34, 197, 94, 0.12)",
}


def inject_professional_theme_css() -> None:
    """Apply the product-grade dark Streamlit theme."""
    st.markdown(
        f"""
        <style>
        :root {{
            --sp-page-bg: {TOKENS["page_bg"]};
            --sp-page-bg-2: {TOKENS["page_bg_2"]};
            --sp-card-bg: {TOKENS["card_bg"]};
            --sp-card-bg-high: {TOKENS["card_bg_high"]};
            --sp-card-bg-hover: {TOKENS["card_bg_hover"]};
            --sp-input-bg: {TOKENS["input_bg"]};
            --sp-border-weak: {TOKENS["border_weak"]};
            --sp-border-strong: {TOKENS["border_strong"]};
            --sp-text: {TOKENS["text"]};
            --sp-text-secondary: {TOKENS["text_secondary"]};
            --sp-text-muted: {TOKENS["text_muted"]};
            --sp-placeholder: {TOKENS["placeholder"]};
            --sp-accent: {TOKENS["accent"]};
            --sp-accent-hover: {TOKENS["accent_hover"]};
            --sp-cyan: {TOKENS["cyan"]};
            --sp-info: {TOKENS["info"]};
            --sp-success: {TOKENS["success"]};
            --sp-warning: {TOKENS["warning"]};
            --sp-risk: {TOKENS["risk"]};
            --sp-danger-bg: {TOKENS["danger_bg"]};
            --sp-warning-bg: {TOKENS["warning_bg"]};
            --sp-success-bg: {TOKENS["success_bg"]};
            --sp-font-cn: {FONT_FAMILY};
        }}

        html, body, [class*="css"], .stApp {{
            font-family: var(--sp-font-cn);
            letter-spacing: 0;
        }}

        .stApp {{
            color: var(--sp-text);
            background:
                linear-gradient(180deg, var(--sp-page-bg) 0%, var(--sp-page-bg-2) 46%, #080D18 100%);
        }}

        [data-testid="stAppViewContainer"] > .main {{
            background: transparent;
        }}

        .block-container {{
            max-width: 1560px;
            padding: 1.55rem 1.5rem 4rem;
        }}

        #MainMenu, footer, header, [data-testid="stDecoration"] {{
            visibility: hidden;
            height: 0;
        }}

        h1, h2, h3, h4, p, label, span, div {{
            letter-spacing: 0 !important;
        }}

        h1 {{
            color: var(--sp-text) !important;
            font-size: 38px !important;
            line-height: 1.16 !important;
            font-weight: 820 !important;
        }}

        h2 {{
            color: var(--sp-text) !important;
            font-size: 22px !important;
            line-height: 1.35 !important;
            font-weight: 760 !important;
        }}

        h3 {{
            color: var(--sp-text) !important;
            font-size: 18px !important;
            line-height: 1.35 !important;
            font-weight: 720 !important;
        }}

        div[data-testid="stMarkdownContainer"] p {{
            color: var(--sp-text-secondary);
            font-size: 15px;
            line-height: 1.72;
        }}

        div[data-testid="stMarkdownContainer"] h4 {{
            color: var(--sp-text) !important;
            font-size: 16px !important;
            margin: 0.45rem 0 0.6rem;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            padding: 6px 8px 0;
            border-bottom: 1px solid var(--sp-border-weak);
            background: rgba(17, 24, 39, 0.44);
            border-radius: 8px 8px 0 0;
            overflow-x: auto;
        }}

        .stTabs [data-baseweb="tab"] {{
            min-height: 44px;
            height: 44px;
            border-radius: 7px 7px 0 0;
            padding: 0 16px;
            color: var(--sp-text-secondary);
            font-size: 15px;
            font-weight: 680;
            background: transparent;
            border-bottom: 2px solid transparent;
        }}

        .stTabs [data-baseweb="tab"]:hover {{
            color: var(--sp-text);
            background: rgba(56, 189, 248, 0.08);
        }}

        .stTabs [aria-selected="true"] {{
            color: var(--sp-text) !important;
            background: rgba(214, 168, 79, 0.10) !important;
            border-bottom: 2px solid var(--sp-accent) !important;
            box-shadow: inset 0 -1px 0 rgba(214, 168, 79, 0.55);
        }}

        .stButton > button,
        .stDownloadButton > button,
        button[data-testid="baseButton-secondary"],
        button[data-testid="baseButton-tertiary"] {{
            min-height: 44px;
            border-radius: 8px;
            padding: 0 17px;
            border: 1px solid var(--sp-border-strong);
            background: rgba(20, 29, 46, 0.92);
            color: var(--sp-text) !important;
            font-size: 15px;
            font-weight: 720;
            box-shadow: none;
            transition: border-color 0.16s ease, background 0.16s ease, color 0.16s ease, box-shadow 0.16s ease, transform 0.16s ease;
        }}

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        button[data-testid="baseButton-secondary"]:hover,
        button[data-testid="baseButton-tertiary"]:hover {{
            border-color: rgba(56, 189, 248, 0.58);
            background: rgba(24, 34, 53, 0.98);
            color: var(--sp-text) !important;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.24);
            transform: translateY(-1px);
        }}

        .stButton > button:focus,
        .stDownloadButton > button:focus {{
            outline: 2px solid rgba(56, 189, 248, 0.22);
            outline-offset: 2px;
        }}

        .stButton > button:active,
        .stDownloadButton > button:active {{
            transform: translateY(0);
        }}

        .stButton > button[kind="primary"],
        button[data-testid="baseButton-primary"] {{
            min-height: 48px;
            border: 1px solid rgba(228, 189, 106, 0.74);
            background: linear-gradient(135deg, #C99438 0%, var(--sp-accent-hover) 100%);
            color: #08111F !important;
            font-size: 16px;
            font-weight: 820;
            box-shadow: 0 16px 34px rgba(214, 168, 79, 0.22);
        }}

        .stButton > button[kind="primary"]:hover,
        button[data-testid="baseButton-primary"]:hover {{
            background: linear-gradient(135deg, #D6A84F 0%, #F1CB7A 100%);
            border-color: rgba(244, 211, 145, 0.92);
            color: #070B14 !important;
            box-shadow: 0 18px 38px rgba(214, 168, 79, 0.28);
        }}

        .stButton > button:disabled,
        .stDownloadButton > button:disabled {{
            color: rgba(203, 213, 225, 0.44) !important;
            background: rgba(15, 23, 42, 0.62) !important;
            border-color: rgba(148, 163, 184, 0.14) !important;
            box-shadow: none !important;
        }}

        label, div[data-testid="stWidgetLabel"] label p {{
            color: var(--sp-text-secondary) !important;
            font-size: 14px !important;
            font-weight: 690 !important;
        }}

        div[data-baseweb="input"],
        div[data-baseweb="textarea"],
        div[data-baseweb="select"],
        div[data-testid="stNumberInput"] > div {{
            border-radius: 8px !important;
            border: 1px solid var(--sp-border-weak) !important;
            background: var(--sp-input-bg) !important;
            min-height: 42px;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.025);
        }}

        div[data-baseweb="input"]:focus-within,
        div[data-baseweb="textarea"]:focus-within,
        div[data-baseweb="select"]:focus-within,
        div[data-testid="stNumberInput"] > div:focus-within {{
            border-color: rgba(56, 189, 248, 0.70) !important;
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.035) !important;
        }}

        input, textarea,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div {{
            color: var(--sp-text) !important;
            font-size: 15px !important;
        }}

        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextArea"] textarea {{
            color: var(--sp-text) !important;
            background: var(--sp-input-bg) !important;
            -webkit-text-fill-color: var(--sp-text) !important;
        }}

        .stSelectbox [data-baseweb="select"],
        [data-testid="stSelectbox"] [data-baseweb="select"],
        [data-baseweb="select"] > div {{
            color: var(--sp-text) !important;
            background: var(--sp-input-bg) !important;
        }}

        [data-baseweb="select"] input,
        [data-baseweb="select"] input[readonly],
        [data-baseweb="select"] [role="combobox"] {{
            color: var(--sp-text) !important;
            background: transparent !important;
            -webkit-text-fill-color: var(--sp-text) !important;
        }}

        input::placeholder,
        textarea::placeholder {{
            color: var(--sp-placeholder) !important;
            opacity: 1 !important;
        }}

        textarea {{
            min-height: 108px;
        }}

        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        ul[role="listbox"] {{
            background: var(--sp-card-bg-high) !important;
            border: 1px solid var(--sp-border-strong) !important;
            border-radius: 8px !important;
            color: var(--sp-text) !important;
        }}

        li[role="option"] {{
            color: var(--sp-text-secondary) !important;
            background: transparent !important;
        }}

        li[role="option"]:hover,
        li[aria-selected="true"] {{
            color: var(--sp-text) !important;
            background: rgba(56, 189, 248, 0.12) !important;
        }}

        div[data-testid="stFileUploader"] section {{
            background: rgba(15, 23, 42, 0.86);
            border: 1px dashed rgba(56, 189, 248, 0.42);
            border-radius: 8px;
        }}

        div[data-testid="stFileUploader"] section * {{
            color: var(--sp-text-secondary) !important;
        }}

        div[data-testid="stExpander"] {{
            border: 1px solid var(--sp-border-weak);
            border-radius: 8px;
            background: rgba(17, 24, 39, 0.58);
            overflow: hidden;
        }}

        div[data-testid="stExpander"] details > summary {{
            color: var(--sp-text) !important;
            font-weight: 720;
        }}

        div[data-testid="stDataFrame"],
        div[data-testid="stDataEditor"] {{
            border: 1px solid var(--sp-border-weak);
            border-radius: 8px;
            overflow: hidden;
            background: rgba(15, 23, 42, 0.84);
        }}

        div[data-testid="stDataFrame"] *,
        div[data-testid="stDataEditor"] * {{
            color: var(--sp-text-secondary);
        }}

        div[data-testid="stDataFrame"] [role="grid"],
        div[data-testid="stDataEditor"] [role="grid"],
        div[data-testid="stDataFrame"] canvas,
        div[data-testid="stDataEditor"] canvas {{
            background: var(--sp-input-bg) !important;
        }}

        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stTextArea"] textarea::placeholder,
        input::placeholder,
        textarea::placeholder {{
            color: #A8B3C7 !important;
            -webkit-text-fill-color: #A8B3C7 !important;
            opacity: 1 !important;
        }}

        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea,
        [data-testid="stNumberInput"] input {{
            color: var(--sp-text) !important;
            background-color: var(--sp-input-bg) !important;
            text-shadow: none !important;
        }}

        button[data-testid="baseButton-primary"] p,
        button[data-testid="baseButton-primary"] span,
        .stButton > button[kind="primary"] p,
        .stButton > button[kind="primary"] span {{
            color: #08111F !important;
            -webkit-text-fill-color: #08111F !important;
        }}

        div[data-testid="stMetric"] {{
            border: 1px solid var(--sp-border-weak);
            border-radius: 8px;
            padding: 13px 14px;
            background: rgba(15, 23, 42, 0.72);
        }}

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] label p {{
            color: var(--sp-text-muted) !important;
        }}

        div[data-testid="stMetricValue"] {{
            color: var(--sp-text) !important;
            font-size: 20px !important;
            overflow-wrap: anywhere;
        }}

        div[data-testid="stAlert"] {{
            border-radius: 8px;
            border: 1px solid var(--sp-border-weak);
            background: rgba(15, 23, 42, 0.82);
            color: var(--sp-text-secondary);
        }}

        div[data-testid="stAlert"] * {{
            color: var(--sp-text-secondary) !important;
        }}

        .hero-card {{
            position: relative;
            overflow: hidden;
            border: 1px solid var(--sp-border-weak);
            border-radius: 8px;
            padding: 28px 30px;
            margin-bottom: 18px;
            background:
                linear-gradient(135deg, rgba(17, 24, 39, 0.98), rgba(20, 29, 46, 0.92)),
                linear-gradient(90deg, rgba(56, 189, 248, 0.08), rgba(214, 168, 79, 0.07));
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.30);
        }}

        .hero-card::before {{
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
        }}

        .hero-content {{
            position: relative;
            display: grid;
            grid-template-columns: minmax(0, 1.38fr) minmax(360px, 0.82fr);
            gap: 28px;
            align-items: start;
        }}

        .hero-title {{
            margin: 0 0 9px;
            color: var(--sp-text);
            font-size: 38px;
            line-height: 1.15;
            font-weight: 850;
        }}

        .hero-subtitle {{
            margin: 0 0 23px;
            color: var(--sp-text-secondary);
            font-size: 16px;
            line-height: 1.72;
        }}

        .match-title {{
            margin-top: 4px;
            color: var(--sp-text);
            font-size: 30px;
            line-height: 1.22;
            font-weight: 780;
            overflow-wrap: anywhere;
        }}

        .hero-meta {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            margin-top: 18px;
        }}

        .meta-item {{
            padding: 12px 14px;
            border-radius: 8px;
            border: 1px solid var(--sp-border-weak);
            background: rgba(15, 23, 42, 0.54);
        }}

        .meta-label {{
            color: var(--sp-text-muted);
            font-size: 12px;
            margin-bottom: 5px;
        }}

        .meta-value {{
            color: var(--sp-text);
            font-size: 15px;
            font-weight: 720;
            overflow-wrap: anywhere;
        }}

        .badge-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 13px;
        }}

        .badge,
        .status-badge {{
            display: inline-flex;
            align-items: center;
            min-height: 28px;
            padding: 5px 10px;
            border-radius: 999px;
            border: 1px solid rgba(96, 165, 250, 0.34);
            color: #BFDBFE;
            background: rgba(96, 165, 250, 0.10);
            font-size: 13px;
            font-weight: 720;
            white-space: nowrap;
        }}

        .status-badge.success {{
            color: #BBF7D0;
            border-color: rgba(34, 197, 94, 0.34);
            background: var(--sp-success-bg);
        }}

        .status-badge.warning {{
            color: #FDE68A;
            border-color: rgba(245, 158, 11, 0.38);
            background: var(--sp-warning-bg);
        }}

        .status-badge.danger {{
            color: #FECACA;
            border-color: rgba(239, 68, 68, 0.38);
            background: var(--sp-danger-bg);
        }}

        .status-badge.accent {{
            color: #F8E2AF;
            border-color: rgba(214, 168, 79, 0.45);
            background: rgba(214, 168, 79, 0.12);
        }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
        }}

        .kpi-card {{
            border: 1px solid var(--sp-border-weak);
            border-radius: 8px;
            padding: 16px 16px 15px;
            background: rgba(15, 23, 42, 0.60);
            min-height: 118px;
        }}

        .kpi-label {{
            color: var(--sp-text-secondary);
            font-size: 14px;
            margin-bottom: 10px;
        }}

        .kpi-value {{
            color: var(--sp-accent-hover);
            font-size: 32px;
            line-height: 1.06;
            font-weight: 850;
            overflow-wrap: anywhere;
        }}

        .kpi-caption {{
            color: var(--sp-text-muted);
            font-size: 13px;
            margin-top: 10px;
            line-height: 1.45;
        }}

        .section-card {{
            border: 1px solid var(--sp-border-weak);
            border-radius: 8px;
            padding: 18px 20px;
            margin: 18px 0 16px;
            background: rgba(17, 24, 39, 0.84);
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.20);
        }}

        .section-card.compact {{
            padding: 14px 16px;
            margin: 12px 0;
        }}

        .section-title {{
            margin: 0;
            color: var(--sp-text);
            font-size: 20px;
            font-weight: 780;
        }}

        .section-copy {{
            margin: 7px 0 0;
            color: var(--sp-text-secondary);
            font-size: 15px;
            line-height: 1.70;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 12px;
            margin-top: 14px;
        }}

        .summary-item {{
            padding: 14px;
            border-left: 3px solid var(--sp-accent);
            background: rgba(15, 23, 42, 0.54);
            border-radius: 7px;
        }}

        .summary-label {{
            color: var(--sp-text-muted);
            font-size: 13px;
            margin-bottom: 7px;
        }}

        .summary-value {{
            color: var(--sp-text);
            font-size: 19px;
            font-weight: 780;
            line-height: 1.35;
            overflow-wrap: anywhere;
        }}

        .status-card {{
            border-radius: 8px;
            padding: 15px 16px;
            margin: 12px 0;
            font-size: 15px;
            line-height: 1.65;
            border: 1px solid var(--sp-border-weak);
            background: rgba(15, 23, 42, 0.82);
            color: var(--sp-text-secondary);
        }}

        .success-card,
        .status-card.success {{
            border-color: rgba(34, 197, 94, 0.34);
            background: var(--sp-success-bg);
            color: #BBF7D0;
        }}

        .warning-card,
        .status-card.warning {{
            border-color: rgba(245, 158, 11, 0.38);
            background: var(--sp-warning-bg);
            color: #FDE68A;
        }}

        .danger-card,
        .status-card.danger {{
            border-color: rgba(239, 68, 68, 0.40);
            background: var(--sp-danger-bg);
            color: #FECACA;
        }}

        .info-card,
        .status-card.info {{
            border-color: rgba(96, 165, 250, 0.36);
            background: rgba(96, 165, 250, 0.10);
            color: #BFDBFE;
        }}

        .empty-state {{
            border: 1px dashed var(--sp-border-strong);
            border-radius: 8px;
            padding: 28px 22px;
            margin: 16px 0;
            background: rgba(15, 23, 42, 0.62);
        }}

        .empty-state-title {{
            color: var(--sp-text);
            font-size: 18px;
            font-weight: 760;
            margin-bottom: 7px;
        }}

        .empty-state-copy {{
            color: var(--sp-text-secondary);
            font-size: 15px;
            line-height: 1.72;
        }}

        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 12px;
            margin: 14px 0 12px;
        }}

        .metadata-item {{
            border: 1px solid var(--sp-border-weak);
            border-radius: 8px;
            padding: 12px 14px;
            background: rgba(15, 23, 42, 0.66);
            min-height: 76px;
        }}

        .metadata-label {{
            color: var(--sp-text-muted);
            font-size: 12px;
            margin-bottom: 5px;
        }}

        .metadata-value {{
            color: var(--sp-text);
            font-size: 15px;
            font-weight: 720;
            line-height: 1.45;
            overflow-wrap: anywhere;
        }}

        .bookmaker-list {{
            border: 1px solid var(--sp-border-weak);
            border-radius: 8px;
            background: rgba(15, 23, 42, 0.58);
            color: var(--sp-text-secondary);
            padding: 12px 14px;
            line-height: 1.7;
            overflow-wrap: anywhere;
        }}

        .sp-table-wrap {{
            border: 1px solid var(--sp-border-weak);
            border-radius: 8px;
            overflow: hidden;
            background: rgba(15, 23, 42, 0.76);
            margin: 10px 0 16px;
        }}

        .sp-table-title {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            min-height: 42px;
            padding: 0 14px;
            color: var(--sp-text);
            font-size: 15px;
            font-weight: 760;
            border-bottom: 1px solid var(--sp-border-weak);
            background: rgba(20, 29, 46, 0.94);
        }}

        .sp-table-scroll {{
            overflow: auto;
            scrollbar-width: thin;
            scrollbar-color: rgba(148, 163, 184, 0.36) transparent;
        }}

        table.sp-table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            color: var(--sp-text-secondary);
            font-size: 14px;
        }}

        table.sp-table thead th {{
            position: sticky;
            top: 0;
            z-index: 1;
            background: #141D2E;
            color: var(--sp-text);
            font-size: 13px;
            font-weight: 760;
            text-align: left;
            padding: 11px 12px;
            border-bottom: 1px solid var(--sp-border-strong);
            white-space: nowrap;
        }}

        table.sp-table tbody td {{
            padding: 11px 12px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.10);
            color: var(--sp-text-secondary);
            line-height: 1.45;
            vertical-align: middle;
            overflow-wrap: anywhere;
        }}

        table.sp-table tbody tr:nth-child(even) td {{
            background: rgba(20, 29, 46, 0.32);
        }}

        table.sp-table tbody tr:hover td {{
            background: rgba(56, 189, 248, 0.075);
        }}

        table.sp-table td.num,
        table.sp-table th.num {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}

        .small-note {{
            color: var(--sp-text-muted);
            font-size: 13px;
            line-height: 1.7;
        }}

        @media (max-width: 980px) {{
            .block-container {{
                padding-left: 1rem;
                padding-right: 1rem;
            }}

            .hero-content,
            .summary-grid {{
                grid-template-columns: 1fr;
            }}

            .kpi-grid,
            .hero-meta {{
                grid-template-columns: 1fr;
            }}

            .hero-title {{
                font-size: 32px;
            }}

            .match-title {{
                font-size: 25px;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
