from __future__ import annotations

import html
from typing import Any, Iterable, Sequence

import pandas as pd
import streamlit as st


def escape_html(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def status_badge(label: Any, tone: str = "info") -> str:
    return f'<span class="status-badge {escape_html(tone)}">{escape_html(label)}</span>'


def render_section_card(title: str, body: str, badge: str | None = None) -> None:
    badge_html = status_badge(badge, "accent") if badge else ""
    st.markdown(
        f"""
        <div class="section-card">
            <div style="display:flex;justify-content:space-between;gap:16px;align-items:flex-start;">
                <div>
                    <p class="section-title">{escape_html(title)}</p>
                    <p class="section-copy">{escape_html(body)}</p>
                </div>
                {badge_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_card(message: str, tone: str = "warning") -> None:
    tone = {
        "low": "success",
        "medium": "warning",
        "high": "danger",
        "success": "success",
        "warning": "warning",
        "danger": "danger",
        "error": "danger",
        "info": "info",
    }.get(tone, "warning")
    st.markdown(
        f'<div class="status-card {tone}">{escape_html(message)}</div>',
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-state-title">{escape_html(title)}</div>
            <div class="empty-state-copy">{escape_html(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metadata_grid(items: Sequence[tuple[str, Any]]) -> None:
    item_html = "\n".join(
        f"""
        <div class="metadata-item">
            <div class="metadata-label">{escape_html(label)}</div>
            <div class="metadata-value">{escape_html(value if value not in (None, "") else "—")}</div>
        </div>
        """
        for label, value in items
    )
    st.markdown(f'<div class="metadata-grid">{item_html}</div>', unsafe_allow_html=True)


def render_badge_row(items: Iterable[tuple[Any, str] | Any]) -> None:
    badges = []
    for item in items:
        if isinstance(item, tuple):
            label, tone = item
        else:
            label, tone = item, "info"
        badges.append(status_badge(label, tone))
    st.markdown(
        f'<div class="badge-row">{"".join(badges)}</div>',
        unsafe_allow_html=True,
    )


def render_text_panel(text: str) -> None:
    st.markdown(
        f'<div class="bookmaker-list">{escape_html(text or "—")}</div>',
        unsafe_allow_html=True,
    )


def _is_numeric_column(column: Any) -> bool:
    name = str(column)
    numeric_hints = (
        "概率",
        "赔率",
        "奖金",
        "RMSE",
        "误差",
        "数值",
        "评分",
        "盘口线",
        "排名",
        "大",
        "小",
        "line",
        "odds",
        "prob",
        "value",
    )
    return any(hint in name for hint in numeric_hints)


def _normalize_table_frame(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, dict):
        return pd.DataFrame([data])
    return pd.DataFrame(list(data or []))


def render_styled_table(
    data: Any,
    title: str | None = None,
    *,
    max_height: int = 340,
    empty_title: str = "暂无数据",
    empty_body: str = "当前没有可展示的表格数据。",
) -> None:
    frame = _normalize_table_frame(data)
    if frame.empty:
        render_empty_state(empty_title, empty_body)
        return

    columns = list(frame.columns)
    header = "".join(
        f'<th class="{"num" if _is_numeric_column(column) else ""}">{escape_html(column)}</th>'
        for column in columns
    )
    body_rows = []
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if value is None or (
                not isinstance(value, (list, tuple, dict, set)) and pd.isna(value)
            ):
                value = "—"
            css_class = "num" if _is_numeric_column(column) else ""
            cells.append(f'<td class="{css_class}">{escape_html(value)}</td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    title_html = f'<div class="sp-table-title">{escape_html(title)}</div>' if title else ""
    st.markdown(
        f"""
        <div class="sp-table-wrap">
            {title_html}
            <div class="sp-table-scroll" style="max-height:{int(max_height)}px;">
                <table class="sp-table">
                    <thead><tr>{header}</tr></thead>
                    <tbody>{''.join(body_rows)}</tbody>
                </table>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
