from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


DEFAULT_OU_LINES = (1.5, 2.5, 3.5, 4.5)

FONT_FAMILY = (
    '"Microsoft YaHei", "Noto Sans SC", "PingFang SC", '
    '"Hiragino Sans GB", "SimHei", sans-serif'
)
COLOR_BG = "rgba(7, 11, 20, 0)"
COLOR_PANEL = "rgba(20, 29, 46, 0.88)"
COLOR_GRID = "rgba(148, 163, 184, 0.14)"
COLOR_TEXT = "#F8FAFC"
COLOR_MUTED = "#CBD5E1"
COLOR_SUBTLE = "#94A3B8"
COLOR_GOLD = "#D6A84F"
COLOR_GOLD_LIGHT = "#E4BD6A"
COLOR_SKY = "#38BDF8"
COLOR_GREEN = "#22C55E"
COLOR_RED = "#F87171"
COLOR_AMBER = "#F59E0B"
COLOR_BLUE = "#60A5FA"


def score_matrix_to_frame(score_matrix: Any) -> pd.DataFrame:
    if isinstance(score_matrix, pd.DataFrame):
        frame = score_matrix.copy()
    else:
        frame = pd.DataFrame(list(score_matrix or []))
    required = {"home_goals", "away_goals", "prob"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"score_matrix is missing columns: {sorted(missing)}")
    frame["home_goals"] = frame["home_goals"].astype(int)
    frame["away_goals"] = frame["away_goals"].astype(int)
    frame["prob"] = frame["prob"].astype(float)
    if "score" not in frame.columns:
        frame["score"] = (
            frame["home_goals"].astype(str) + "-" + frame["away_goals"].astype(str)
        )
    return frame.sort_values("prob", ascending=False).reset_index(drop=True)


def top_scores_frame(top_scores: Iterable[dict[str, Any]], top_n: int = 5) -> pd.DataFrame:
    frame = pd.DataFrame(list(top_scores or [])[:top_n])
    if frame.empty:
        return pd.DataFrame(columns=["score", "probability"])
    frame = frame.rename(columns={"prob": "probability"})
    return frame[["score", "probability"]]


def total_goals_distribution(score_matrix: Any, plus_bucket: int = 7) -> pd.DataFrame:
    frame = score_matrix_to_frame(score_matrix)
    totals = frame["home_goals"] + frame["away_goals"]
    rows = []
    for total in range(plus_bucket):
        rows.append(
            {
                "total_goals": str(total),
                "probability": float(frame.loc[totals == total, "prob"].sum()),
            }
        )
    rows.append(
        {
            "total_goals": f"{plus_bucket}+",
            "probability": float(frame.loc[totals >= plus_bucket, "prob"].sum()),
        }
    )
    return pd.DataFrame(rows)


def over_under_probabilities(
    score_matrix: Any,
    lines: Iterable[float] = DEFAULT_OU_LINES,
) -> pd.DataFrame:
    frame = score_matrix_to_frame(score_matrix)
    totals = frame["home_goals"] + frame["away_goals"]
    rows = []
    for line in lines:
        over = float(frame.loc[totals > float(line), "prob"].sum())
        rows.append({"line": f"{float(line):g}", "over": over, "under": 1.0 - over})
    return pd.DataFrame(rows)


def btts_probabilities(score_matrix: Any) -> dict[str, float]:
    frame = score_matrix_to_frame(score_matrix)
    yes = float(
        frame.loc[
            (frame["home_goals"] > 0) & (frame["away_goals"] > 0),
            "prob",
        ].sum()
    )
    return {"yes": yes, "no": 1.0 - yes}


def _apply_dark_theme(fig: Any, title: str, height: int) -> Any:
    fig.update_layout(
        title={
            "text": title,
            "font": {"size": 20, "color": COLOR_TEXT, "family": FONT_FAMILY},
            "x": 0.02,
            "xanchor": "left",
        },
        height=height,
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font={"family": FONT_FAMILY, "size": 13, "color": COLOR_MUTED},
        margin={"l": 56, "r": 34, "t": 66, "b": 48},
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": COLOR_MUTED, "size": 13, "family": FONT_FAMILY},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        hoverlabel={
            "bgcolor": "#141D2E",
            "bordercolor": "rgba(148, 163, 184, 0.34)",
            "font": {"color": COLOR_TEXT, "family": FONT_FAMILY, "size": 13},
        },
        separators=".,",
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=COLOR_GRID,
        zeroline=False,
        linecolor=COLOR_GRID,
        tickfont={"size": 12, "color": COLOR_SUBTLE, "family": FONT_FAMILY},
        title_font={"size": 13, "color": COLOR_MUTED, "family": FONT_FAMILY},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLOR_GRID,
        zeroline=False,
        linecolor=COLOR_GRID,
        tickfont={"size": 12, "color": COLOR_SUBTLE, "family": FONT_FAMILY},
        title_font={"size": 13, "color": COLOR_MUTED, "family": FONT_FAMILY},
    )
    return fig


def _percent_text(series: pd.Series) -> pd.Series:
    return series.map(lambda value: f"{float(value):.1%}")


def plot_score_heatmap(score_matrix: Any, home_team: str, away_team: str):
    import plotly.graph_objects as go

    frame = score_matrix_to_frame(score_matrix)
    pivot = frame.pivot_table(
        index="home_goals",
        columns="away_goals",
        values="prob",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    colorscale = [
        [0.0, "#0F172A"],
        [0.32, "#1E3A5F"],
        [0.58, COLOR_BLUE],
        [0.78, COLOR_SKY],
        [1.0, COLOR_GOLD_LIGHT],
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(value) for value in pivot.columns],
            y=[str(value) for value in pivot.index],
            colorscale=colorscale,
            colorbar={
                "title": {"text": "概率", "font": {"color": COLOR_MUTED}},
                "tickformat": ".0%",
                "tickfont": {"color": COLOR_MUTED},
                "outlinewidth": 0,
            },
            hovertemplate=(
                f"{home_team} %{{y}} - %{{x}} {away_team}<br>"
                "概率：%{z:.2%}<extra></extra>"
            ),
        )
    )
    fig.update_traces(xgap=2, ygap=2)
    _apply_dark_theme(fig, "比分概率热力图", 540)
    fig.update_layout(margin={"l": 70, "r": 64, "t": 72, "b": 58})
    fig.update_xaxes(title=f"{away_team} 进球数", showgrid=False)
    fig.update_yaxes(title=f"{home_team} 进球数", showgrid=False, autorange="reversed")
    return fig


def plot_top_scores(top_scores: Iterable[dict[str, Any]]):
    import plotly.graph_objects as go

    frame = top_scores_frame(top_scores, top_n=5).sort_values("probability")
    colors = ["#182235", "#1E3A5F", COLOR_BLUE, COLOR_SKY, COLOR_GOLD_LIGHT]
    fig = go.Figure(
        data=[
            go.Bar(
                x=frame["probability"],
                y=frame["score"],
                orientation="h",
                text=_percent_text(frame["probability"]),
                textposition="outside",
                marker={
                    "color": colors[-len(frame) :] if len(frame) else [],
                    "line": {"color": "rgba(255,255,255,0.18)", "width": 1},
                },
                hovertemplate="比分 %{y}<br>概率：%{x:.2%}<extra></extra>",
            )
        ]
    )
    _apply_dark_theme(fig, "最可能比分 Top 5", 330)
    fig.update_xaxes(title="概率", tickformat=".0%", range=[0, max(0.01, frame["probability"].max() * 1.24) if not frame.empty else 1])
    fig.update_yaxes(title="比分")
    return fig


def plot_1x2_pie(prob_home: float, prob_draw: float, prob_away: float):
    import plotly.graph_objects as go

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["主胜", "平局", "客胜"],
                values=[prob_home, prob_draw, prob_away],
                hole=0.58,
                textinfo="label+percent",
                textfont={"size": 14, "color": COLOR_TEXT, "family": FONT_FAMILY},
                marker={
                    "colors": [COLOR_GOLD_LIGHT, COLOR_SKY, COLOR_RED],
                    "line": {"color": "rgba(255,255,255,0.12)", "width": 1},
                },
                hovertemplate="%{label}<br>概率：%{percent}<extra></extra>",
            )
        ]
    )
    _apply_dark_theme(fig, "胜平负概率", 340)
    fig.update_layout(
        showlegend=True,
        margin={"l": 24, "r": 24, "t": 72, "b": 24},
        annotations=[
            {
                "text": "1X2",
                "showarrow": False,
                "font": {"size": 20, "color": COLOR_GOLD_LIGHT, "family": FONT_FAMILY},
            }
        ],
    )
    return fig


def plot_total_goals_distribution(score_matrix: Any):
    import plotly.graph_objects as go

    frame = total_goals_distribution(score_matrix)
    fig = go.Figure(
        data=[
            go.Bar(
                x=frame["total_goals"],
                y=frame["probability"],
                text=_percent_text(frame["probability"]),
                textposition="outside",
                marker={
                    "color": frame["probability"],
                    "colorscale": [
                        [0.0, "#1E3A5F"],
                        [0.45, COLOR_SKY],
                        [1.0, COLOR_GOLD_LIGHT],
                    ],
                    "line": {"color": "rgba(255,255,255,0.14)", "width": 1},
                },
                hovertemplate="总进球 %{x}<br>概率：%{y:.2%}<extra></extra>",
            )
        ]
    )
    _apply_dark_theme(fig, "总进球分布", 360)
    fig.update_xaxes(title="总进球数")
    fig.update_yaxes(title="概率", tickformat=".0%", range=[0, max(0.01, frame["probability"].max() * 1.22)])
    return fig


def plot_over_under_table(probabilities: dict[str, dict[str, float]] | pd.DataFrame):
    import plotly.express as px

    if isinstance(probabilities, pd.DataFrame):
        frame = probabilities.copy()
    else:
        frame = pd.DataFrame(
            [
                {"line": line, "outcome": "大", "probability": values["over"]}
                for line, values in (probabilities or {}).items()
            ]
            + [
                {"line": line, "outcome": "小", "probability": values["under"]}
                for line, values in (probabilities or {}).items()
            ]
        )
    frame["outcome"] = frame["outcome"].replace({"Over": "大", "Under": "小"})
    fig = px.bar(
        frame,
        x="probability",
        y="line",
        color="outcome",
        orientation="h",
        barmode="group",
        text=_percent_text(frame["probability"]),
        color_discrete_map={"大": COLOR_GOLD, "小": COLOR_SKY},
    )
    fig.update_traces(
        marker_line={"color": "rgba(255,255,255,0.12)", "width": 1},
        hovertemplate="盘口线 %{y}<br>%{legendgroup}：%{x:.2%}<extra></extra>",
    )
    _apply_dark_theme(fig, "大小球概率", 370)
    fig.update_xaxes(title="概率", tickformat=".0%")
    fig.update_yaxes(title="盘口线")
    return fig


def plot_btts_bar(prob_yes: float, prob_no: float):
    import plotly.graph_objects as go

    frame = pd.DataFrame(
        [
            {"outcome": "双方进球：是", "probability": prob_yes, "color": COLOR_GREEN},
            {"outcome": "双方进球：否", "probability": prob_no, "color": COLOR_AMBER},
        ]
    )
    fig = go.Figure(
        data=[
            go.Bar(
                x=frame["probability"],
                y=frame["outcome"],
                orientation="h",
                text=_percent_text(frame["probability"]),
                textposition="outside",
                marker={
                    "color": frame["color"],
                    "line": {"color": "rgba(255,255,255,0.12)", "width": 1},
                },
                hovertemplate="%{y}<br>概率：%{x:.2%}<extra></extra>",
            )
        ]
    )
    _apply_dark_theme(fig, "双方进球概率", 280)
    fig.update_xaxes(title="概率", tickformat=".0%", range=[0, max(prob_yes, prob_no, 0.01) * 1.2])
    fig.update_yaxes(title="")
    return fig


def plot_market_fit_errors(fit_errors: pd.DataFrame):
    import plotly.graph_objects as go

    frame = fit_errors.copy()
    if frame.empty:
        frame = pd.DataFrame(columns=["盘口类型", "RMSE 误差"])
    market_col = "盘口类型" if "盘口类型" in frame.columns else "market"
    rmse_col = "RMSE 误差" if "RMSE 误差" in frame.columns else "rmse"
    line_col = "盘口线" if "盘口线" in frame.columns else "line"
    frame["展示名称"] = frame.apply(
        lambda row: (
            f"{row.get(market_col, '')} {row.get(line_col, '')}".strip()
            if line_col in frame.columns
            else str(row.get(market_col, ""))
        ),
        axis=1,
    )
    fig = go.Figure(
        data=[
            go.Bar(
                x=frame[rmse_col] if rmse_col in frame.columns else [],
                y=frame["展示名称"],
                orientation="h",
                text=(frame[rmse_col].map(lambda value: f"{float(value):.3f}") if rmse_col in frame.columns else []),
                textposition="outside",
                marker={
                    "color": COLOR_AMBER,
                    "line": {"color": "rgba(255,255,255,0.14)", "width": 1},
                },
                hovertemplate="%{y}<br>RMSE：%{x:.4f}<extra></extra>",
            )
        ]
    )
    _apply_dark_theme(fig, "盘口拟合误差", 320)
    max_error = float(frame[rmse_col].max()) if rmse_col in frame.columns and not frame.empty else 0.1
    fig.update_xaxes(title="RMSE 误差", range=[0, max(0.1, max_error * 1.25)])
    fig.update_yaxes(title="")
    return fig
