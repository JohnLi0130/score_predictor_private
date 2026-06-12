from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from score_predictor.poisson import score_matrix

from .dixon_coles import apply_dixon_coles_adjustment


DEFAULT_OU_LINES = (1.5, 2.5, 3.5, 4.5)


def calibrated_score_matrix(
    lambda_home: float,
    lambda_away: float,
    rho: float = 0.0,
    dc_enabled: bool = False,
    max_goals: int = 10,
) -> pd.DataFrame:
    matrix = score_matrix(lambda_home, lambda_away, max_goals=max_goals)
    if dc_enabled:
        matrix = apply_dixon_coles_adjustment(matrix, lambda_home, lambda_away, rho)
    return matrix


def summarize_score_matrix(
    score_df: pd.DataFrame,
    over_under_lines: Iterable[float] = DEFAULT_OU_LINES,
    top_n: int = 10,
) -> dict[str, Any]:
    total_goals = score_df["home_goals"] + score_df["away_goals"]
    home_win = score_df.loc[score_df["home_goals"] > score_df["away_goals"], "prob"].sum()
    draw = score_df.loc[score_df["home_goals"] == score_df["away_goals"], "prob"].sum()
    away_win = score_df.loc[score_df["home_goals"] < score_df["away_goals"], "prob"].sum()
    btts_yes = score_df.loc[
        (score_df["home_goals"] > 0) & (score_df["away_goals"] > 0),
        "prob",
    ].sum()

    over_under: dict[str, dict[str, float]] = {}
    for line in over_under_lines:
        over = score_df.loc[total_goals > float(line), "prob"].sum()
        over_under[f"{float(line):g}"] = {
            "over": float(over),
            "under": float(1.0 - over),
        }

    total_goals_distribution = {
        str(total): float(score_df.loc[total_goals == total, "prob"].sum())
        for total in range(7)
    }
    total_goals_distribution["7+"] = float(
        score_df.loc[total_goals >= 7, "prob"].sum()
    )

    top_scores = score_df.head(top_n)[["score", "prob"]].to_dict(orient="records")
    correct_scores = {
        str(row["score"]): float(row["prob"])
        for row in score_df[["score", "prob"]].to_dict(orient="records")
    }

    return {
        "one_x_two": {
            "home": float(home_win),
            "draw": float(draw),
            "away": float(away_win),
        },
        "over_under": over_under,
        "total_goals": total_goals_distribution,
        "btts": {
            "yes": float(btts_yes),
            "no": float(1.0 - btts_yes),
        },
        "correct_scores": correct_scores,
        "top_scores": top_scores,
    }


def score_matrix_records(score_df: pd.DataFrame) -> list[dict[str, float | int | str]]:
    return [
        {
            "home_goals": int(row["home_goals"]),
            "away_goals": int(row["away_goals"]),
            "score": str(row["score"]),
            "prob": float(row["prob"]),
        }
        for row in score_df.to_dict(orient="records")
    ]


def result_direction(one_x_two: dict[str, float], tolerance: float = 0.015) -> str:
    home = float(one_x_two.get("home", 0.0))
    away = float(one_x_two.get("away", 0.0))
    if abs(home - away) <= tolerance:
        return "balanced"
    return "home" if home > away else "away"
