from __future__ import annotations

import math

import pandas as pd


def poisson_pmf(k: int, lam: float) -> float:
    if k < 0:
        raise ValueError("k must be non-negative.")
    if lam <= 0:
        raise ValueError("Lambda must be positive.")
    return math.exp(-lam) * lam**k / math.factorial(k)


def score_matrix(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 7,
    normalize: bool = True,
) -> pd.DataFrame:
    if lambda_home <= 0 or lambda_away <= 0:
        raise ValueError("Lambda values must be positive.")
    if max_goals < 1:
        raise ValueError("max_goals must be at least 1.")

    rows = []
    for home_goals in range(max_goals + 1):
        home_prob = poisson_pmf(home_goals, lambda_home)
        for away_goals in range(max_goals + 1):
            away_prob = poisson_pmf(away_goals, lambda_away)
            rows.append(
                {
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "score": f"{home_goals}-{away_goals}",
                    "prob": home_prob * away_prob,
                }
            )

    df = pd.DataFrame(rows)
    if normalize:
        total = df["prob"].sum()
        if total <= 0:
            raise RuntimeError("Score matrix probability mass is not positive.")
        df["prob"] = df["prob"] / total

    return df.sort_values("prob", ascending=False).reset_index(drop=True)
