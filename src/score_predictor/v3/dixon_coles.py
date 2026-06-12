from __future__ import annotations

import pandas as pd


def dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_home * lambda_away * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_home * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lambda_away * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def apply_dixon_coles_adjustment(
    score_df: pd.DataFrame,
    lambda_home: float,
    lambda_away: float,
    rho: float = 0.0,
) -> pd.DataFrame:
    if lambda_home <= 0 or lambda_away <= 0:
        raise ValueError("Lambda values must be positive.")
    if rho < -0.30 or rho > 0.30:
        raise ValueError("rho must be between -0.30 and 0.30.")

    adjusted = score_df.copy()
    if rho == 0.0:
        return adjusted

    low_score_mask = (
        (adjusted["home_goals"] <= 1)
        & (adjusted["away_goals"] <= 1)
    )
    for index, row in adjusted.loc[low_score_mask].iterrows():
        tau = dixon_coles_tau(
            int(row["home_goals"]),
            int(row["away_goals"]),
            lambda_home,
            lambda_away,
            rho,
        )
        adjusted.at[index, "prob"] = max(0.0, float(row["prob"]) * tau)

    total = float(adjusted["prob"].sum())
    if total <= 0:
        raise RuntimeError("Dixon-Coles adjusted matrix probability mass is not positive.")
    adjusted["prob"] = adjusted["prob"] / total
    return adjusted.sort_values("prob", ascending=False).reset_index(drop=True)
