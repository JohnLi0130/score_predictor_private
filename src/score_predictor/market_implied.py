from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.optimize import minimize

from .poisson import score_matrix


def probs_from_lambdas(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 10,
    over_under_line: float = 2.5,
) -> dict[str, float]:
    df = score_matrix(lambda_home, lambda_away, max_goals=max_goals)
    total_goals = df["home_goals"] + df["away_goals"]

    home_win = df.loc[df["home_goals"] > df["away_goals"], "prob"].sum()
    draw = df.loc[df["home_goals"] == df["away_goals"], "prob"].sum()
    away_win = df.loc[df["home_goals"] < df["away_goals"], "prob"].sum()
    over = df.loc[total_goals > over_under_line, "prob"].sum()
    btts_yes = df.loc[(df["home_goals"] > 0) & (df["away_goals"] > 0), "prob"].sum()

    return {
        "home": float(home_win),
        "draw": float(draw),
        "away": float(away_win),
        "over": float(over),
        "under": float(1.0 - over),
        "btts_yes": float(btts_yes),
        "btts_no": float(1.0 - btts_yes),
    }


def infer_lambdas_from_market(
    fair_1x2: Mapping[str, float],
    over_probability: float | None = None,
    over_under_line: float = 2.5,
    initial: tuple[float, float] = (1.4, 1.1),
    max_goals: int = 10,
) -> tuple[float, float]:
    required = {"home", "draw", "away"}
    missing = required.difference(fair_1x2)
    if missing:
        raise ValueError(f"fair_1x2 is missing keys: {sorted(missing)}")

    def objective(x: np.ndarray) -> float:
        lambda_home, lambda_away = float(x[0]), float(x[1])
        pred = probs_from_lambdas(
            lambda_home,
            lambda_away,
            max_goals=max_goals,
            over_under_line=over_under_line,
        )
        loss = 0.0
        loss += (pred["home"] - fair_1x2["home"]) ** 2
        loss += (pred["draw"] - fair_1x2["draw"]) ** 2
        loss += (pred["away"] - fair_1x2["away"]) ** 2

        if over_probability is not None:
            loss += 0.7 * (pred["over"] - over_probability) ** 2

        return float(loss)

    result = minimize(
        objective,
        x0=np.array(initial, dtype=float),
        method="L-BFGS-B",
        bounds=((0.05, 5.5), (0.05, 5.5)),
    )
    if not result.success:
        raise RuntimeError(f"Market lambda inference failed: {result.message}")

    lambda_home, lambda_away = result.x
    return float(lambda_home), float(lambda_away)
