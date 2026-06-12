from __future__ import annotations

import pytest

from score_predictor.poisson import score_matrix


def test_score_matrix_sums_to_one() -> None:
    df = score_matrix(1.54, 0.76, max_goals=7)
    assert df["prob"].sum() == pytest.approx(1.0)
    assert {"home_goals", "away_goals", "score", "prob"}.issubset(df.columns)
