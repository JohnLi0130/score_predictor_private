from __future__ import annotations

import pandas as pd

from .v3.dixon_coles import apply_dixon_coles_adjustment as _apply_v3_dc


def apply_dixon_coles_adjustment(
    score_df: pd.DataFrame,
    lambda_home: float,
    lambda_away: float,
    rho: float = 0.0,
) -> pd.DataFrame:
    """Apply the V3 Dixon-Coles low-score correction and renormalize."""
    return _apply_v3_dc(score_df, lambda_home, lambda_away, rho)
