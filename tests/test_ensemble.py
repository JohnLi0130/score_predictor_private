from __future__ import annotations

import pytest

from score_predictor.ensemble import blend_lambdas


def test_log_space_blending_weight_boundaries() -> None:
    assert blend_lambdas(2.0, 1.0, 1.0, 2.0, market_weight=1.0) == pytest.approx(
        (2.0, 1.0)
    )
    assert blend_lambdas(2.0, 1.0, 1.0, 2.0, market_weight=0.0) == pytest.approx(
        (1.0, 2.0)
    )


def test_log_space_blending_rejects_invalid_weight() -> None:
    with pytest.raises(ValueError):
        blend_lambdas(2.0, 1.0, 1.0, 2.0, market_weight=1.1)
