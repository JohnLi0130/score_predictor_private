from __future__ import annotations

from score_predictor.market_implied import infer_lambdas_from_market, probs_from_lambdas
from score_predictor.odds import fair_1x2_probs, fair_over_under_probs


def test_market_lambda_inference_returns_positive_lambdas() -> None:
    fair_1x2 = fair_1x2_probs(1.35, 4.8, 8.5)
    fair_ou = fair_over_under_probs(1.9, 1.95)

    home, away = infer_lambdas_from_market(
        fair_1x2,
        over_probability=fair_ou["over"],
        over_under_line=2.5,
    )

    assert home > 0
    assert away > 0

    probs = probs_from_lambdas(home, away)
    assert 0 < probs["home"] < 1
    assert 0 < probs["draw"] < 1
    assert 0 < probs["away"] < 1
