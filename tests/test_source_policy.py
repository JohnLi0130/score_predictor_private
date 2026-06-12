from __future__ import annotations

from score_predictor.intelligence.source_policy import validate_source


def test_prediction_title_rejected() -> None:
    result = validate_source("世界杯比分预测", "看好主队赢球")
    assert result["allowed"] is False
    assert result["reason"] == "prediction_or_betting_content_detected"


def test_official_squad_title_accepted() -> None:
    result = validate_source("Official squad list", "starting XI and roster update")
    assert result["allowed"] is True
    assert result["reason"] == "fact_source_detected"


def test_unknown_neutral_page_rejected() -> None:
    result = validate_source("Match preview", "General background text.")
    assert result["allowed"] is False
    assert result["reason"] == "source_not_clearly_factual"

