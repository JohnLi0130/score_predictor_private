from __future__ import annotations

from score_predictor.connectors import http_client


def test_whitelisted_official_page_allowed(monkeypatch) -> None:
    monkeypatch.setattr(
        http_client,
        "_request_text_with_retry",
        lambda url, timeout: (200, "Official squad list and starting XI update."),
    )

    result = http_client.fetch_url("https://www.fifa.com/match-centre")

    assert result["allowed"] is True
    assert result["status_code"] == 200
    assert result["trust_tier"] == "tier1_official"


def test_untrusted_domain_rejected_without_fetch(monkeypatch) -> None:
    called = False

    def fake_fetch(url, timeout):
        nonlocal called
        called = True
        return 200, "Should not be fetched."

    monkeypatch.setattr(http_client, "_request_text_with_retry", fake_fetch)

    result = http_client.fetch_url("https://not-allowed.example/article")

    assert result["allowed"] is False
    assert result["reason"] == "domain_not_in_whitelist"
    assert called is False


def test_prediction_or_betting_content_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        http_client,
        "_request_text_with_retry",
        lambda url, timeout: (200, "\u6bd4\u5206\u9884\u6d4b \u7ade\u5f69\u63a8\u8350"),
    )

    result = http_client.fetch_url("https://www.fifa.com/news")

    assert result["allowed"] is False
    assert result["reason"] == "prediction_or_betting_content_detected"

