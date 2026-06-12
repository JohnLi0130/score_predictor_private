from __future__ import annotations

from score_predictor.connectors.source_whitelist import (
    classify_trust_tier,
    get_domain,
    is_domain_allowed,
    load_whitelist,
)


def test_get_domain_normalizes_www_and_path() -> None:
    assert get_domain("https://www.fifa.com/en/matches") == "fifa.com"


def test_classify_trust_tier_allows_subdomains() -> None:
    whitelist = load_whitelist()
    assert classify_trust_tier("https://inside.fifa.com/news", whitelist) == "tier1_official"
    assert is_domain_allowed("https://api.open-meteo.com/v1/forecast", whitelist)


def test_untrusted_domain_rejected() -> None:
    whitelist = load_whitelist()
    assert classify_trust_tier("https://example-bets.test/post", whitelist) == "untrusted"
    assert not is_domain_allowed("https://example-bets.test/post", whitelist)

