from __future__ import annotations

from score_predictor.research import match_research


def test_research_bundle_contains_facts_market_weather_and_audit(monkeypatch) -> None:
    monkeypatch.setattr(
        match_research,
        "fetch_official_fact_page",
        lambda url, whitelist_path=None: {
            "url": url,
            "trust_tier": "tier1_official",
            "allowed": True,
            "reason": None,
            "warnings": [],
            "factual_text": "Official starting XI: Cristiano Ronaldo, Bruno Fernandes. 4-3-3",
        },
    )
    monkeypatch.setattr(
        match_research,
        "fetch_weather_for_match",
        lambda latitude, longitude, kickoff_time, timezone: {
            "temperature_c": 20.0,
            "humidity_pct": 60,
            "rain_probability_pct": 10,
            "rain_mm": 0.0,
            "wind_kph": 12.0,
            "weather_code": 1,
            "source": "open-meteo",
            "retrieved_at": "2026-06-09T00:00:00+00:00",
            "warnings": [],
        },
    )

    bundle = match_research.build_match_research_bundle(
        {
            "match": {
                "home_team": "Portugal",
                "away_team": "Nigeria",
                "kickoff_time": "2026-06-10 02:45",
                "timezone": "Asia/Shanghai",
                "venue": {"latitude": 38.761, "longitude": -9.161},
            },
            "official_source_urls": ["https://www.fifa.com/news/team"],
            "weather": {"enabled": True},
            "sporttery": {"spf": {"home": 1.45, "draw": 4.1, "away": 5.8}},
        }
    )

    assert "facts" in bundle
    assert "market" in bundle
    assert "weather" in bundle
    assert "audit" in bundle
    assert bundle["excluded_sources"] == []
    assert bundle["audit"]["used_prediction_sources"] is False


def test_research_bundle_records_excluded_sources(monkeypatch) -> None:
    monkeypatch.setattr(
        match_research,
        "fetch_official_fact_page",
        lambda url, whitelist_path=None: {
            "url": url,
            "trust_tier": "untrusted",
            "allowed": False,
            "reason": "domain_not_in_whitelist",
            "warnings": ["domain_not_in_whitelist"],
        },
    )

    bundle = match_research.build_match_research_bundle(
        {
            "match": {"home_team": "A", "away_team": "B"},
            "official_source_urls": ["https://bad.example/post"],
            "sporttery": {"spf": {"home": 2.0, "draw": 3.2, "away": 3.5}},
        }
    )

    assert bundle["excluded_sources"][0]["reason"] == "domain_not_in_whitelist"

