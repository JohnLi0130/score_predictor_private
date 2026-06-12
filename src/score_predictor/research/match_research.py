from __future__ import annotations

from typing import Any

from score_predictor.connectors.lineup_extractor import extract_lineup_candidates
from score_predictor.connectors.odds_api import fetch_international_odds
from score_predictor.connectors.official_page_fetcher import fetch_official_fact_page
from score_predictor.connectors.sporttery_manual import normalize_sporttery_manual
from score_predictor.connectors.weather_open_meteo import fetch_weather_for_match
from score_predictor.market.market_snapshot import compute_snapshot_movements


def _truthy(value: Any) -> bool:
    return bool(value) and str(value).lower() not in {"false", "0", "no", "none"}


def _collect_official_pages(config: dict) -> tuple[list[dict], list[dict], list[str]]:
    urls = (
        config.get("official_source_urls")
        or config.get("official_sources")
        or config.get("sources", {}).get("official_urls", [])
    )
    pages: list[dict] = []
    excluded: list[dict] = []
    warnings: list[str] = []
    for url in urls or []:
        page = fetch_official_fact_page(str(url), config.get("whitelist_path"))
        source_record = {
            "url": url,
            "trust_tier": page.get("trust_tier"),
            "allowed": page.get("allowed"),
            "reason": page.get("reason"),
            "warnings": page.get("warnings", []),
        }
        if page.get("allowed"):
            pages.append({**source_record, "factual_text": page.get("factual_text")})
        else:
            excluded.append(source_record)
        warnings.extend(page.get("warnings", []))
    return pages, excluded, list(dict.fromkeys(warnings))


def _build_weather(config: dict, warnings: list[str]) -> dict:
    weather_config = config.get("weather", {})
    if weather_config is False:
        return {}
    weather_enabled = _truthy(weather_config.get("enabled", False))
    venue = (config.get("match") or {}).get("venue", {}) or {}
    latitude = weather_config.get("latitude", venue.get("latitude"))
    longitude = weather_config.get("longitude", venue.get("longitude"))
    kickoff_time = weather_config.get(
        "kickoff_time", (config.get("match") or {}).get("kickoff_time")
    )
    timezone = weather_config.get(
        "timezone", (config.get("match") or {}).get("timezone", "Asia/Shanghai")
    )
    if not weather_enabled:
        return {}
    if latitude is None or longitude is None or not kickoff_time:
        warnings.append("weather_enabled_but_match_coordinates_or_kickoff_missing")
        return {
            "source": "open-meteo",
            "warnings": ["weather_coordinates_or_kickoff_missing"],
        }
    return fetch_weather_for_match(float(latitude), float(longitude), str(kickoff_time), str(timezone))


def build_match_research_bundle(config: dict) -> dict:
    warnings: list[str] = []
    sources: list[dict] = []
    excluded_sources: list[dict] = []

    match = config.get("match", {})
    facts: dict[str, Any] = {
        "match": match,
        "official_pages": [],
        "lineup_candidates": [],
    }

    official_pages, excluded, page_warnings = _collect_official_pages(config)
    facts["official_pages"] = official_pages
    excluded_sources.extend(excluded)
    warnings.extend(page_warnings)
    for page in official_pages:
        sources.append(
            {
                "url": page.get("url"),
                "trust_tier": page.get("trust_tier"),
                "source_type": "official_fact_page",
            }
        )
        lineup = extract_lineup_candidates(page.get("factual_text") or "")
        facts["lineup_candidates"].append(lineup)
        warnings.extend(lineup.get("warnings", []))

    market: dict[str, Any] = {}
    market_features: dict[str, Any] = {}
    if config.get("sporttery"):
        sporttery = normalize_sporttery_manual(config["sporttery"])
        market = sporttery["market"]
        market_features = sporttery["features"]
        warnings.extend(sporttery.get("warnings", []))
        sources.append(
            {
                "source": sporttery["source"],
                "source_type": "manual_sporttery",
                "requires_manual_confirmation": True,
            }
        )
    else:
        warnings.append("sporttery_manual_market_missing")

    snapshots = config.get("market_snapshots") or []
    if snapshots:
        snapshot_features = compute_snapshot_movements(snapshots)
        market_features["movement"] = snapshot_features
        warnings.extend(snapshot_features.get("warnings", []))

    weather = _build_weather(config, warnings)
    if weather:
        warnings.extend(weather.get("warnings", []))
        sources.append({"source": "open-meteo", "source_type": "weather"})

    international_odds = {}
    odds_config = config.get("international_odds", {}) or {}
    if _truthy(odds_config.get("enabled", False)):
        international_odds = fetch_international_odds(
            sport_key=str(odds_config.get("sport_key", "soccer_fifa_world_cup")),
            regions=str(odds_config.get("regions", "eu,uk,us")),
            markets=str(odds_config.get("markets", "h2h,totals,spreads")),
            api_key=odds_config.get("api_key"),
        )
        warnings.extend(international_odds.get("warnings", []))
        sources.append(
            {
                "source": "international_market_reference",
                "source_type": "optional_odds_api",
            }
        )

    lineup_confirmed = any(
        candidate.get("confirmed")
        for candidate in facts.get("lineup_candidates", [])
    )
    requires_manual_confirmation = not lineup_confirmed or bool(config.get("sporttery"))

    return {
        "facts": facts,
        "market": market,
        "market_features": market_features,
        "weather": weather,
        "international_market_reference": international_odds,
        "sources": sources,
        "excluded_sources": excluded_sources,
        "warnings": list(dict.fromkeys(warnings)),
        "requires_manual_confirmation": requires_manual_confirmation,
        "audit": {
            "source_policy": "facts_only_no_prediction_articles",
            "used_prediction_sources": False,
            "sporttery_entry_mode": "manual",
            "official_sporttery_manual_entry_recommended": True,
            "international_odds_are_reference_only": True,
            "sources": sources,
            "excluded_sources": excluded_sources,
        },
    }

