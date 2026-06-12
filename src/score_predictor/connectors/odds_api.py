from __future__ import annotations

import json
import os
import statistics
from urllib.parse import urlencode
from urllib.request import urlopen

from .base import utc_now_iso

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4/sports"


def _fetch_json(url: str, timeout_seconds: int = 15) -> list[dict]:
    try:
        import httpx

        response = httpx.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        return response.json()
    except ModuleNotFoundError:
        with urlopen(url, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


def _median_odds(events: list[dict]) -> dict[str, float]:
    by_outcome: dict[str, list[float]] = {}
    for event in events:
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                for outcome in market.get("outcomes", []):
                    name = str(outcome.get("name"))
                    price = outcome.get("price")
                    if isinstance(price, (int, float)):
                        by_outcome.setdefault(name, []).append(float(price))
    return {
        outcome: statistics.median(values)
        for outcome, values in by_outcome.items()
        if values
    }


def fetch_international_odds(
    sport_key: str,
    regions: str = "eu,uk,us",
    markets: str = "h2h,totals,spreads",
    api_key: str | None = None,
) -> dict:
    resolved_key = api_key or os.getenv("THE_ODDS_API_KEY")
    if not resolved_key:
        return {
            "source": "international_market_reference",
            "bookmaker_count": 0,
            "median_odds": {},
            "market_disagreement": {},
            "last_update": None,
            "retrieved_at": utc_now_iso(),
            "warnings": ["the_odds_api_key_missing"],
        }

    params = {
        "apiKey": resolved_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
    }
    url = f"{ODDS_API_BASE_URL}/{sport_key}/odds?{urlencode(params)}"
    events = _fetch_json(url)
    bookmaker_keys = {
        bookmaker.get("key")
        for event in events
        for bookmaker in event.get("bookmakers", [])
        if bookmaker.get("key")
    }
    last_updates = [
        bookmaker.get("last_update")
        for event in events
        for bookmaker in event.get("bookmakers", [])
        if bookmaker.get("last_update")
    ]
    medians = _median_odds(events)

    disagreement: dict[str, float] = {}
    by_outcome: dict[str, list[float]] = {}
    for event in events:
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                for outcome in market.get("outcomes", []):
                    price = outcome.get("price")
                    if isinstance(price, (int, float)):
                        by_outcome.setdefault(str(outcome.get("name")), []).append(
                            float(price)
                        )
    for outcome, values in by_outcome.items():
        if len(values) >= 2:
            disagreement[outcome] = max(values) - min(values)

    return {
        "source": "international_market_reference",
        "bookmaker_count": len(bookmaker_keys),
        "median_odds": medians,
        "market_disagreement": disagreement,
        "last_update": max(last_updates) if last_updates else None,
        "retrieved_at": utc_now_iso(),
        "warnings": [],
    }

