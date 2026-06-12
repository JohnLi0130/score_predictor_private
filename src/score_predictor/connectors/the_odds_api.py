from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
import os
from pathlib import Path
import re
from typing import Any
import unicodedata
from urllib.parse import urlencode

import yaml

from .cache import RAW_CACHE_DIR, safe_filename_part, write_raw_response


MISSING_API_KEY_MESSAGE = (
    "Missing THE_ODDS_API_KEY. Please set it before fetching The Odds API data."
)

DEFAULT_CONFIG = {
    "base_url": "https://api.the-odds-api.com",
    "timeout_seconds": 20,
    "default_regions": "eu,uk",
    "default_markets": "h2h,spreads,totals",
    "default_odds_format": "decimal",
    "default_date_format": "iso",
    "preferred_bookmakers": [
        "pinnacle",
        "betfair_ex_eu",
        "betfair_exchange",
        "bet365",
        "betfair",
        "matchbook",
    ],
}

TEAM_ALIASES = {
    "bosnia and herzegovina": "bosnia and herzegovina",
    "bosnia & herzegovina": "bosnia and herzegovina",
    "bosnia-herzegovina": "bosnia and herzegovina",
    "bosnia herzegovina": "bosnia and herzegovina",
    "bosnia": "bosnia and herzegovina",
    "korea republic": "korea republic",
    "south korea": "korea republic",
    "czech republic": "czechia",
    "czechia": "czechia",
    "usa": "usa",
    "us": "usa",
    "united states": "usa",
    "united states of america": "usa",
    "u.s.a.": "usa",
    "turkiye": "turkiye",
    "turkey": "turkiye",
    "türkiye": "turkiye",
    "cote divoire": "cote divoire",
    "côte d’ivoire": "cote divoire",
    "cote d'ivoire": "cote divoire",
    "ivory coast": "cote divoire",
    "curacao": "curacao",
    "curaçao": "curacao",
    "ir iran": "ir iran",
    "iran": "ir iran",
    "cabo verde": "cabo verde",
    "cape verde": "cabo verde",
    "dr congo": "dr congo",
    "congo dr": "dr congo",
    "democratic republic of the congo": "dr congo",
    "england": "england",
}


@dataclass(frozen=True)
class OddsApiResponse:
    data: Any
    metadata: dict[str, Any]


def request_time_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_api_key() -> str:
    api_key = os.getenv("THE_ODDS_API_KEY")
    if not api_key:
        raise RuntimeError(MISSING_API_KEY_MESSAGE)
    return api_key


def load_provider_config(path: str | Path | None = None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path is not None and Path(path).exists():
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            config.update({key: value for key, value in loaded.items() if key != "api_key"})
    env_map = {
        "base_url": "THE_ODDS_API_BASE_URL",
        "timeout_seconds": "THE_ODDS_API_TIMEOUT_SECONDS",
        "default_regions": "THE_ODDS_API_DEFAULT_REGIONS",
        "default_markets": "THE_ODDS_API_DEFAULT_MARKETS",
        "default_odds_format": "THE_ODDS_API_DEFAULT_ODDS_FORMAT",
        "default_date_format": "THE_ODDS_API_DEFAULT_DATE_FORMAT",
    }
    for key, env_name in env_map.items():
        value = os.getenv(env_name)
        if value:
            config[key] = int(value) if key == "timeout_seconds" else value
    return config


def _headers_subset(headers: Any) -> dict[str, str | None]:
    def get(name: str) -> str | None:
        if headers is None:
            return None
        try:
            return headers.get(name) or headers.get(name.lower())
        except AttributeError:
            return None

    return {
        "x_requests_remaining": get("x-requests-remaining"),
        "x_requests_used": get("x-requests-used"),
        "x_requests_last": get("x-requests-last"),
    }


def _build_url(base_url: str, endpoint: str) -> str:
    return base_url.rstrip("/") + endpoint


def _request_json(
    endpoint: str,
    *,
    params: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    client: Any | None = None,
    safe_match_id: str = "the_odds_api",
    kind: str = "response",
    cache_dir: Path | str = RAW_CACHE_DIR,
) -> OddsApiResponse:
    resolved_config = dict(DEFAULT_CONFIG)
    resolved_config.update(config or {})
    api_key = get_api_key()
    params_without_api_key = {
        key: value for key, value in (params or {}).items() if key != "apiKey"
    }
    request_params = {"apiKey": api_key, **params_without_api_key}
    url = _build_url(str(resolved_config["base_url"]), endpoint)
    request_time = request_time_iso()
    warnings: list[str] = []

    if client is None:
        import httpx

        with httpx.Client(timeout=float(resolved_config["timeout_seconds"])) as http_client:
            response = http_client.get(url, params=request_params)
    else:
        response = client.get(url, params=request_params, timeout=float(resolved_config["timeout_seconds"]))

    status_code = int(getattr(response, "status_code", 0))
    if hasattr(response, "raise_for_status"):
        response.raise_for_status()
    raw_json = response.json()
    cache_file = write_raw_response(
        safe_match_id=safe_match_id,
        kind=kind,
        raw_json=raw_json,
        cache_dir=cache_dir,
    )
    metadata = {
        "api_provider": "The Odds API",
        "request_time": request_time,
        "endpoint": endpoint,
        "params_without_api_key": params_without_api_key,
        "status_code": status_code,
        **_headers_subset(getattr(response, "headers", {})),
        "cache_file": str(cache_file),
        "warnings": warnings,
    }
    return OddsApiResponse(data=raw_json, metadata=metadata)


def fetch_sports(
    all: bool = True,
    *,
    config: dict[str, Any] | None = None,
    client: Any | None = None,
    cache_dir: Path | str = RAW_CACHE_DIR,
) -> OddsApiResponse:
    params = {"all": "true"} if all else {}
    return _request_json(
        "/v4/sports",
        params=params,
        config=config,
        client=client,
        safe_match_id="sports",
        kind="sports",
        cache_dir=cache_dir,
    )


def find_world_cup_sport_key(sports: list[dict[str, Any]] | OddsApiResponse) -> str | None:
    data = sports.data if isinstance(sports, OddsApiResponse) else sports
    if not data:
        return None
    preferred_keys = [
        "soccer_fifa_world_cup",
        "soccer_fifa_world_cup_winner",
        "soccer_world_cup",
    ]
    by_key = {str(row.get("key", "")).lower(): str(row.get("key")) for row in data}
    for key in preferred_keys:
        if key in by_key:
            return by_key[key]
    keywords = ("fifa world cup", "world cup")
    for row in data:
        fields = " ".join(
            str(row.get(field, ""))
            for field in ("key", "title", "description", "group")
        ).lower()
        if "soccer" in fields and any(keyword in fields for keyword in keywords):
            return str(row.get("key"))
    return None


def fetch_events(
    sport_key: str,
    *,
    config: dict[str, Any] | None = None,
    client: Any | None = None,
    cache_dir: Path | str = RAW_CACHE_DIR,
) -> OddsApiResponse:
    return _request_json(
        f"/v4/sports/{sport_key}/events",
        params={},
        config=config,
        client=client,
        safe_match_id=sport_key,
        kind="events",
        cache_dir=cache_dir,
    )


def _normalize_team_key(name: str) -> str:
    text = str(name or "").strip().casefold()
    text = (
        text.replace("&", " and ")
        .replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
        .replace("´", "'")
    )
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _alias_lookup(aliases: dict[str, str] | None = None) -> dict[str, str]:
    lookup = {
        _normalize_team_key(alias): _normalize_team_key(canonical)
        for alias, canonical in TEAM_ALIASES.items()
    }
    for alias, canonical in (aliases or {}).items():
        lookup[_normalize_team_key(alias)] = _normalize_team_key(canonical)
        lookup[_normalize_team_key(canonical)] = _normalize_team_key(canonical)
    return lookup


def normalize_team_name(name: str, aliases: dict[str, str] | None = None) -> str:
    normalized = _normalize_team_key(name)
    return _alias_lookup(aliases).get(normalized, normalized)


def _match_quality(
    event_home: str,
    event_away: str,
    wanted_home: str,
    wanted_away: str,
) -> float:
    home_score = SequenceMatcher(None, event_home, wanted_home).ratio()
    away_score = SequenceMatcher(None, event_away, wanted_away).ratio()
    return round((home_score + away_score) / 2.0, 3)


def _enriched_event_candidate(
    event: dict[str, Any],
    *,
    home_team: str,
    away_team: str,
    match_quality: float,
    matched_by_alias: bool,
    reversed_home_away: bool,
) -> dict[str, Any]:
    event_copy = dict(event)
    api_home_team = str(event.get("home_team", ""))
    api_away_team = str(event.get("away_team", ""))
    event_copy.update(
        {
            "event_id": event.get("id") or event.get("event_id"),
            "api_home_team": api_home_team,
            "api_away_team": api_away_team,
            "selected_home_team": home_team,
            "selected_away_team": away_team,
            "match_quality": match_quality,
            "matched_by_alias": matched_by_alias,
            "reversed_home_away": reversed_home_away,
            "matched_reversed_home_away": reversed_home_away,
        }
    )
    return event_copy


def match_event_by_teams(
    events: list[dict[str, Any]] | OddsApiResponse,
    home_team: str,
    away_team: str,
    *,
    team_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    data = events.data if isinstance(events, OddsApiResponse) else events
    wanted_home = normalize_team_name(home_team, team_aliases)
    wanted_away = normalize_team_name(away_team, team_aliases)
    raw_wanted_home = _normalize_team_key(home_team)
    raw_wanted_away = _normalize_team_key(away_team)
    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []

    for event in data or []:
        api_home_text = str(event.get("home_team", ""))
        api_away_text = str(event.get("away_team", ""))
        api_home_raw = _normalize_team_key(str(event.get("home_team", "")))
        api_away_raw = _normalize_team_key(str(event.get("away_team", "")))
        api_home = normalize_team_name(api_home_text, team_aliases)
        api_away = normalize_team_name(api_away_text, team_aliases)

        direct_match = api_home == wanted_home and api_away == wanted_away
        reversed_match = api_home == wanted_away and api_away == wanted_home
        matched_by_alias = (
            direct_match
            and (
                api_home_raw != raw_wanted_home
                or api_away_raw != raw_wanted_away
                or api_home_text.strip().casefold() != str(home_team).strip().casefold()
                or api_away_text.strip().casefold() != str(away_team).strip().casefold()
            )
        ) or (
            reversed_match
            and (
                api_home_raw != raw_wanted_away
                or api_away_raw != raw_wanted_home
                or api_home_text.strip().casefold() != str(away_team).strip().casefold()
                or api_away_text.strip().casefold() != str(home_team).strip().casefold()
            )
        )

        if direct_match or reversed_match:
            candidates.append(
                _enriched_event_candidate(
                    event,
                    home_team=home_team,
                    away_team=away_team,
                    match_quality=1.0 if direct_match else 0.95,
                    matched_by_alias=matched_by_alias,
                    reversed_home_away=reversed_match,
                )
            )
            continue

        direct_quality = _match_quality(api_home, api_away, wanted_home, wanted_away)
        reversed_quality = _match_quality(api_home, api_away, wanted_away, wanted_home)
        if max(direct_quality, reversed_quality) >= 0.55:
            reversed_home_away = reversed_quality > direct_quality
            candidates.append(
                _enriched_event_candidate(
                    event,
                    home_team=home_team,
                    away_team=away_team,
                    match_quality=max(direct_quality, reversed_quality),
                    matched_by_alias=False,
                    reversed_home_away=reversed_home_away,
                )
            )

    candidates.sort(key=lambda row: float(row.get("match_quality") or 0.0), reverse=True)
    has_exact = any(float(candidate.get("match_quality") or 0.0) >= 0.95 for candidate in candidates)
    if not candidates:
        warnings.append(
            "The Odds API 当前没有找到该比赛赛事，可能是该场未开放盘口、队名不匹配或赛事尚未收录。"
        )
    elif not has_exact:
        warnings.append("未找到完全匹配赛事，请从候选赛事列表手动选择。")
    elif len(candidates) > 1:
        warnings.append("找到多个候选赛事，请手动确认 event_id。")
    return {"candidates": candidates, "warnings": warnings}


def fetch_event_odds(
    sport_key: str,
    event_id: str,
    *,
    markets: str | None = None,
    regions: str | None = None,
    bookmakers: str | None = None,
    odds_format: str | None = None,
    date_format: str | None = None,
    config: dict[str, Any] | None = None,
    client: Any | None = None,
    cache_dir: Path | str = RAW_CACHE_DIR,
) -> OddsApiResponse:
    resolved_config = dict(DEFAULT_CONFIG)
    resolved_config.update(config or {})
    params: dict[str, Any] = {
        "regions": regions or resolved_config["default_regions"],
        "markets": markets or resolved_config["default_markets"],
        "oddsFormat": odds_format or resolved_config["default_odds_format"],
        "dateFormat": date_format or resolved_config["default_date_format"],
    }
    if bookmakers and bookmakers != "auto":
        params["bookmakers"] = bookmakers
    return _request_json(
        f"/v4/sports/{sport_key}/events/{event_id}/odds",
        params=params,
        config=resolved_config,
        client=client,
        safe_match_id=f"{sport_key}_{event_id}",
        kind="odds",
        cache_dir=cache_dir,
    )


def fetch_event_markets(
    sport_key: str,
    event_id: str,
    regions: str | None = None,
    *,
    config: dict[str, Any] | None = None,
    client: Any | None = None,
    cache_dir: Path | str = RAW_CACHE_DIR,
) -> OddsApiResponse:
    resolved_config = dict(DEFAULT_CONFIG)
    resolved_config.update(config or {})
    params = {"regions": regions or resolved_config["default_regions"]}
    return _request_json(
        f"/v4/sports/{sport_key}/events/{event_id}/markets",
        params=params,
        config=resolved_config,
        client=client,
        safe_match_id=f"{sport_key}_{event_id}",
        kind="markets",
        cache_dir=cache_dir,
    )


def extract_available_market_keys(markets_response: Any) -> list[str]:
    data = markets_response.data if isinstance(markets_response, OddsApiResponse) else markets_response
    keys: list[str] = []

    def add(value: Any) -> None:
        key = str(value or "").strip()
        if key and key not in keys:
            keys.append(key)

    if isinstance(data, dict):
        if isinstance(data.get("markets"), list):
            data = data["markets"]
        elif isinstance(data.get("data"), list):
            data = data["data"]
        elif data.get("key"):
            add(data.get("key"))
            return keys

    if isinstance(data, list):
        for row in data:
            if isinstance(row, str):
                add(row)
            elif isinstance(row, dict):
                add(row.get("key") or row.get("market_key") or row.get("name"))
    return keys


def build_query_string_for_display(params_without_api_key: dict[str, Any]) -> str:
    return urlencode(params_without_api_key)


def safe_match_id_from_teams(home_team: str, away_team: str) -> str:
    return f"{safe_filename_part(home_team)}_vs_{safe_filename_part(away_team)}"
