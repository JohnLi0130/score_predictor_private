from __future__ import annotations

import pytest

from score_predictor.connectors.the_odds_api import (
    MISSING_API_KEY_MESSAGE,
    extract_available_market_keys,
    fetch_event_markets,
    fetch_sports,
    find_world_cup_sport_key,
    match_event_by_teams,
    fetch_event_odds,
    get_api_key,
    normalize_team_name,
)


class FakeResponse:
    def __init__(self, data, status_code: int = 200):
        self._data = data
        self.status_code = status_code
        self.headers = {
            "x-requests-remaining": "99",
            "x-requests-used": "1",
            "x-requests-last": "1",
        }

    def json(self):
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("http error")


class FakeClient:
    def __init__(self, data):
        self.data = data
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse(self.data)


def test_missing_api_key_raises_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match=MISSING_API_KEY_MESSAGE):
        get_api_key()


def test_api_key_not_in_metadata(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("THE_ODDS_API_KEY", "secret-key-never-log")
    client = FakeClient([{"key": "soccer_fifa_world_cup", "title": "FIFA World Cup"}])

    response = fetch_sports(client=client, cache_dir=tmp_path)

    assert "secret-key-never-log" not in str(response.metadata)
    assert "apiKey" not in response.metadata["params_without_api_key"]
    assert client.calls[0]["params"]["apiKey"] == "secret-key-never-log"


def test_find_world_cup_sport_key_from_mock_sports() -> None:
    sports = [
        {"key": "soccer_epl", "title": "EPL"},
        {"key": "soccer_fifa_world_cup", "title": "FIFA World Cup"},
    ]

    assert find_world_cup_sport_key(sports) == "soccer_fifa_world_cup"


def test_match_event_by_team_aliases() -> None:
    events = [
        {
            "id": "evt_1",
            "home_team": "South Korea",
            "away_team": "Czechia",
            "commence_time": "2026-06-12T12:00:00Z",
        }
    ]

    result = match_event_by_teams(events, "Korea Republic", "Czech Republic")

    assert result["warnings"] == []
    assert result["candidates"][0]["id"] == "evt_1"


def test_normalize_team_name_matches_bosnia_ampersand_alias() -> None:
    aliases = {
        "Bosnia and Herzegovina": "Bosnia and Herzegovina",
        "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    }

    assert normalize_team_name("Bosnia & Herzegovina", aliases) == normalize_team_name(
        "Bosnia and Herzegovina",
        aliases,
    )


def test_match_event_by_teams_matches_canada_bosnia_alias() -> None:
    events = [
        {
            "id": "d1f4f946c70a0b4e81f5d43e9d32361c",
            "home_team": "Canada",
            "away_team": "Bosnia & Herzegovina",
            "commence_time": "2026-06-12T19:00:00Z",
        }
    ]

    result = match_event_by_teams(events, "Canada", "Bosnia and Herzegovina")

    assert result["warnings"] == []
    candidate = result["candidates"][0]
    assert candidate["event_id"] == "d1f4f946c70a0b4e81f5d43e9d32361c"
    assert candidate["api_home_team"] == "Canada"
    assert candidate["api_away_team"] == "Bosnia & Herzegovina"
    assert candidate["matched_by_alias"] is True
    assert candidate["reversed_home_away"] is False


def test_match_event_by_teams_reversed_home_away_becomes_candidate() -> None:
    events = [
        {
            "id": "evt_reversed",
            "home_team": "Bosnia & Herzegovina",
            "away_team": "Canada",
            "commence_time": "2026-06-12T19:00:00Z",
        }
    ]

    result = match_event_by_teams(events, "Canada", "Bosnia and Herzegovina")

    assert result["warnings"] == []
    candidate = result["candidates"][0]
    assert candidate["event_id"] == "evt_reversed"
    assert candidate["reversed_home_away"] is True
    assert candidate["match_quality"] == 0.95


def test_fetch_event_odds_uses_safe_metadata(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("THE_ODDS_API_KEY", "secret-key-never-log")
    client = FakeClient({"id": "evt_1", "bookmakers": []})

    response = fetch_event_odds(
        "soccer_fifa_world_cup",
        "evt_1",
        markets="h2h,spreads,totals",
        regions="eu,uk",
        client=client,
        cache_dir=tmp_path,
    )

    assert response.metadata["endpoint"] == "/v4/sports/soccer_fifa_world_cup/events/evt_1/odds"
    assert response.metadata["params_without_api_key"]["markets"] == "h2h,spreads,totals"
    assert "secret-key-never-log" not in str(response.metadata)


def test_fetch_event_markets_uses_endpoint_and_extracts_keys(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("THE_ODDS_API_KEY", "secret-key-never-log")
    client = FakeClient(
        [
            {"key": "h2h", "title": "Head to Head"},
            {"key": "totals", "title": "Totals"},
            {"key": "btts", "title": "Both Teams To Score"},
        ]
    )

    response = fetch_event_markets(
        "soccer_fifa_world_cup",
        "evt_1",
        regions="eu,uk",
        client=client,
        cache_dir=tmp_path,
    )

    assert response.metadata["endpoint"] == "/v4/sports/soccer_fifa_world_cup/events/evt_1/markets"
    assert response.metadata["params_without_api_key"]["regions"] == "eu,uk"
    assert "secret-key-never-log" not in str(response.metadata)
    assert extract_available_market_keys(response) == ["h2h", "totals", "btts"]
