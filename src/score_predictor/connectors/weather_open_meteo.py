from __future__ import annotations

import json
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import urlopen

from .base import utc_now_iso
from .normalizers import parse_datetime

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_FIELDS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation_probability",
    "rain",
    "wind_speed_10m",
    "weather_code",
]


def _fetch_json(url: str, timeout_seconds: int = 15) -> dict:
    try:
        import httpx

        response = httpx.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        return response.json()
    except ModuleNotFoundError:
        with urlopen(url, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


def _nearest_hour_index(times: list[str], kickoff: datetime) -> int | None:
    if not times:
        return None
    parsed_times = [parse_datetime(value) for value in times]
    if kickoff.tzinfo is not None:
        parsed_times = [
            value.replace(tzinfo=kickoff.tzinfo) if value.tzinfo is None else value
            for value in parsed_times
        ]
    deltas = [
        abs((value - kickoff).total_seconds())
        for value in parsed_times
    ]
    return min(range(len(deltas)), key=deltas.__getitem__)


def _value_at(hourly: dict, field: str, index: int, warnings: list[str]):
    values = hourly.get(field)
    if not isinstance(values, list) or index >= len(values):
        warnings.append(f"open_meteo_missing_field:{field}")
        return None
    return values[index]


def fetch_weather_for_match(
    latitude: float,
    longitude: float,
    kickoff_time: str,
    timezone: str,
) -> dict:
    warnings: list[str] = []
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(HOURLY_FIELDS),
        "timezone": timezone,
    }
    url = f"{OPEN_METEO_URL}?{urlencode(params)}"
    retrieved_at = utc_now_iso()

    payload = _fetch_json(url)
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    kickoff = parse_datetime(kickoff_time)
    index = _nearest_hour_index(times, kickoff)
    if index is None:
        return {
            "temperature_c": None,
            "humidity_pct": None,
            "rain_probability_pct": None,
            "rain_mm": None,
            "wind_kph": None,
            "weather_code": None,
            "source": "open-meteo",
            "retrieved_at": retrieved_at,
            "warnings": ["open_meteo_hourly_time_missing"],
        }

    return {
        "temperature_c": _value_at(hourly, "temperature_2m", index, warnings),
        "humidity_pct": _value_at(hourly, "relative_humidity_2m", index, warnings),
        "rain_probability_pct": _value_at(
            hourly, "precipitation_probability", index, warnings
        ),
        "rain_mm": _value_at(hourly, "rain", index, warnings),
        "wind_kph": _value_at(hourly, "wind_speed_10m", index, warnings),
        "weather_code": _value_at(hourly, "weather_code", index, warnings),
        "source": "open-meteo",
        "retrieved_at": retrieved_at,
        "warnings": list(dict.fromkeys(warnings)),
    }

