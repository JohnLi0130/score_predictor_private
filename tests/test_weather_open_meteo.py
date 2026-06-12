from __future__ import annotations

from score_predictor.connectors import weather_open_meteo


def test_open_meteo_selects_nearest_hour(monkeypatch) -> None:
    monkeypatch.setattr(
        weather_open_meteo,
        "_fetch_json",
        lambda url: {
            "hourly": {
                "time": ["2026-06-10T01:00", "2026-06-10T03:00"],
                "temperature_2m": [18.0, 20.0],
                "relative_humidity_2m": [60, 65],
                "precipitation_probability": [10, 30],
                "rain": [0.0, 0.2],
                "wind_speed_10m": [12.0, 15.0],
                "weather_code": [1, 3],
            }
        },
    )

    result = weather_open_meteo.fetch_weather_for_match(
        38.761, -9.161, "2026-06-10 02:45", "Asia/Shanghai"
    )

    assert result["temperature_c"] == 20.0
    assert result["rain_probability_pct"] == 30
    assert result["warnings"] == []


def test_open_meteo_missing_field_warns(monkeypatch) -> None:
    monkeypatch.setattr(
        weather_open_meteo,
        "_fetch_json",
        lambda url: {"hourly": {"time": ["2026-06-10T03:00"]}},
    )

    result = weather_open_meteo.fetch_weather_for_match(
        38.761, -9.161, "2026-06-10 02:45", "Asia/Shanghai"
    )

    assert result["temperature_c"] is None
    assert "open_meteo_missing_field:temperature_2m" in result["warnings"]

