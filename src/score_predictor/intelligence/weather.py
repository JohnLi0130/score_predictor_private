from __future__ import annotations

from .schemas import WeatherInfo


def weather_warnings(weather: WeatherInfo | None) -> list[str]:
    if weather is None:
        return []
    warnings: list[str] = []
    if (
        weather.temperature_c is not None
        and weather.temperature_c >= 30
        and weather.humidity_pct is not None
        and weather.humidity_pct >= 70
    ):
        warnings.append("heat_humidity_total_goals_discount")
    if weather.rain_probability_pct is not None and weather.rain_probability_pct >= 60:
        warnings.append("rain_total_goals_discount")
    return warnings

