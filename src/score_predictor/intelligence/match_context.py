from __future__ import annotations

from .schemas import IntelligenceInput


BASE_SCORES = {
    "world_cup": 75.0,
    "continental_cup": 75.0,
    "qualifier": 75.0,
    "nations_league": 65.0,
    "friendly": 45.0,
    "club_friendly": 35.0,
    "unknown": 50.0,
}


def _level(score: float) -> str:
    if score >= 75:
        return "high"
    if score >= 55:
        return "medium"
    return "low"


def compute_match_intensity(intel: IntelligenceInput | None) -> dict:
    if intel is None:
        return {
            "score": 50.0,
            "level": "low",
            "drivers": ["intelligence_missing"],
            "warnings": ["match_context_unknown"],
        }

    score = BASE_SCORES.get(intel.match_type, 50.0)
    drivers = [f"match_type_{intel.match_type}"]
    warnings: list[str] = []

    importance = {key: value.lower() for key, value in intel.match_importance.items()}
    if importance.get("home") == "must_win" and importance.get("away") == "must_win":
        score += 10
        drivers.append("both_teams_must_win")
    if any(value in {"rivalry", "ranking_points"} for value in importance.values()):
        score += 5
        drivers.append("ranking_or_rivalry_context")
    if all(value in {"experimenting", "rotation"} for value in importance.values()) and importance:
        score -= 10
        warnings.append("both_teams_experimenting")
    if any(value == "major_rotation" for value in importance.values()):
        score -= 10
        warnings.append("major_rotation_likely")

    if intel.weather:
        hot = (
            intel.weather.temperature_c is not None
            and intel.weather.temperature_c >= 30
            and intel.weather.humidity_pct is not None
            and intel.weather.humidity_pct >= 70
        )
        if hot:
            score -= 8
            warnings.append("extreme_heat_humidity")
        if (
            intel.weather.rain_probability_pct is not None
            and intel.weather.rain_probability_pct >= 60
        ):
            score -= 5
            warnings.append("heavy_rain_risk")

    if intel.narrative_flags.ceremonial_match and intel.match_type in {
        "friendly",
        "club_friendly",
        "unknown",
    }:
        score -= 8
        warnings.append("ceremonial_narrative_without_competitive_stakes")

    score = max(0.0, min(100.0, score))
    return {
        "score": score,
        "level": _level(score),
        "drivers": drivers,
        "warnings": warnings,
    }

