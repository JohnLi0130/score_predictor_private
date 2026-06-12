from __future__ import annotations

from typing import Any

from .intelligence.schemas import IntelligenceInput


def compute_data_quality(input_data: Any, intel: IntelligenceInput | None) -> dict:
    score = 100.0
    warnings: list[str] = []

    if intel is None:
        return {
            "score": 55.0,
            "level": "low",
            "warnings": ["intelligence_missing"],
        }

    if not intel.official_lineups_available:
        score -= 25
        warnings.append("no_official_lineups")
    if not intel.official_squads_available:
        score -= 15
        warnings.append("no_official_squads")
    if not intel.injuries_suspensions:
        score -= 10
        warnings.append("no_injury_suspension_info")
    if intel.weather is None:
        score -= 10
        warnings.append("no_venue_weather")
    if getattr(input_data, "over_under", None) is None:
        score -= 10
        warnings.append("no_over_under_odds")
    if getattr(input_data, "asian_handicap", None) is None:
        score -= 10
        warnings.append("no_asian_handicap")
    if intel.source_mode == "manual" and not intel.sources:
        score -= 15
        warnings.append("manual_only_without_factual_source_record")
    if intel.conflicting_information:
        score -= 20
        warnings.append("conflicting_information")

    score = max(0.0, min(100.0, score))
    if score >= 80:
        level = "high"
    elif score >= 60:
        level = "medium"
    else:
        level = "low"

    return {"score": score, "level": level, "warnings": warnings}

