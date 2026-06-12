from __future__ import annotations

from .lineup_strength import compute_lineup_strength
from .match_context import compute_match_intensity
from .narrative_risk import compute_narrative_heat
from .schemas import IntelligenceInput


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _team_has_absence(intel: IntelligenceInput, team: str, keyword: str) -> bool:
    injuries = intel.injuries_suspensions.get(team)
    if injuries is None:
        return False
    items = injuries.absent + injuries.doubtful + injuries.suspended
    return any(keyword.lower() in item.lower() for item in items)


def build_intelligence_adjustments(
    intel: IntelligenceInput | None,
    base_market_weight: float,
) -> dict:
    if intel is None:
        return {
            "market_weight": base_market_weight,
            "home_lambda_factor": 1.0,
            "away_lambda_factor": 1.0,
            "total_lambda_factor": 1.0,
            "warnings": ["intelligence_missing"],
            "drivers": [],
            "diagnostics": {
                "lineup_strength": {
                    "home": {"score": None, "level": "unknown"},
                    "away": {"score": None, "level": "unknown"},
                },
                "match_intensity": compute_match_intensity(None),
                "narrative_heat": compute_narrative_heat(None),
            },
        }

    market_weight = base_market_weight
    home_factor = 1.0
    away_factor = 1.0
    total_factor = 1.0
    warnings: list[str] = []
    drivers: list[str] = []

    if intel.match_type == "friendly":
        total_factor *= 0.88
        warnings.append("friendly_match_total_goals_discount")
    elif intel.match_type == "club_friendly":
        total_factor *= 0.82
        warnings.append("club_friendly_match_total_goals_discount")

    if not intel.official_lineups_available:
        warnings.append("official_lineups_not_available")
    if not intel.official_squads_available:
        warnings.append("official_squads_not_available")

    home_lsi = compute_lineup_strength(
        intel.lineups.get("home"),
        intel.injuries_suspensions.get("home"),
        match_type=intel.match_type,
    )
    away_lsi = compute_lineup_strength(
        intel.lineups.get("away"),
        intel.injuries_suspensions.get("away"),
        match_type=intel.match_type,
    )
    match_intensity = compute_match_intensity(intel)
    narrative_heat = compute_narrative_heat(intel.narrative_flags)

    if home_lsi["score"] is not None and home_lsi["score"] < 65:
        home_factor *= 0.88
        warnings.append("home_low_lineup_strength")
    if away_lsi["score"] is not None and away_lsi["score"] < 65:
        away_factor *= 0.88
        warnings.append("away_low_lineup_strength")

    home_low = home_lsi["score"] is not None and home_lsi["score"] < 70
    away_low = away_lsi["score"] is not None and away_lsi["score"] < 70
    if home_low and away_low:
        total_factor *= 0.90
        warnings.append("both_teams_rotation_or_low_strength")

    home_formation = (intel.lineups.get("home").formation if intel.lineups.get("home") else "") or ""
    away_formation = (intel.lineups.get("away").formation if intel.lineups.get("away") else "") or ""
    if intel.match_type in {"friendly", "club_friendly"} and home_formation.startswith("5"):
        away_factor *= 0.90
        total_factor *= 0.94
        warnings.append("home_five_back_friendly_discount")
    if intel.match_type in {"friendly", "club_friendly"} and away_formation.startswith("5"):
        home_factor *= 0.90
        total_factor *= 0.94
        warnings.append("away_five_back_friendly_discount")

    for team, attr in (("home", "home"), ("away", "away")):
        if _team_has_absence(intel, team, "striker") or _team_has_absence(intel, team, "fw"):
            if attr == "home":
                home_factor *= 0.88
            else:
                away_factor *= 0.88
            warnings.append(f"{team}_key_striker_absent")
        if _team_has_absence(intel, team, "playmaker") or _team_has_absence(intel, team, "am"):
            if attr == "home":
                home_factor *= 0.92
            else:
                away_factor *= 0.92
            warnings.append(f"{team}_key_playmaker_absent")

    if intel.weather:
        if (
            intel.weather.temperature_c is not None
            and intel.weather.temperature_c >= 30
            and intel.weather.humidity_pct is not None
            and intel.weather.humidity_pct >= 70
        ):
            total_factor *= 0.92
            warnings.append("heat_humidity_total_goals_discount")
        if (
            intel.weather.rain_probability_pct is not None
            and intel.weather.rain_probability_pct >= 60
        ):
            total_factor *= 0.95
            warnings.append("rain_total_goals_discount")

    if narrative_heat["level"] == "high":
        market_weight -= 0.10
        warnings.append("market_may_be_public_sentiment_polluted")
    elif narrative_heat["level"] == "medium":
        market_weight -= 0.05

    drivers.extend(home_lsi.get("drivers", []))
    drivers.extend(away_lsi.get("drivers", []))
    drivers.extend(match_intensity.get("drivers", []))
    drivers.extend(narrative_heat.get("drivers", []))
    warnings.extend(home_lsi.get("warnings", []))
    warnings.extend(away_lsi.get("warnings", []))
    warnings.extend(match_intensity.get("warnings", []))

    return {
        "market_weight": _clamp(market_weight, 0.40, 0.80),
        "home_lambda_factor": _clamp(home_factor, 0.70, 1.25),
        "away_lambda_factor": _clamp(away_factor, 0.70, 1.25),
        "total_lambda_factor": _clamp(total_factor, 0.70, 1.15),
        "warnings": list(dict.fromkeys(warnings)),
        "drivers": list(dict.fromkeys(drivers)),
        "diagnostics": {
            "lineup_strength": {"home": home_lsi, "away": away_lsi},
            "match_intensity": match_intensity,
            "narrative_heat": narrative_heat,
        },
    }

