from __future__ import annotations

from .schemas import NarrativeFlags


def compute_narrative_heat(flags: NarrativeFlags | None) -> dict:
    if flags is None:
        flags = NarrativeFlags()

    score = 0.0
    drivers: list[str] = []

    if flags.coach_debut:
        score += 25
        drivers.append("coach_debut")
    if flags.player_milestone:
        score += 25
        drivers.append("player_milestone")
    if flags.public_hype_home:
        score += 20
        drivers.append("public_hype_home")
    if flags.revenge_talk:
        score += 15
        drivers.append("revenge_talk")
    if flags.public_hype_away:
        score += 15
        drivers.append("public_hype_away")
    if flags.ceremonial_match:
        score += 10
        drivers.append("ceremonial_match")

    if score >= 60:
        level = "high"
    elif score >= 30:
        level = "medium"
    else:
        level = "low"

    return {"score": min(score, 100.0), "level": level, "drivers": drivers}

