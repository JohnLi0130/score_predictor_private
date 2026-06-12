from __future__ import annotations

from .schemas import InjurySuspensionInfo, LineupInfo, PlayerInfo


def _level(score: float) -> str:
    if score >= 80:
        return "high"
    if score >= 65:
        return "medium"
    if score >= 45:
        return "low"
    return "very_low"


def _has_absence(injuries: InjurySuspensionInfo | None, keyword: str) -> bool:
    if injuries is None:
        return False
    names = injuries.absent + injuries.doubtful + injuries.suspended
    return any(keyword.lower() in name.lower() for name in names)


def _regulars(players: list[PlayerInfo], positions: set[str] | None = None) -> list[PlayerInfo]:
    result = [
        player
        for player in players
        if player.is_regular_starter and (positions is None or player.position in positions)
    ]
    return result


def _key_attackers(players: list[PlayerInfo]) -> list[PlayerInfo]:
    return [
        player
        for player in players
        if player.position in {"AM", "W", "FW"}
        and (player.role_importance in {"key", "starter"} or player.is_regular_starter)
    ]


def compute_lineup_strength(
    lineup: LineupInfo | None,
    injuries: InjurySuspensionInfo | None = None,
    match_type: str | None = None,
) -> dict:
    if lineup is None or not lineup.starters:
        return {
            "score": None,
            "level": "unknown",
            "drivers": [],
            "warnings": ["lineup_not_confirmed"],
        }

    score = 50.0
    absence_penalty = 0.0
    drivers: list[str] = []
    warnings: list[str] = []
    starters = lineup.starters

    if lineup.confirmed:
        score += 10
        drivers.append("official_lineup_confirmed")
    else:
        warnings.append("lineup_not_confirmed")

    if any(player.position == "GK" and player.is_regular_starter for player in starters):
        score += 10
        drivers.append("main_goalkeeper_starts")
    elif _has_absence(injuries, "gk") or _has_absence(injuries, "goalkeeper"):
        absence_penalty += 10
        warnings.append("main_goalkeeper_absent")

    if len(_regulars(starters, {"CB", "FB", "DM"})) >= 2:
        score += 15
        drivers.append("defensive_core_present")
    if len(_regulars(starters, {"DM", "CM", "AM"})) >= 2:
        score += 15
        drivers.append("midfield_core_present")
    if len(_key_attackers(starters)) >= 2:
        score += 20
        drivers.append("attacking_core_present")
    if len(_regulars(starters)) >= 7:
        score += 10
        drivers.append("seven_regular_starters")

    formation = (lineup.formation or "").strip()
    if match_type in {"friendly", "club_friendly"} and formation.startswith("5"):
        score -= 10
        warnings.append("defensive_five_back_in_friendly")

    if _has_absence(injuries, "striker") or _has_absence(injuries, "fw"):
        absence_penalty += 15
        warnings.append("key_striker_absent")
    if _has_absence(injuries, "playmaker") or _has_absence(injuries, "am"):
        absence_penalty += 10
        warnings.append("key_playmaker_absent")

    bench_like = [
        player
        for player in starters
        if player.role_importance in {"rotation", "bench"} or not player.is_regular_starter
    ]
    if len(bench_like) >= 5:
        score -= 10
        warnings.append("many_youngsters_or_bench_players_start")

    score = max(0.0, min(100.0, score) - absence_penalty)
    return {
        "score": score,
        "level": _level(score),
        "drivers": drivers,
        "warnings": warnings,
    }
