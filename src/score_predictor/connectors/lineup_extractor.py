from __future__ import annotations

import re

OFFICIAL_LINEUP_MARKERS = [
    "official starting xi",
    "starting xi",
    "starting lineup",
    "\u9996\u53d1\u540d\u5355",
    "\u9996\u53d1\u9635\u5bb9",
]


def extract_lineup_candidates(text: str) -> dict:
    normalized = text.lower()
    explicit_lineup = any(marker in normalized for marker in OFFICIAL_LINEUP_MARKERS)
    formation_candidates = sorted(
        set(re.findall(r"\b[1-5]-[1-5]-[1-5](?:-[1-5])?\b", text))
    )

    player_candidates: list[str] = []
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", text):
        name = match.group(0).strip()
        if name.lower() in {"official starting", "starting lineup"}:
            continue
        player_candidates.append(name)
    players = list(dict.fromkeys(player_candidates))[:30]

    warnings: list[str] = []
    if not explicit_lineup:
        warnings.append("lineup_not_explicitly_confirmed")
    if not players:
        warnings.append("lineup_players_not_detected")

    return {
        "players": players,
        "formation_candidates": formation_candidates,
        "confidence": "medium" if explicit_lineup and players else "low",
        "confirmed": bool(explicit_lineup and players),
        "requires_manual_confirmation": not bool(explicit_lineup and players),
        "warnings": warnings,
    }

