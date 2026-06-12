import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Set


ALIASES_PATH = Path(__file__).with_name("team_aliases.json")


@lru_cache(maxsize=1)
def load_team_aliases() -> Dict[str, str]:
    """Load team aliases used to standardize data from mixed sources."""

    if not ALIASES_PATH.exists():
        return {}
    with ALIASES_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def normalize_team_name(name: object) -> str:
    """Normalize common country/team aliases without guessing unknown names."""

    if name is None:
        return ""
    text = str(name).strip()
    if not text or text.lower() == "nan":
        return ""

    aliases = load_team_aliases()
    return aliases.get(text, text)


def normalize_team_columns(rows, columns: Iterable[str]):
    """Normalize team columns in a pandas DataFrame-like object."""

    for column in columns:
        if column in rows:
            rows[column] = rows[column].apply(normalize_team_name)
    return rows


def find_unknown_teams(teams: Iterable[str], known_teams: Iterable[str]) -> Set[str]:
    normalized_known = {normalize_team_name(team) for team in known_teams}
    return {
        normalize_team_name(team)
        for team in teams
        if normalize_team_name(team) not in normalized_known
    }

