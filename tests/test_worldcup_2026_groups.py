from __future__ import annotations

from pathlib import Path

from score_predictor.connectors.the_odds_api import normalize_team_name
from score_predictor.ui.streamlit_app import WORLD_CUP_GROUPS_PATH
from score_predictor.ui.worldcup_config import (
    api_match_name,
    load_team_aliases,
    load_worldcup_groups,
    team_label,
)


def _canonical_names(group: list[dict]) -> list[str]:
    return [team["canonical_name"] for team in group]


def test_group_a_is_correct() -> None:
    groups = load_worldcup_groups(WORLD_CUP_GROUPS_PATH)

    assert _canonical_names(groups["A"]) == [
        "Mexico",
        "South Africa",
        "Korea Republic",
        "Czechia",
    ]


def test_norway_not_in_group_a_and_in_group_i() -> None:
    groups = load_worldcup_groups(WORLD_CUP_GROUPS_PATH)

    assert "Norway" not in _canonical_names(groups["A"])
    assert "Norway" in _canonical_names(groups["I"])


def test_every_group_has_four_teams_and_total_is_48() -> None:
    groups = load_worldcup_groups(WORLD_CUP_GROUPS_PATH)

    assert len(groups) == 12
    assert all(len(teams) == 4 for teams in groups.values())
    assert sum(len(teams) for teams in groups.values()) == 48


def test_ui_group_team_labels_only_use_selected_group() -> None:
    groups = load_worldcup_groups(WORLD_CUP_GROUPS_PATH)
    labels = [team_label(team) for team in groups["A"]]

    assert labels == [
        "墨西哥 / Mexico",
        "南非 / South Africa",
        "韩国 / Korea Republic",
        "捷克 / Czechia",
    ]
    assert "挪威 / Norway" not in labels


def test_api_match_name_uses_api_names_not_display_name() -> None:
    groups = load_worldcup_groups(WORLD_CUP_GROUPS_PATH)
    czechia = groups["A"][3]

    assert team_label(czechia) == "捷克 / Czechia"
    assert api_match_name(czechia) == "Czechia"


def test_fallback_groups_are_correct_when_config_missing(tmp_path) -> None:
    groups = load_worldcup_groups(tmp_path / "missing_groups.yaml")

    assert _canonical_names(groups["A"]) == [
        "Mexico",
        "South Africa",
        "Korea Republic",
        "Czechia",
    ]
    assert "Norway" in _canonical_names(groups["I"])


def test_team_aliases_match_common_api_names() -> None:
    aliases = load_team_aliases(Path("config/team_aliases_worldcup_2026.yaml"))

    assert normalize_team_name("South Korea", aliases) == normalize_team_name(
        "Korea Republic",
        aliases,
    )
    assert normalize_team_name("Czech Republic", aliases) == normalize_team_name(
        "Czechia",
        aliases,
    )
    assert normalize_team_name("United States", aliases) == normalize_team_name(
        "USA",
        aliases,
    )
    assert normalize_team_name("Bosnia & Herzegovina", aliases) == normalize_team_name(
        "Bosnia and Herzegovina",
        aliases,
    )
