from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_WORLD_CUP_2026_GROUPS: dict[str, list[dict[str, Any]]] = {
    "A": [
        {"display_name": "墨西哥", "canonical_name": "Mexico", "api_names": ["Mexico", "México"]},
        {"display_name": "南非", "canonical_name": "South Africa", "api_names": ["South Africa"]},
        {"display_name": "韩国", "canonical_name": "Korea Republic", "api_names": ["Korea Republic", "South Korea"]},
        {"display_name": "捷克", "canonical_name": "Czechia", "api_names": ["Czechia", "Czech Republic"]},
    ],
    "B": [
        {"display_name": "加拿大", "canonical_name": "Canada", "api_names": ["Canada"]},
        {"display_name": "波黑", "canonical_name": "Bosnia and Herzegovina", "api_names": ["Bosnia and Herzegovina", "Bosnia & Herzegovina", "Bosnia-Herzegovina", "Bosnia Herzegovina", "Bosnia"]},
        {"display_name": "卡塔尔", "canonical_name": "Qatar", "api_names": ["Qatar"]},
        {"display_name": "瑞士", "canonical_name": "Switzerland", "api_names": ["Switzerland"]},
    ],
    "C": [
        {"display_name": "巴西", "canonical_name": "Brazil", "api_names": ["Brazil"]},
        {"display_name": "摩洛哥", "canonical_name": "Morocco", "api_names": ["Morocco"]},
        {"display_name": "苏格兰", "canonical_name": "Scotland", "api_names": ["Scotland"]},
        {"display_name": "海地", "canonical_name": "Haiti", "api_names": ["Haiti"]},
    ],
    "D": [
        {"display_name": "美国", "canonical_name": "USA", "api_names": ["USA", "United States", "United States of America"]},
        {"display_name": "澳大利亚", "canonical_name": "Australia", "api_names": ["Australia"]},
        {"display_name": "巴拉圭", "canonical_name": "Paraguay", "api_names": ["Paraguay"]},
        {"display_name": "土耳其", "canonical_name": "Türkiye", "api_names": ["Türkiye", "Turkey", "Turkiye"]},
    ],
    "E": [
        {"display_name": "德国", "canonical_name": "Germany", "api_names": ["Germany"]},
        {"display_name": "厄瓜多尔", "canonical_name": "Ecuador", "api_names": ["Ecuador"]},
        {"display_name": "科特迪瓦", "canonical_name": "Côte d’Ivoire", "api_names": ["Côte d’Ivoire", "Cote d'Ivoire", "Ivory Coast"]},
        {"display_name": "库拉索", "canonical_name": "Curaçao", "api_names": ["Curaçao", "Curacao"]},
    ],
    "F": [
        {"display_name": "荷兰", "canonical_name": "Netherlands", "api_names": ["Netherlands", "Holland"]},
        {"display_name": "日本", "canonical_name": "Japan", "api_names": ["Japan"]},
        {"display_name": "瑞典", "canonical_name": "Sweden", "api_names": ["Sweden"]},
        {"display_name": "突尼斯", "canonical_name": "Tunisia", "api_names": ["Tunisia"]},
    ],
    "G": [
        {"display_name": "比利时", "canonical_name": "Belgium", "api_names": ["Belgium"]},
        {"display_name": "伊朗", "canonical_name": "IR Iran", "api_names": ["IR Iran", "Iran"]},
        {"display_name": "埃及", "canonical_name": "Egypt", "api_names": ["Egypt"]},
        {"display_name": "新西兰", "canonical_name": "New Zealand", "api_names": ["New Zealand"]},
    ],
    "H": [
        {"display_name": "西班牙", "canonical_name": "Spain", "api_names": ["Spain"]},
        {"display_name": "乌拉圭", "canonical_name": "Uruguay", "api_names": ["Uruguay"]},
        {"display_name": "沙特阿拉伯", "canonical_name": "Saudi Arabia", "api_names": ["Saudi Arabia"]},
        {"display_name": "佛得角", "canonical_name": "Cabo Verde", "api_names": ["Cabo Verde", "Cape Verde"]},
    ],
    "I": [
        {"display_name": "法国", "canonical_name": "France", "api_names": ["France"]},
        {"display_name": "塞内加尔", "canonical_name": "Senegal", "api_names": ["Senegal"]},
        {"display_name": "伊拉克", "canonical_name": "Iraq", "api_names": ["Iraq"]},
        {"display_name": "挪威", "canonical_name": "Norway", "api_names": ["Norway"]},
    ],
    "J": [
        {"display_name": "阿根廷", "canonical_name": "Argentina", "api_names": ["Argentina"]},
        {"display_name": "阿尔及利亚", "canonical_name": "Algeria", "api_names": ["Algeria"]},
        {"display_name": "奥地利", "canonical_name": "Austria", "api_names": ["Austria"]},
        {"display_name": "约旦", "canonical_name": "Jordan", "api_names": ["Jordan"]},
    ],
    "K": [
        {"display_name": "葡萄牙", "canonical_name": "Portugal", "api_names": ["Portugal"]},
        {"display_name": "哥伦比亚", "canonical_name": "Colombia", "api_names": ["Colombia"]},
        {"display_name": "乌兹别克斯坦", "canonical_name": "Uzbekistan", "api_names": ["Uzbekistan"]},
        {"display_name": "刚果民主共和国", "canonical_name": "DR Congo", "api_names": ["DR Congo", "Congo DR", "Democratic Republic of the Congo"]},
    ],
    "L": [
        {"display_name": "英格兰", "canonical_name": "England", "api_names": ["England"]},
        {"display_name": "克罗地亚", "canonical_name": "Croatia", "api_names": ["Croatia"]},
        {"display_name": "加纳", "canonical_name": "Ghana", "api_names": ["Ghana"]},
        {"display_name": "巴拿马", "canonical_name": "Panama", "api_names": ["Panama"]},
    ],
}

DEFAULT_TEAM_ALIASES: dict[str, list[str]] = {
    "Bosnia and Herzegovina": ["Bosnia and Herzegovina", "Bosnia & Herzegovina", "Bosnia-Herzegovina", "Bosnia Herzegovina", "Bosnia"],
    "Mexico": ["Mexico", "México"],
    "South Africa": ["South Africa"],
    "Korea Republic": ["Korea Republic", "South Korea"],
    "Czechia": ["Czechia", "Czech Republic"],
    "USA": ["USA", "United States", "United States of America"],
    "Türkiye": ["Türkiye", "Turkey", "Turkiye"],
    "Côte d’Ivoire": ["Côte d’Ivoire", "Cote d'Ivoire", "Ivory Coast"],
    "Curaçao": ["Curaçao", "Curacao"],
    "IR Iran": ["IR Iran", "Iran"],
    "Cabo Verde": ["Cabo Verde", "Cape Verde"],
    "DR Congo": ["DR Congo", "Congo DR", "Democratic Republic of the Congo"],
}


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


def load_worldcup_groups(path: Path) -> dict[str, list[dict[str, Any]]]:
    groups = _load_yaml_file(path).get("groups")
    return groups if isinstance(groups, dict) and groups else DEFAULT_WORLD_CUP_2026_GROUPS


def load_team_aliases(path: Path) -> dict[str, str]:
    data = _load_yaml_file(path)
    aliases_source = data.get("aliases") if isinstance(data.get("aliases"), dict) else DEFAULT_TEAM_ALIASES
    aliases: dict[str, str] = {}
    for canonical, values in aliases_source.items():
        aliases[str(canonical)] = str(canonical)
        for value in values or []:
            aliases[str(value)] = str(canonical)
    return aliases


def team_label(team: dict[str, Any]) -> str:
    return f"{team.get('display_name', team.get('canonical_name'))} / {team.get('canonical_name')}"


def team_from_label(teams: list[dict[str, Any]], label: str) -> dict[str, Any]:
    for team in teams:
        if team_label(team) == label:
            return team
    return teams[0] if teams else {"display_name": "", "canonical_name": "", "api_names": []}


def api_match_name(team: dict[str, Any]) -> str:
    api_names = team.get("api_names") or []
    return str(api_names[0] if api_names else team.get("canonical_name", ""))
