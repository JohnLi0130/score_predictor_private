from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, validator


Position = Literal["GK", "CB", "FB", "DM", "CM", "AM", "W", "FW", "UNKNOWN"]
RoleImportance = Literal["key", "starter", "rotation", "bench", "unknown"]
MatchType = Literal[
    "world_cup",
    "continental_cup",
    "qualifier",
    "nations_league",
    "friendly",
    "club_friendly",
    "unknown",
]


class PlayerInfo(BaseModel):
    name: str
    position: Position = "UNKNOWN"
    is_regular_starter: bool = False
    role_importance: RoleImportance = "unknown"

    @validator("name")
    def name_must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Player name must not be blank.")
        return value.strip()


class LineupInfo(BaseModel):
    formation: str | None = None
    confirmed: bool = False
    starters: list[PlayerInfo] = Field(default_factory=list)
    bench: list[PlayerInfo] = Field(default_factory=list)


class TeamSquadInfo(BaseModel):
    source_url: str | None = None
    confirmed: bool = False
    key_absent_players: list[str] = Field(default_factory=list)
    notes: str | None = None


class InjurySuspensionInfo(BaseModel):
    absent: list[str] = Field(default_factory=list)
    doubtful: list[str] = Field(default_factory=list)
    suspended: list[str] = Field(default_factory=list)


class WeatherInfo(BaseModel):
    temperature_c: float | None = None
    humidity_pct: float | None = None
    rain_probability_pct: float | None = None
    wind_kph: float | None = None


class NarrativeFlags(BaseModel):
    coach_debut: bool = False
    player_milestone: bool = False
    revenge_talk: bool = False
    public_hype_home: bool = False
    public_hype_away: bool = False
    ceremonial_match: bool = False


class IntelligenceInput(BaseModel):
    source_mode: Literal["manual", "web_stub", "mixed"] = "manual"
    official_squads_available: bool = False
    official_lineups_available: bool = False
    match_type: MatchType = "unknown"
    match_importance: dict[str, str] = Field(default_factory=dict)
    squad: dict[str, TeamSquadInfo] = Field(default_factory=dict)
    lineups: dict[str, LineupInfo] = Field(default_factory=dict)
    injuries_suspensions: dict[str, InjurySuspensionInfo] = Field(default_factory=dict)
    rest_days: dict[str, int | None] = Field(default_factory=dict)
    travel: dict[str, float | None] = Field(default_factory=dict)
    weather: WeatherInfo | None = None
    tactical_notes: dict[str, str] = Field(default_factory=dict)
    narrative_flags: NarrativeFlags = Field(default_factory=NarrativeFlags)
    conflicting_information: bool = False
    sources: list[dict] = Field(default_factory=list)
    excluded_sources: list[dict] = Field(default_factory=list)
    source_quality: dict[str, object] = Field(default_factory=dict)
    model_adjustment_policy: dict[str, object] = Field(default_factory=dict)
    prematch_context: dict[str, object] = Field(default_factory=dict)
    prematch_audit_notes: list[str] = Field(default_factory=list)
    prematch_warnings: list[str] = Field(default_factory=list)
