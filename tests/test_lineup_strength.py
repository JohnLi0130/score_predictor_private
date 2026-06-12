from __future__ import annotations

from score_predictor.intelligence.lineup_strength import compute_lineup_strength
from score_predictor.intelligence.schemas import InjurySuspensionInfo, LineupInfo, PlayerInfo


def _full_lineup() -> LineupInfo:
    return LineupInfo(
        formation="4-3-3",
        confirmed=True,
        starters=[
            PlayerInfo(name="GK", position="GK", is_regular_starter=True, role_importance="key"),
            PlayerInfo(name="CB1", position="CB", is_regular_starter=True, role_importance="starter"),
            PlayerInfo(name="CB2", position="CB", is_regular_starter=True, role_importance="starter"),
            PlayerInfo(name="DM", position="DM", is_regular_starter=True, role_importance="starter"),
            PlayerInfo(name="CM", position="CM", is_regular_starter=True, role_importance="starter"),
            PlayerInfo(name="AM", position="AM", is_regular_starter=True, role_importance="key"),
            PlayerInfo(name="W", position="W", is_regular_starter=True, role_importance="starter"),
            PlayerInfo(name="FW", position="FW", is_regular_starter=True, role_importance="key"),
        ],
    )


def test_full_strength_lineup_returns_high() -> None:
    result = compute_lineup_strength(_full_lineup())
    assert result["level"] == "high"
    assert result["score"] >= 80


def test_no_lineup_returns_unknown_warning() -> None:
    result = compute_lineup_strength(None)
    assert result["level"] == "unknown"
    assert "lineup_not_confirmed" in result["warnings"]


def test_key_striker_absent_lowers_score() -> None:
    base = compute_lineup_strength(_full_lineup())
    injured = compute_lineup_strength(
        _full_lineup(),
        InjurySuspensionInfo(absent=["key striker"]),
    )
    assert injured["score"] < base["score"]
    assert "key_striker_absent" in injured["warnings"]


def test_five_back_friendly_adds_warning() -> None:
    lineup = _full_lineup()
    lineup.formation = "5-3-2"
    result = compute_lineup_strength(lineup, match_type="friendly")
    assert "defensive_five_back_in_friendly" in result["warnings"]

