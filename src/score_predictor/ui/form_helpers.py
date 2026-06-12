from __future__ import annotations

from copy import deepcopy
import math
from typing import Any


COMMON_CORRECT_SCORES = [
    "0-0",
    "1-0",
    "0-1",
    "1-1",
    "2-0",
    "0-2",
    "2-1",
    "1-2",
    "2-2",
    "3-0",
    "0-3",
    "3-1",
    "1-3",
    "3-2",
    "2-3",
    "4-0",
    "0-4",
]


def default_form_state() -> dict[str, Any]:
    return {
        "match_id": "",
        "date": "2026-06-10 20:00",
        "home_team": "Home FC",
        "away_team": "Away FC",
        "competition": "World Cup",
        "stage": "Group stage",
        "neutral_site": False,
        "odds_source": "manual",
        "snapshot_time": "T-24h",
        "timezone": "Asia/Shanghai",
        "odds_home_win": 1.85,
        "odds_draw": 3.45,
        "odds_away_win": 4.40,
        "ou_rows": [
            {"line": 1.5, "over_odds": 1.35, "under_odds": 3.05},
            {"line": 2.5, "over_odds": 1.92, "under_odds": 1.92},
            {"line": 3.5, "over_odds": 3.10, "under_odds": 1.34},
            {"line": 4.5, "over_odds": 5.50, "under_odds": 1.12},
        ],
        "btts_yes_odds": 1.70,
        "btts_no_odds": 2.15,
        "correct_score_rows": [
            {"score": score, "odds": None} for score in COMMON_CORRECT_SCORES
        ],
        "home_other": None,
        "draw_other": None,
        "away_other": None,
        "asian_handicap_line": None,
        "asian_handicap_home_odds": None,
        "asian_handicap_away_odds": None,
        "rqspf_handicap": None,
        "rqspf_home_odds": None,
        "rqspf_draw_odds": None,
        "rqspf_away_odds": None,
        "home_elo": None,
        "away_elo": None,
        "home_fifa_rank": None,
        "away_fifa_rank": None,
        "home_rest_days": None,
        "away_rest_days": None,
        "home_key_players_missing": "",
        "away_key_players_missing": "",
        "home_lineup_strength": None,
        "away_lineup_strength": None,
        "weather_note": "",
        "injury_notes": "",
        "lineup_notes": "",
        "schedule_notes": "",
        "motivation_notes": "",
        "source_notes": "",
        "dc_enabled": True,
        "max_goals": 8,
        "market_weight": 1.0,
        "h2h_weight": 1.0,
        "totals_weight": 1.0,
        "alternate_totals_weight": 0.8,
        "correct_score_weight": 0.35,
        "btts_weight": 0.6,
        "spreads_weight": 0.5,
        "ou_weight": 1.0,
        "x1x2_weight": 1.0,
        "sporttery_1x2_weight": 0.15,
        "sporttery_handicap_3way_weight": 0.15,
        "sporttery_total_goals_weight": 0.30,
        "sporttery_correct_score_weight": 0.20,
        "sporttery_half_full_weight": 0.0,
        "team_adjustment_strength": 1.0,
        "market_only_mode": True,
        "internal_lambda_home": None,
        "internal_lambda_away": None,
        "calibration_sources": "international\npinnacle\nbetfair\nbet365",
        "value_comparison_sources": "sporttery",
    }


def copy_default_form_state() -> dict[str, Any]:
    return deepcopy(default_form_state())


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def optional_int(value: Any) -> int | None:
    number = optional_float(value)
    if number is None:
        return None
    return int(number)


def split_text_items(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).replace("\n", ",")
    return [item.strip() for item in text.split(",") if item.strip()]


def rows_from_table(value: Any) -> list[dict[str, Any]]:
    if hasattr(value, "to_dict"):
        return list(value.to_dict(orient="records"))
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, dict)]
    return []


def build_input_warnings(state: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for field in ("home_team", "away_team", "date"):
        if not str(state.get(field) or "").strip():
            warnings.append(f"missing_{field}")
    if optional_float(state.get("odds_home_win")) is None:
        warnings.append("missing_1x2_home_odds")
    if optional_float(state.get("odds_draw")) is None:
        warnings.append("missing_1x2_draw_odds")
    if optional_float(state.get("odds_away_win")) is None:
        warnings.append("missing_1x2_away_odds")
    if not rows_from_table(state.get("ou_rows")):
        warnings.append("missing_over_under_rows")
    if state.get("market_only_mode") is False:
        if optional_float(state.get("internal_lambda_home")) is None:
            warnings.append("missing_internal_lambda_home")
        if optional_float(state.get("internal_lambda_away")) is None:
            warnings.append("missing_internal_lambda_away")
    return warnings


def build_margin_warnings(state: dict[str, Any], threshold: float = 0.12) -> list[str]:
    warnings: list[str] = []
    home = optional_float(state.get("odds_home_win"))
    draw = optional_float(state.get("odds_draw"))
    away = optional_float(state.get("odds_away_win"))
    if home and draw and away:
        margin = 1.0 / home + 1.0 / draw + 1.0 / away - 1.0
        if margin > threshold:
            warnings.append(f"high_1x2_margin:{margin:.3f}")

    for row in rows_from_table(state.get("ou_rows")):
        line = optional_float(row.get("line"))
        over = optional_float(row.get("over_odds"))
        under = optional_float(row.get("under_odds"))
        if line is not None and over and under:
            margin = 1.0 / over + 1.0 / under - 1.0
            if margin > threshold:
                warnings.append(f"high_ou_{line:g}_margin:{margin:.3f}")

    yes = optional_float(state.get("btts_yes_odds"))
    no = optional_float(state.get("btts_no_odds"))
    if yes and no:
        margin = 1.0 / yes + 1.0 / no - 1.0
        if margin > threshold:
            warnings.append(f"high_btts_margin:{margin:.3f}")
    return warnings


def audit_only_score_warning(state: dict[str, Any]) -> list[str]:
    if any(
        optional_float(state.get(field)) is not None
        for field in ("home_other", "draw_other", "away_other")
    ):
        return ["other_score_odds_are_audit_only_unless_supported_by_calibration"]
    return []
