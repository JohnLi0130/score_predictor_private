from __future__ import annotations

from copy import deepcopy
from typing import Any

import yaml

from .schemas import IntelligenceInput


SCHEMA_VERSION = "prematch_context_v1"

SUBJECTIVE_PATTERNS = (
    "专家推荐",
    "投注建议",
    "盘口解读",
    "我觉得",
    "看好",
    "稳赚",
    "稳赢",
    "稳胆",
    "稳",
    "方向",
    "应该怎么选",
    "推荐下注",
    "必买",
    "包中",
    "expert pick",
    "betting advice",
    "market read",
    "i think",
)

MATCH_TYPE_BY_STAGE = {
    "group": "world_cup",
    "knockout": "world_cup",
    "qualifier": "qualifier",
    "friendly": "friendly",
}


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _join(items: list[Any]) -> str:
    return "\n".join(_text(item) for item in items if _text(item))


def _contains_subjective(value: Any) -> bool:
    normalized = _text(value).casefold()
    return any(pattern.casefold() in normalized for pattern in SUBJECTIVE_PATTERNS)


def _safe_list(value: Any, path: str, audit_notes: list[str]) -> list[str]:
    safe: list[str] = []
    for item in _as_list(value):
        text = _text(item)
        if not text:
            continue
        if _contains_subjective(text):
            audit_notes.append(f"{path}: {text}")
            continue
        safe.append(text)
    return safe


def _scan_subjective(value: Any, path: str, audit_notes: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _scan_subjective(child, f"{path}.{key}" if path else str(key), audit_notes)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_subjective(child, f"{path}[{index}]", audit_notes)
    elif _contains_subjective(value):
        audit_notes.append(f"{path}: {_text(value)}")


def is_prematch_context_payload(data: dict[str, Any]) -> bool:
    return _text(data.get("schema_version")) == SCHEMA_VERSION


def load_prematch_context(content: bytes | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(content, dict):
        data = deepcopy(content)
    else:
        text = content.decode("utf-8") if isinstance(content, bytes) else content
        data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError("Prematch context YAML must contain a mapping/object.")
    if not is_prematch_context_payload(data):
        raise ValueError("schema_version must be prematch_context_v1.")
    return data


def parse_prematch_context(content: bytes | str | dict[str, Any]) -> dict[str, Any]:
    data = load_prematch_context(content)
    audit_notes: list[str] = []
    _scan_subjective(data, "", audit_notes)

    home = _as_mapping(data.get("home_context"))
    away = _as_mapping(data.get("away_context"))
    match = _as_mapping(data.get("match"))
    match_context = _as_mapping(data.get("match_context"))
    source_quality = _as_mapping(data.get("source_quality"))
    policy = _as_mapping(data.get("model_adjustment_policy"))

    safe_home_injuries = _safe_list(home.get("injuries"), "home_context.injuries", audit_notes)
    safe_away_injuries = _safe_list(away.get("injuries"), "away_context.injuries", audit_notes)
    safe_home_lineup = _safe_list(
        home.get("lineup_notes"), "home_context.lineup_notes", audit_notes
    )
    safe_away_lineup = _safe_list(
        away.get("lineup_notes"), "away_context.lineup_notes", audit_notes
    )
    safe_home_strengths = _safe_list(
        home.get("strengths"), "home_context.strengths", audit_notes
    )
    safe_away_strengths = _safe_list(
        away.get("strengths"), "away_context.strengths", audit_notes
    )
    safe_match_risks = _safe_list(
        match_context.get("risk_notes"), "match_context.risk_notes", audit_notes
    )

    parsed = deepcopy(data)
    parsed["_audit"] = {
        "subjective_detected": bool(audit_notes),
        "audit_notes": list(dict.fromkeys(audit_notes)),
        "warnings": (
            ["prematch_context_subjective_content_audit_only"] if audit_notes else []
        ),
    }
    parsed["_safe"] = {
        "home_injuries": safe_home_injuries,
        "away_injuries": safe_away_injuries,
        "home_lineup_notes": safe_home_lineup,
        "away_lineup_notes": safe_away_lineup,
        "home_strengths": safe_home_strengths,
        "away_strengths": safe_away_strengths,
        "risk_notes": safe_match_risks,
        "source_quality": source_quality,
        "policy": policy,
        "match": match,
    }
    return parsed


def _stage_to_match_type(stage: Any) -> str:
    stage_text = _text(stage).casefold()
    for keyword, match_type in MATCH_TYPE_BY_STAGE.items():
        if keyword in stage_text:
            return match_type
    return "world_cup" if stage_text else "unknown"


def _source_multiplier(confidence: str) -> float:
    normalized = confidence.strip().lower()
    if normalized == "low":
        return 0.5
    if normalized == "medium":
        return 0.75
    return 1.0


def prematch_context_to_intelligence(content: bytes | str | dict[str, Any]) -> IntelligenceInput:
    parsed = parse_prematch_context(content)
    safe = parsed["_safe"]
    match = safe["match"]
    source_quality = safe["source_quality"]
    policy = safe["policy"]
    overall_confidence = _text(source_quality.get("overall_confidence") or "medium").lower()
    source_type = _text(source_quality.get("source_type") or "prematch_context_yaml")
    source_multiplier = _source_multiplier(overall_confidence)

    home_injuries = safe["home_injuries"]
    away_injuries = safe["away_injuries"]
    home_lineup_notes = safe["home_lineup_notes"]
    away_lineup_notes = safe["away_lineup_notes"]

    return IntelligenceInput(
        source_mode="mixed",
        official_squads_available=not bool(
            source_quality.get("requires_official_confirmation", False)
        ),
        official_lineups_available=any(
            "official" in note.casefold() or "首发" in note
            for note in home_lineup_notes + away_lineup_notes
        ),
        match_type=_stage_to_match_type(match.get("stage")),
        match_importance={
            "home": _text(_as_mapping(parsed.get("match_context")).get("motivation")),
            "away": _text(_as_mapping(parsed.get("match_context")).get("motivation")),
        },
        injuries_suspensions={
            "home": {"absent": home_injuries, "doubtful": [], "suspended": []},
            "away": {"absent": away_injuries, "doubtful": [], "suspended": []},
        },
        tactical_notes={
            "home_lineup_notes": _join(home_lineup_notes),
            "away_lineup_notes": _join(away_lineup_notes),
            "risk_notes": _join(safe["risk_notes"]),
        },
        sources=[{"source_type": source_type, "overall_confidence": overall_confidence}],
        excluded_sources=[
            {"reason": "subjective_content_audit_only", "note": note}
            for note in parsed["_audit"]["audit_notes"]
        ],
        source_quality={
            **source_quality,
            "source_strength_multiplier": source_multiplier,
        },
        model_adjustment_policy={
            "enabled": bool(policy.get("enabled", True)),
            "adjustment_type": policy.get(
                "adjustment_type", "bounded_multiplicative"
            ),
            "max_single_fact_adjustment": float(
                policy.get("max_single_fact_adjustment", 0.05)
            ),
            "max_major_fact_adjustment": float(
                policy.get("max_major_fact_adjustment", 0.12)
            ),
            "max_total_adjustment": float(policy.get("max_total_adjustment", 0.15)),
            "do_not_override_market": bool(policy.get("do_not_override_market", True)),
            "confidence": policy.get("confidence", overall_confidence),
            "source_strength_multiplier": source_multiplier,
        },
        prematch_context={
            "schema_version": SCHEMA_VERSION,
            "match_id": parsed.get("match_id"),
            "audit": parsed["_audit"],
        },
        prematch_audit_notes=parsed["_audit"]["audit_notes"],
        prematch_warnings=parsed["_audit"]["warnings"],
    )


def prematch_context_to_form_state(content: bytes | str | dict[str, Any]) -> dict[str, Any]:
    parsed = parse_prematch_context(content)
    safe = parsed["_safe"]
    match = safe["match"]
    home = _as_mapping(parsed.get("home_context"))
    away = _as_mapping(parsed.get("away_context"))
    venue = _as_mapping(match.get("venue"))
    match_context = _as_mapping(parsed.get("match_context"))
    source_quality = safe["source_quality"]
    policy = safe["policy"]
    audit = parsed["_audit"]

    weather_note = _join(
        [
            _text(_as_mapping(match_context.get("weather")).get("summary")),
            _text(_as_mapping(parsed.get("weather")).get("summary")),
            _text(venue.get("altitude_type")),
        ]
    )
    if venue.get("neutral_site") is not None:
        neutral_text = "neutral_site=true" if venue.get("neutral_site") else "neutral_site=false"
        weather_note = _join([weather_note, neutral_text])

    return {
        "match_id": parsed.get("match_id") or "",
        "home_team": match.get("home_team", ""),
        "away_team": match.get("away_team", ""),
        "competition": match.get("competition", ""),
        "stage": match.get("stage", ""),
        "date": match.get("kickoff_time_local", ""),
        "neutral_site": bool(venue.get("neutral_site", False)),
        "home_fifa_rank": home.get("fifa_rank"),
        "away_fifa_rank": away.get("fifa_rank"),
        "home_elo": home.get("elo"),
        "away_elo": away.get("elo"),
        "home_key_players_missing": _join(safe["home_injuries"]),
        "away_key_players_missing": _join(safe["away_injuries"]),
        "home_lineup_strength": _join(safe["home_strengths"]),
        "away_lineup_strength": _join(safe["away_strengths"]),
        "weather_note": weather_note,
        "injury_notes": _join(
            [
                "home: " + _join(safe["home_injuries"])
                if safe["home_injuries"]
                else "",
                "away: " + _join(safe["away_injuries"])
                if safe["away_injuries"]
                else "",
            ]
        ),
        "lineup_notes": _join(safe["home_lineup_notes"] + safe["away_lineup_notes"]),
        "schedule_notes": _join(
            [
                _text(match_context.get("schedule")),
                _text(match_context.get("rest_days")),
            ]
        ),
        "motivation_notes": _text(match_context.get("motivation")),
        "source_notes": _join(
            [
                _text(source_quality.get("source_type")),
                f"overall_confidence={_text(source_quality.get('overall_confidence'))}",
                "requires_official_confirmation="
                + str(bool(source_quality.get("requires_official_confirmation", False))),
            ]
        ),
        "prematch_context": parsed,
        "prematch_source_type": source_quality.get("source_type", ""),
        "prematch_overall_confidence": source_quality.get("overall_confidence", ""),
        "prematch_requires_official_confirmation": bool(
            source_quality.get("requires_official_confirmation", False)
        ),
        "prematch_max_total_adjustment": float(policy.get("max_total_adjustment", 0.15)),
        "prematch_subjective_detected": audit["subjective_detected"],
        "prematch_adjustment_enabled": bool(policy.get("enabled", True)),
        "prematch_audit_notes": "\n".join(audit["audit_notes"]),
    }
