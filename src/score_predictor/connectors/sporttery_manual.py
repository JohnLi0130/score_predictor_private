from __future__ import annotations

from typing import Any

from score_predictor.market.implied import build_market_probability_table
from score_predictor.market.sporttery_features import (
    analyze_correct_score_odds,
    analyze_total_goals_odds,
)

from .base import ensure_mapping
from .normalizers import to_float_mapping


def _extract_sporttery(data: dict[str, Any]) -> dict[str, Any]:
    if "sporttery" in data and isinstance(data["sporttery"], dict):
        return data["sporttery"]
    return data


def _validate_required_market(data: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = data.get(key)
    if value is None:
        return None
    return ensure_mapping(value, key)


def normalize_sporttery_manual(data: dict) -> dict:
    sporttery = _extract_sporttery(ensure_mapping(data, "sporttery manual data"))
    warnings: list[str] = []

    spf_input = _validate_required_market(sporttery, "spf")
    rqspf_input = _validate_required_market(sporttery, "rqspf")
    total_goals_input = _validate_required_market(sporttery, "total_goals")
    correct_score_input = _validate_required_market(sporttery, "correct_score")

    if spf_input is None:
        raise ValueError("sporttery.spf is required.")

    spf = to_float_mapping(spf_input)
    rqspf: dict[str, Any] | None = None
    if rqspf_input is not None:
        rqspf = {
            "handicap": rqspf_input.get("handicap"),
            **to_float_mapping(rqspf_input, skip_keys={"handicap"}),
        }

    market: dict[str, Any] = {
        "odds_1x2": spf,
        "spf": spf,
    }
    if rqspf is not None:
        market["rqspf"] = rqspf
        market["sporttery_handicap"] = rqspf
    if total_goals_input is not None:
        market["total_goals"] = to_float_mapping(total_goals_input)
    if correct_score_input is not None:
        market["correct_score"] = to_float_mapping(correct_score_input)

    features: dict[str, Any] = {
        "spf": build_market_probability_table(spf),
    }
    if rqspf is not None:
        rqspf_odds = {
            key: value for key, value in rqspf.items() if key != "handicap"
        }
        features["rqspf"] = build_market_probability_table(rqspf_odds)
        features["rqspf"]["handicap"] = rqspf.get("handicap")
    if total_goals_input is not None:
        total_features = analyze_total_goals_odds(to_float_mapping(total_goals_input))
        features["total_goals"] = total_features
        warnings.extend(total_features.get("warnings", []))
    else:
        warnings.append("sporttery_total_goals_missing")
    if correct_score_input is not None:
        score_features = analyze_correct_score_odds(to_float_mapping(correct_score_input))
        features["correct_score"] = score_features
        warnings.extend(score_features.get("warnings", []))
    else:
        warnings.append("sporttery_correct_score_missing")

    if "asian_handicap" in market:
        warnings.append("sporttery_rqspf_must_not_be_named_asian_handicap")

    return {
        "market": market,
        "features": features,
        "warnings": list(dict.fromkeys(warnings)),
        "source": "manual_sporttery",
    }

