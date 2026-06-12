from __future__ import annotations

import math
import re
from typing import Any

from score_predictor.odds import implied_prob, normalize_probs

SPORTTERY_OTHER_KEYS = {"home_other", "draw_other", "away_other"}
TOTAL_GOALS_BUCKETS = ("0", "1", "2", "3", "4", "5", "6", "7+")


def clamp_weight_score(value: float) -> float:
    return max(0.25, min(1.0, float(value)))


def score_key(value: Any) -> str | None:
    text = str(value).strip()
    match = re.fullmatch(r"(\d{1,2})\s*[-:]\s*(\d{1,2})", text)
    if not match:
        return None
    return f"{int(match.group(1))}-{int(match.group(2))}"


def odds_mapping_from_market(market: Any) -> dict[str, float]:
    if not isinstance(market, dict):
        return {}
    if isinstance(market.get("scores"), dict):
        return {
            str(outcome): float(odds)
            for outcome, odds in market["scores"].items()
            if odds is not None
        }
    if isinstance(market.get("odds"), dict):
        return {
            str(outcome): float(odds)
            for outcome, odds in market["odds"].items()
            if odds is not None
        }
    ignored = {
        "type",
        "market",
        "source",
        "provider",
        "bookmaker",
        "last_update",
        "snapshot_time",
        "history",
        "note",
        "line",
        "handicap",
        "weight",
    }
    result: dict[str, float] = {}
    for outcome, odds in market.items():
        if outcome in ignored or odds is None:
            continue
        try:
            result[str(outcome)] = float(odds)
        except (TypeError, ValueError):
            continue
    return result


def devig_probabilities(odds: dict[str, float]) -> dict[str, float]:
    valid = {
        str(outcome): implied_prob(float(price))
        for outcome, price in odds.items()
        if price is not None and float(price) > 1.0
    }
    if not valid:
        return {}
    return normalize_probs(valid)


def overround_metrics(odds: dict[str, float]) -> dict[str, float | None]:
    valid = [float(price) for price in odds.values() if price is not None and float(price) > 1.0]
    if not valid:
        return {"raw_prob_sum": None, "payout_rate": None, "overround": None}
    raw_prob_sum = sum(1.0 / price for price in valid)
    payout_rate = 1.0 / raw_prob_sum if raw_prob_sum > 0 else None
    return {
        "raw_prob_sum": raw_prob_sum,
        "payout_rate": payout_rate,
        "overround": raw_prob_sum - 1.0,
    }


def split_correct_score_market(
    market: Any,
) -> tuple[dict[str, float], dict[str, float]]:
    odds = odds_mapping_from_market(market)
    scores: dict[str, float] = {}
    other: dict[str, float] = {}
    for outcome, price in odds.items():
        normalized_score = score_key(outcome)
        if normalized_score is not None:
            scores[normalized_score] = float(price)
        elif outcome in SPORTTERY_OTHER_KEYS:
            other[outcome] = float(price)
    return scores, other


def normalize_total_goals_market(market: Any) -> dict[str, float]:
    odds = odds_mapping_from_market(market)
    normalized: dict[str, float] = {}
    for outcome, price in odds.items():
        key = str(outcome).strip()
        if key == "7" and "7+" not in odds:
            key = "7+"
        if key in TOTAL_GOALS_BUCKETS:
            normalized[key] = float(price)
    return normalized


def score_market_quality(market: Any) -> dict[str, Any]:
    market_data = dict(market) if isinstance(market, dict) else {}
    market_type = str(market_data.get("type") or market_data.get("market") or "generic")
    odds = odds_mapping_from_market(market_data)
    metrics = overround_metrics(odds)
    warnings: list[str] = []
    drivers: list[str] = []
    score = 1.0

    if not odds:
        return {
            "score": 0.25,
            "level": "low",
            "drivers": [],
            "warnings": ["market_missing_or_empty"],
            "raw_prob_sum": None,
            "payout_rate": None,
            "overround": None,
        }

    payout_rate = metrics["payout_rate"]
    if payout_rate is not None:
        if payout_rate < 0.65:
            score *= 0.35
            warnings.append("sporttery_market_low_payout_rate")
        elif payout_rate < 0.75:
            score *= 0.55
            warnings.append("sporttery_market_low_payout_rate")
        elif payout_rate < 0.85:
            score *= 0.80
            warnings.append("market_payout_rate_moderate")
        else:
            drivers.append("payout_rate_reasonable")

    if market_type == "correct_score":
        listed_scores, other = split_correct_score_market(market_data)
        listed_count = len(listed_scores)
        if listed_count < 8:
            score = min(score, 0.55)
            warnings.append("correct_score_incomplete")
        elif listed_count < 16:
            score = min(score, 0.70)
            warnings.append("correct_score_incomplete")
        else:
            drivers.append("correct_score_has_broad_score_grid")
        if other:
            warnings.append("correct_score_other_not_used")
        if not SPORTTERY_OTHER_KEYS.issubset(set(other)):
            score = min(score, 0.70)
            warnings.append("correct_score_incomplete")
    elif market_type == "total_goals":
        buckets = set(normalize_total_goals_market(market_data))
        missing = [bucket for bucket in TOTAL_GOALS_BUCKETS if bucket not in buckets]
        if missing:
            score = min(score, 0.65)
            warnings.append("total_goals_incomplete")
        else:
            drivers.append("total_goals_complete_0_to_7_plus")
    elif market_type == "half_full":
        score = min(score, 0.75)
        drivers.append("half_full_audit_only")

    score = clamp_weight_score(score)
    if score >= 0.80:
        level = "high"
    elif score >= 0.55:
        level = "medium"
    else:
        level = "low"
    return {
        "score": score,
        "level": level,
        "drivers": drivers,
        "warnings": list(dict.fromkeys(warnings)),
        **metrics,
    }


def expected_total_goals(probabilities: dict[str, float]) -> float | None:
    if not probabilities:
        return None
    expected = 0.0
    total = 0.0
    for bucket, probability in probabilities.items():
        p = float(probability)
        if bucket == "7+":
            goals = 7.25
        else:
            try:
                goals = float(bucket)
            except ValueError:
                continue
        expected += goals * p
        total += p
    if total <= 0:
        return None
    return expected / total


def _direction_from_probs(probabilities: dict[str, float], tolerance: float = 0.05) -> str:
    home = float(probabilities.get("home", 0.0))
    away = float(probabilities.get("away", 0.0))
    if abs(home - away) <= tolerance:
        return "balanced"
    return "home" if home > away else "away"


def score_channel_consistency(
    international_constraints: dict[str, Any],
    sporttery_constraints: dict[str, Any],
) -> dict[str, Any]:
    conflicts: list[str] = []
    warnings: list[str] = []
    strong_conflicts = 0

    international_totals = international_constraints.get("over_under") or {}
    sporttery_total_goals = sporttery_constraints.get("total_goals") or {}
    sporttery_total_expected = expected_total_goals(sporttery_total_goals)
    primary_25 = international_totals.get("2.5") or international_totals.get("2.25")
    if primary_25 and sporttery_total_expected is not None:
        international_high = float(primary_25.get("over", 0.0)) >= 0.56
        international_low = float(primary_25.get("under", 0.0)) >= 0.56
        if international_high and sporttery_total_expected < 2.15:
            conflicts.append("international_totals_high_vs_sporttery_total_goals_low")
            strong_conflicts += 1
        elif international_low and sporttery_total_expected > 3.0:
            conflicts.append("international_totals_low_vs_sporttery_total_goals_high")
            strong_conflicts += 1

    international_1x2 = international_constraints.get("one_x_two") or {}
    sporttery_handicap = sporttery_constraints.get("handicap_3way") or {}
    if international_1x2 and sporttery_handicap:
        if (
            _direction_from_probs(international_1x2) == "home"
            and float(international_1x2.get("home", 0.0)) - float(international_1x2.get("away", 0.0)) > 0.20
            and _direction_from_probs(sporttery_handicap, tolerance=0.03) == "away"
        ):
            conflicts.append("international_h2h_home_strong_vs_sporttery_handicap_not_supporting")
            strong_conflicts += 1

    btts = international_constraints.get("btts") or {}
    correct_score = sporttery_constraints.get("correct_score") or {}
    if btts and correct_score:
        zero_side_mass = 0.0
        for score, probability in correct_score.items():
            parsed = score_key(score)
            if parsed is None:
                continue
            home_text, away_text = parsed.split("-", 1)
            if int(home_text) == 0 or int(away_text) == 0:
                zero_side_mass += float(probability)
        if float(btts.get("yes", 0.0)) >= 0.58 and zero_side_mass >= 0.45:
            conflicts.append("international_btts_yes_high_vs_sporttery_zero_score_cluster")
            strong_conflicts += 1

    sporttery_1x2 = sporttery_constraints.get("one_x_two") or {}
    if sporttery_1x2 and correct_score:
        aggregate = {"home": 0.0, "draw": 0.0, "away": 0.0}
        for score, probability in correct_score.items():
            parsed = score_key(score)
            if parsed is None:
                continue
            home_goals, away_goals = (int(value) for value in parsed.split("-", 1))
            if home_goals > away_goals:
                aggregate["home"] += float(probability)
            elif home_goals == away_goals:
                aggregate["draw"] += float(probability)
            else:
                aggregate["away"] += float(probability)
        if sum(aggregate.values()) > 0:
            normalized = normalize_probs(aggregate)
            if _direction_from_probs(sporttery_1x2) != _direction_from_probs(normalized):
                conflicts.append("sporttery_correct_score_vs_1x2_direction_conflict")

    if correct_score and sporttery_total_expected is not None:
        correct_total_expected = 0.0
        correct_total_mass = 0.0
        for score, probability in correct_score.items():
            parsed = score_key(score)
            if parsed is None:
                continue
            correct_total_expected += sum(int(part) for part in parsed.split("-", 1)) * float(probability)
            correct_total_mass += float(probability)
        if correct_total_mass > 0:
            correct_total_expected /= correct_total_mass
        else:
            correct_total_expected = None
        if correct_total_expected is not None and abs(correct_total_expected - sporttery_total_expected) > 0.8:
            conflicts.append("sporttery_total_goals_vs_correct_score_total_conflict")

    if strong_conflicts:
        score = 0.25
        level = "strong_conflict"
        warnings.append("odds_channel_conflict")
    elif conflicts:
        score = 0.70
        level = "mild_conflict"
        warnings.append("odds_channel_mild_conflict")
    else:
        score = 1.0
        level = "aligned"

    return {
        "score": score,
        "level": level,
        "conflicts": conflicts,
        "warnings": warnings,
    }


def final_weight(base_weight: float, quality: dict[str, Any], consistency: dict[str, Any]) -> float:
    return float(base_weight) * clamp_weight_score(float(quality.get("score", 0.25))) * clamp_weight_score(
        float(consistency.get("score", 0.25))
    )


def build_constraints_from_channels(
    international_channel: dict[str, Any],
    sporttery_channel: dict[str, Any],
    settings: Any,
) -> dict[str, Any]:
    consistency = score_channel_consistency(international_channel, sporttery_channel)
    qualities = {}
    for channel_name, channel in (
        ("international", international_channel),
        ("sporttery", sporttery_channel),
    ):
        for market_name, market in (channel.get("raw_markets") or {}).items():
            qualities[f"{channel_name}.{market_name}"] = score_market_quality(market)
    return {
        "international": international_channel,
        "sporttery": sporttery_channel,
        "market_quality": qualities,
        "channel_consistency": consistency,
        "settings": settings,
    }
