from __future__ import annotations

import re

from .implied import build_market_probability_table


def _sort_outcomes_by_fair_prob(table: dict, limit: int) -> list[dict]:
    rows = [
        {
            "outcome": outcome,
            "odds": data["odds"],
            "fair_prob": data["fair_prob"],
            "fair_prob_pct": data["fair_prob_pct"],
        }
        for outcome, data in table["outcomes"].items()
    ]
    return sorted(rows, key=lambda row: row["fair_prob"], reverse=True)[:limit]


def _total_goal_value(outcome: str) -> float | None:
    normalized = outcome.strip()
    if normalized.endswith("+"):
        try:
            return float(normalized[:-1])
        except ValueError:
            return None
    try:
        return float(normalized)
    except ValueError:
        return None


def analyze_total_goals_odds(total_goals_odds: dict[str, float]) -> dict:
    table = build_market_probability_table(
        {str(key): float(value) for key, value in total_goals_odds.items()}
    )
    warnings: list[str] = []
    expected_total = 0.0
    included_mass = 0.0
    for outcome, fair_prob in table["fair_probs"].items():
        goal_value = _total_goal_value(outcome)
        if goal_value is None:
            warnings.append(f"total_goals_outcome_unparsed:{outcome}")
            continue
        if outcome.endswith("+"):
            warnings.append("total_goals_plus_bucket_capped_for_mean")
        expected_total += goal_value * fair_prob
        included_mass += fair_prob

    if included_mass <= 0:
        expected_total = 0.0

    top_total_goals = _sort_outcomes_by_fair_prob(table, limit=3)
    most_likely = top_total_goals[0]["outcome"] if top_total_goals else None
    return {
        "raw_probs": table["raw_probs"],
        "fair_probs": table["fair_probs"],
        "overround": table["overround"],
        "payout_rate": table["payout_rate"],
        "most_likely_total_goals": most_likely,
        "top_total_goals": top_total_goals,
        "expected_total_goals_from_distribution": expected_total,
        "warnings": list(dict.fromkeys(warnings)),
    }


def _parse_score(outcome: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", outcome)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def analyze_correct_score_odds(correct_score_odds: dict[str, float]) -> dict:
    table = build_market_probability_table(
        {str(key): float(value) for key, value in correct_score_odds.items()}
    )
    warnings: list[str] = []
    parsed_scores: dict[str, tuple[int, int]] = {}
    for outcome in table["fair_probs"]:
        parsed = _parse_score(outcome)
        if parsed is None:
            warnings.append(f"correct_score_outcome_unparsed:{outcome}")
            continue
        parsed_scores[outcome] = parsed

    if len(parsed_scores) < 20:
        warnings.append("correct_score_odds_incomplete")

    implied_home = 0.0
    implied_away = 0.0
    included_mass = 0.0
    for outcome, (home_goals, away_goals) in parsed_scores.items():
        fair_prob = table["fair_probs"][outcome]
        implied_home += home_goals * fair_prob
        implied_away += away_goals * fair_prob
        included_mass += fair_prob

    if included_mass > 0:
        implied_home /= included_mass
        implied_away /= included_mass

    return {
        "raw_probs": table["raw_probs"],
        "fair_probs": table["fair_probs"],
        "overround": table["overround"],
        "payout_rate": table["payout_rate"],
        "top_scores_by_market": _sort_outcomes_by_fair_prob(table, limit=5),
        "implied_home_goal_mean": implied_home,
        "implied_away_goal_mean": implied_away,
        "implied_total_goal_mean": implied_home + implied_away,
        "warnings": list(dict.fromkeys(warnings)),
    }

