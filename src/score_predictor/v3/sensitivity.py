from __future__ import annotations

from itertools import product
from typing import Any

from .market_calibration import LAMBDA_BOUNDS, RHO_BOUNDS
from .score_calibration import calibrated_score_matrix, result_direction, summarize_score_matrix


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))


def _range(values: list[float]) -> dict[str, float]:
    return {"min": float(min(values)), "max": float(max(values))}


def run_sensitivity_analysis(
    lambda_home: float,
    lambda_away: float,
    rho: float = 0.0,
    dc_enabled: bool = False,
    max_goals: int = 10,
    lambda_pct_delta: float = 0.05,
    rho_delta: float = 0.05,
) -> dict[str, Any]:
    home_values = [
        _clamp(lambda_home * (1.0 - lambda_pct_delta), LAMBDA_BOUNDS),
        _clamp(lambda_home, LAMBDA_BOUNDS),
        _clamp(lambda_home * (1.0 + lambda_pct_delta), LAMBDA_BOUNDS),
    ]
    away_values = [
        _clamp(lambda_away * (1.0 - lambda_pct_delta), LAMBDA_BOUNDS),
        _clamp(lambda_away, LAMBDA_BOUNDS),
        _clamp(lambda_away * (1.0 + lambda_pct_delta), LAMBDA_BOUNDS),
    ]
    rho_values = [rho]
    if dc_enabled:
        rho_values = [
            _clamp(rho - rho_delta, RHO_BOUNDS),
            _clamp(rho, RHO_BOUNDS),
            _clamp(rho + rho_delta, RHO_BOUNDS),
        ]

    scenarios = []
    for home_value, away_value, rho_value in product(home_values, away_values, rho_values):
        matrix = calibrated_score_matrix(
            home_value,
            away_value,
            rho=rho_value,
            dc_enabled=dc_enabled,
            max_goals=max_goals,
        )
        summary = summarize_score_matrix(matrix, over_under_lines=(2.5,), top_n=5)
        scenarios.append(
            {
                "lambda_home": home_value,
                "lambda_away": away_value,
                "rho": rho_value,
                "summary": summary,
            }
        )

    base_matrix = calibrated_score_matrix(
        lambda_home,
        lambda_away,
        rho=rho,
        dc_enabled=dc_enabled,
        max_goals=max_goals,
    )
    base_summary = summarize_score_matrix(base_matrix, over_under_lines=(2.5,), top_n=5)
    base_top_scores = [row["score"] for row in base_summary["top_scores"]]
    base_top_score = base_top_scores[0] if base_top_scores else None
    base_direction = result_direction(base_summary["one_x_two"])

    home_probs = [scenario["summary"]["one_x_two"]["home"] for scenario in scenarios]
    draw_probs = [scenario["summary"]["one_x_two"]["draw"] for scenario in scenarios]
    away_probs = [scenario["summary"]["one_x_two"]["away"] for scenario in scenarios]
    over_25_probs = [
        scenario["summary"]["over_under"]["2.5"]["over"]
        for scenario in scenarios
    ]
    top_score_changed = any(
        scenario["summary"]["top_scores"][0]["score"] != base_top_score
        for scenario in scenarios
        if scenario["summary"]["top_scores"]
    )
    result_direction_changed = any(
        result_direction(scenario["summary"]["one_x_two"]) != base_direction
        for scenario in scenarios
    )
    top_5_sets = [
        {row["score"] for row in scenario["summary"]["top_scores"]}
        for scenario in scenarios
    ]
    base_top_5 = set(base_top_scores)
    overlap_values = [
        len(base_top_5.intersection(score_set)) / max(1, len(base_top_5))
        for score_set in top_5_sets
    ]
    top_5_overlap_min = min(overlap_values) if overlap_values else 1.0
    top_5_stable = top_5_overlap_min >= 0.8 and not top_score_changed

    warnings = []
    if top_score_changed:
        warnings.append("most_likely_score_sensitive_to_small_lambda_or_rho_changes")
    if result_direction_changed:
        warnings.append("result_direction_sensitive_to_small_lambda_or_rho_changes")
    if top_5_overlap_min < 0.8:
        warnings.append("top_5_scores_not_stable_under_small_perturbations")

    stability_score = 1.0
    if top_score_changed:
        stability_score -= 0.25
    if result_direction_changed:
        stability_score -= 0.30
    if top_5_overlap_min < 0.8:
        stability_score -= 0.15
    stability_score = max(0.0, min(1.0, stability_score))

    return {
        "perturbations": {
            "lambda_home_pct": lambda_pct_delta,
            "lambda_away_pct": lambda_pct_delta,
            "rho_abs": rho_delta if dc_enabled else 0.0,
        },
        "scenario_count": len(scenarios),
        "result_probability_ranges": {
            "home_win": _range(home_probs),
            "draw": _range(draw_probs),
            "away_win": _range(away_probs),
        },
        "over_2_5_probability_range": _range(over_25_probs),
        "top_5_base": base_summary["top_scores"],
        "top_5_stable": top_5_stable,
        "top_5_overlap_min": float(top_5_overlap_min),
        "most_likely_score_changed": top_score_changed,
        "result_direction_changed": result_direction_changed,
        "warnings": warnings,
        "stability_score": stability_score,
    }
