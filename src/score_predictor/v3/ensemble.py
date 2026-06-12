from __future__ import annotations

from typing import Any

from score_predictor.adjustments import apply_multiplicative_adjustments
from score_predictor.ensemble import blend_lambdas
from score_predictor.schemas import MatchInput

from .handicap_consistency import check_handicap_consistency
from .market_calibration import calibrate_markets
from .score_calibration import (
    DEFAULT_OU_LINES,
    calibrated_score_matrix,
    score_matrix_records,
    summarize_score_matrix,
)
from .sensitivity import run_sensitivity_analysis


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _audit_reference(audit: dict[str, Any]) -> str:
    sources = audit.get("sources") or []
    if sources:
        first = sources[0]
        return str(first.get("url") or first.get("title") or "facts_source_record")
    return str(audit.get("source_policy", "facts_only_no_prediction_articles"))


def _confidence_label(data_quality: dict[str, Any]) -> str:
    level = str(data_quality.get("level", "low"))
    if level == "high":
        return "high"
    if level == "medium":
        return "medium"
    return "low"


def _build_contribution_table(
    match_input: MatchInput,
    market_home: float,
    market_away: float,
    blended_home: float,
    blended_away: float,
    team_adjusted_home: float,
    team_adjusted_away: float,
    final_home: float,
    final_away: float,
    intel_adjustments: dict[str, Any],
    data_quality: dict[str, Any],
    audit: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audit_ref = _audit_reference(audit)
    confidence = _confidence_label(data_quality)

    if blended_home != market_home or blended_away != market_away:
        rows.append(
            {
                "factor_name": "internal_model_log_space_blend",
                "source": "manual_internal_lambda",
                "effect_on_home_lambda": blended_home - market_home,
                "effect_on_away_lambda": blended_away - market_away,
                "reason": "log_space_blending_with_manual_internal_lambda",
                "confidence": "manual",
                "audit_reference": "internal_model",
            }
        )

    total_factor = float(intel_adjustments.get("total_lambda_factor", 1.0))
    home_factor = float(intel_adjustments.get("home_lambda_factor", 1.0))
    away_factor = float(intel_adjustments.get("away_lambda_factor", 1.0))
    after_total_home = blended_home * total_factor
    after_total_away = blended_away * total_factor

    if total_factor != 1.0:
        rows.append(
            {
                "factor_name": "facts_total_lambda_factor",
                "source": "facts_only_intelligence",
                "effect_on_home_lambda": after_total_home - blended_home,
                "effect_on_away_lambda": after_total_away - blended_away,
                "reason": ",".join(intel_adjustments.get("warnings", [])) or "match_context",
                "confidence": confidence,
                "audit_reference": audit_ref,
            }
        )
    if home_factor != 1.0:
        rows.append(
            {
                "factor_name": "facts_home_lambda_factor",
                "source": "facts_only_intelligence",
                "effect_on_home_lambda": team_adjusted_home - after_total_home,
                "effect_on_away_lambda": 0.0,
                "reason": ",".join(intel_adjustments.get("drivers", [])) or "home_team_facts",
                "confidence": confidence,
                "audit_reference": audit_ref,
            }
        )
    if away_factor != 1.0:
        rows.append(
            {
                "factor_name": "facts_away_lambda_factor",
                "source": "facts_only_intelligence",
                "effect_on_home_lambda": 0.0,
                "effect_on_away_lambda": team_adjusted_away - after_total_away,
                "reason": ",".join(intel_adjustments.get("drivers", [])) or "away_team_facts",
                "confidence": confidence,
                "audit_reference": audit_ref,
            }
        )
    if not any(row["source"] == "facts_only_intelligence" for row in rows):
        rows.append(
            {
                "factor_name": "facts_only_intelligence_no_lambda_change",
                "source": "facts_only_intelligence",
                "effect_on_home_lambda": 0.0,
                "effect_on_away_lambda": 0.0,
                "reason": "no_facts_based_lambda_adjustment_applied",
                "confidence": confidence,
                "audit_reference": audit_ref,
            }
        )

    if final_home != team_adjusted_home or final_away != team_adjusted_away:
        rows.append(
            {
                "factor_name": "manual_pre_match_multiplicative_adjustments",
                "source": "manual_pre_match_adjustment",
                "effect_on_home_lambda": final_home - team_adjusted_home,
                "effect_on_away_lambda": final_away - team_adjusted_away,
                "reason": ",".join(match_input.adjustments.reasons) or "manual_adjustment",
                "confidence": "manual",
                "audit_reference": "adjustments",
            }
        )

    return rows


def _confidence_split(
    final_summary: dict[str, Any],
    consistency: dict[str, Any],
    data_quality: dict[str, Any],
    sensitivity: dict[str, Any],
) -> dict[str, float | str]:
    result_values = sorted(final_summary["one_x_two"].values(), reverse=True)
    result_margin = result_values[0] - result_values[1] if len(result_values) > 1 else 0.0
    result_confidence = _clamp01(0.35 + result_margin * 2.0)
    top_score_probability = (
        float(final_summary["top_scores"][0]["prob"])
        if final_summary.get("top_scores")
        else 0.0
    )
    score_confidence = _clamp01(top_score_probability * 6.0)
    market_consistency_score = _clamp01(float(consistency.get("score", 0.0)))
    data_quality_score = _clamp01(float(data_quality.get("score", 0.0)) / 100.0)
    sensitivity_stability_score = _clamp01(
        float(sensitivity.get("stability_score", 0.0))
    )
    final_confidence_score = _clamp01(
        0.25 * result_confidence
        + 0.25 * market_consistency_score
        + 0.25 * data_quality_score
        + 0.25 * sensitivity_stability_score
    )
    return {
        "result_confidence": result_confidence,
        "score_confidence": score_confidence,
        "market_consistency_score": market_consistency_score,
        "data_quality_score": data_quality_score,
        "sensitivity_stability_score": sensitivity_stability_score,
        "final_confidence_score": final_confidence_score,
        "note": "final_confidence_score is a model reliability score, not a hit probability.",
    }


def _market_probabilities_for_report(market_probabilities: dict[str, Any]) -> dict[str, Any]:
    return {
        "one_x_two": market_probabilities.get("one_x_two", {}),
        "one_x_two_sources": market_probabilities.get("one_x_two_sources", []),
        "over_under": market_probabilities.get("over_under", {}),
        "spreads": market_probabilities.get("spreads"),
        "btts": market_probabilities.get("btts"),
        "correct_score": market_probabilities.get("correct_score", {}),
        "correct_score_sources": market_probabilities.get("correct_score_sources", []),
        "sporttery_total_goals": market_probabilities.get("sporttery_total_goals", {}),
        "sporttery_handicap_3way": market_probabilities.get("sporttery_handicap_3way"),
    }


def _a_source_audit_markets(match_input: MatchInput) -> dict[str, Any]:
    market = match_input.calibration_market or {}
    audit_markets = market.get("audit_markets") or {}
    return {
        key: audit_markets.get(key)
        for key in ("alternate_spreads", "draw_no_bet", "team_totals", "ignored_market_keys")
        if audit_markets.get(key)
    }


def build_v3_prediction(
    match_input: MatchInput,
    market_initial_lambda: tuple[float, float],
    intel_adjustments: dict[str, Any],
    data_quality: dict[str, Any],
    audit: dict[str, Any],
    dc_enabled: bool = False,
) -> dict[str, Any]:
    fit_max_goals = max(10, int(match_input.settings.max_goals))
    calibration = calibrate_markets(
        match_input,
        initial=market_initial_lambda,
        dc_enabled=dc_enabled,
        max_goals=fit_max_goals,
    )
    market_home = float(calibration["lambda_home"])
    market_away = float(calibration["lambda_away"])
    rho = float(calibration["rho"])

    if match_input.settings.market_only_mode:
        blended_home, blended_away = market_home, market_away
    else:
        blended_home, blended_away = blend_lambdas(
            market_home,
            market_away,
            match_input.internal_model.home_lambda,
            match_input.internal_model.away_lambda,
            market_weight=float(intel_adjustments["market_weight"]),
        )
    team_adjusted_home = (
        blended_home
        * float(intel_adjustments["total_lambda_factor"])
        * float(intel_adjustments["home_lambda_factor"])
    )
    team_adjusted_away = (
        blended_away
        * float(intel_adjustments["total_lambda_factor"])
        * float(intel_adjustments["away_lambda_factor"])
    )
    final_home, final_away = apply_multiplicative_adjustments(
        team_adjusted_home,
        team_adjusted_away,
        home_factors=match_input.adjustments.home_factors,
        away_factors=match_input.adjustments.away_factors,
    )
    final_home = max(0.05, min(5.0, final_home))
    final_away = max(0.05, min(5.0, final_away))

    final_matrix = calibrated_score_matrix(
        final_home,
        final_away,
        rho=rho,
        dc_enabled=dc_enabled,
        max_goals=int(match_input.settings.max_goals),
    )
    final_summary = summarize_score_matrix(
        final_matrix,
        over_under_lines=DEFAULT_OU_LINES,
        top_n=10,
    )
    consistency = check_handicap_consistency(
        calibration["market_probabilities"]["one_x_two"],
        final_summary["one_x_two"],
        asian_handicap=match_input.asian_handicap,
        rqspf=match_input.rqspf,
    )
    channel_consistency = calibration.get("channel_consistency") or {}
    if channel_consistency:
        consistency["channel_consistency"] = channel_consistency
        consistency["score"] = min(
            float(consistency.get("score", 1.0)),
            float(channel_consistency.get("score", 1.0)),
        )
    a_source_audit_markets = _a_source_audit_markets(match_input)
    if a_source_audit_markets:
        consistency["a_source_audit_markets"] = a_source_audit_markets
    sensitivity = run_sensitivity_analysis(
        final_home,
        final_away,
        rho=rho,
        dc_enabled=dc_enabled,
        max_goals=fit_max_goals,
    )
    confidence = _confidence_split(
        final_summary,
        consistency,
        data_quality,
        sensitivity,
    )
    contribution_table = _build_contribution_table(
        match_input,
        market_home,
        market_away,
        blended_home,
        blended_away,
        team_adjusted_home,
        team_adjusted_away,
        final_home,
        final_away,
        intel_adjustments,
        data_quality,
        audit,
    )

    warnings = []
    warnings.extend(calibration.get("warnings", []))
    warnings.extend(consistency.get("warnings", []))
    warnings.extend(channel_consistency.get("warnings", []))
    warnings.extend(sensitivity.get("warnings", []))
    warnings = list(dict.fromkeys(warnings))

    return {
        "enabled": True,
        "de_vig_market_probabilities": _market_probabilities_for_report(
            calibration["market_probabilities"]
        ),
        "joint_fit": {
            "lambda_home": market_home,
            "lambda_away": market_away,
            "rho": rho,
            "dc_enabled": bool(dc_enabled),
            "loss": float(calibration["loss"]),
            "optimizer_success": bool(calibration["optimizer_success"]),
            "optimizer_message": calibration["optimizer_message"],
            "weights": calibration["market_probabilities"]["weights"],
        },
        "market_quality": calibration.get("market_quality", {}),
        "sporttery_market_status": calibration.get("sporttery_market_status", {}),
        "channel_consistency": channel_consistency,
        "market_fit_errors": calibration["market_fit_errors"],
        "correct_score_fit_error": calibration["correct_score_fit_error"],
        "correct_score_calibration": calibration["correct_score_calibration"],
        "btts_fit_error": calibration["btts_fit_error"],
        "handicap_consistency": consistency,
        "lambda_flow": {
            "market_prior_lambda_home": market_home,
            "market_prior_lambda_away": market_away,
            "team_adjusted_lambda_home": team_adjusted_home,
            "team_adjusted_lambda_away": team_adjusted_away,
            "final_lambda_home": final_home,
            "final_lambda_away": final_away,
        },
        "team_adjustment_contribution_table": contribution_table,
        "sensitivity": sensitivity,
        "final_score_matrix": score_matrix_records(final_matrix),
        "top_scores": final_summary["top_scores"],
        "probabilities": {
            "one_x_two": final_summary["one_x_two"],
            "over_under": final_summary["over_under"],
            "btts": final_summary["btts"],
        },
        "confidence": confidence,
        "risk_warnings": warnings,
    }
