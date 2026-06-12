from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize

from score_predictor.odds import (
    fair_1x2_probs,
    fair_over_under_probs,
    fair_two_way_probs,
    implied_prob,
    normalize_probs,
)
from score_predictor.schemas import MatchInput

from .odds_channels import (
    SPORTTERY_OTHER_KEYS,
    TOTAL_GOALS_BUCKETS,
    build_constraints_from_channels,
    devig_probabilities,
    final_weight,
    normalize_total_goals_market,
    odds_mapping_from_market,
    overround_metrics,
    score_market_quality,
    split_correct_score_market,
)
from .score_calibration import calibrated_score_matrix, summarize_score_matrix

LAMBDA_BOUNDS = (0.05, 5.0)
RHO_BOUNDS = (-0.30, 0.30)

MARKET_WEIGHTS = {
    "one_x_two": 1.0,
    "totals": 1.0,
    "alternate_totals": 0.8,
    "spreads": 0.5,
    "btts": 0.6,
    "correct_score": 0.35,
    "sporttery_1x2": 0.15,
    "sporttery_handicap_3way": 0.15,
    "sporttery_total_goals": 0.30,
    "sporttery_correct_score": 0.20,
    "sporttery_half_full": 0.0,
    "handicap_consistency": 0.0,
}


def _line_key(line: float | str) -> str:
    return f"{float(line):g}"


def _sporttery_used_in_calibration(match_input: MatchInput) -> bool:
    channel = match_input.odds_channels.sporttery
    if channel.role == "supplemental_calibration" and channel.weight > 0:
        return True
    roles = match_input.market_roles
    calibration_sources = {source.lower().strip() for source in roles.calibration_sources}
    return "sporttery" in calibration_sources


def _rmse(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def _mean_square(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(value * value for value in values) / len(values)


def _model_spread_probs(score_df: Any, line: float) -> dict[str, float]:
    home_margin = score_df["home_goals"].astype(float) + float(line)
    away_goals = score_df["away_goals"].astype(float)
    home_prob = float(score_df.loc[home_margin > away_goals, "prob"].sum())
    away_prob = float(score_df.loc[home_margin < away_goals, "prob"].sum())
    total = home_prob + away_prob
    if total <= 0:
        return {"home": 0.5, "away": 0.5}
    return {"home": home_prob / total, "away": away_prob / total}


def _model_handicap_3way_probs(score_df: Any, line: float) -> dict[str, float]:
    adjusted_home = score_df["home_goals"].astype(float) + float(line)
    away_goals = score_df["away_goals"].astype(float)
    home_prob = float(score_df.loc[adjusted_home > away_goals, "prob"].sum())
    draw_prob = float(score_df.loc[adjusted_home == away_goals, "prob"].sum())
    away_prob = float(score_df.loc[adjusted_home < away_goals, "prob"].sum())
    total = home_prob + draw_prob + away_prob
    if total <= 0:
        return {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}
    return {
        "home": home_prob / total,
        "draw": draw_prob / total,
        "away": away_prob / total,
    }


def _first_present_market(*markets: dict[str, Any] | None) -> dict[str, Any]:
    for market in markets:
        if isinstance(market, dict) and market:
            return market
    return {}


def build_market_probabilities(match_input: MatchInput) -> dict[str, Any]:
    weights = dict(MARKET_WEIGHTS)
    one_x_two = fair_1x2_probs(
        match_input.odds_1x2.home,
        match_input.odds_1x2.draw,
        match_input.odds_1x2.away,
    )
    international_raw_markets: dict[str, Any] = {
        "one_x_two": {
            "type": "one_x_two",
            "home": match_input.odds_1x2.home,
            "draw": match_input.odds_1x2.draw,
            "away": match_input.odds_1x2.away,
        }
    }

    over_under: dict[str, dict[str, float]] = {}
    for market in match_input.over_under_markets:
        over_under[_line_key(market.line)] = fair_over_under_probs(
            market.over_odds,
            market.under_odds,
        )
        international_raw_markets[f"totals_{_line_key(market.line)}"] = {
            "type": "totals",
            "over": market.over_odds,
            "under": market.under_odds,
        }

    primary_over_under_line = None
    if match_input.over_under is not None:
        primary_over_under_line = _line_key(match_input.over_under.line)

    btts: dict[str, float] | None = None
    if match_input.btts is not None:
        fair = normalize_probs(
            {
                "yes": implied_prob(match_input.btts.yes),
                "no": implied_prob(match_input.btts.no),
            }
        )
        btts = {"yes": fair["yes"], "no": fair["no"]}
        international_raw_markets["btts"] = {
            "type": "btts",
            "yes": match_input.btts.yes,
            "no": match_input.btts.no,
        }

    spreads: dict[str, float] | None = None
    if match_input.asian_handicap is not None:
        fair = fair_two_way_probs(
            match_input.asian_handicap.home_odds,
            match_input.asian_handicap.away_odds,
        )
        spreads = {
            "line": float(match_input.asian_handicap.line),
            "home": fair["first"],
            "away": fair["second"],
        }
        international_raw_markets["spreads"] = {
            "type": "spreads",
            "home": match_input.asian_handicap.home_odds,
            "away": match_input.asian_handicap.away_odds,
        }

    generic_correct_score: dict[str, float] = {}
    if match_input.correct_score_odds:
        generic_correct_score = devig_probabilities(match_input.correct_score_odds)
        international_raw_markets["correct_score"] = {
            "type": "correct_score",
            "scores": dict(match_input.correct_score_odds),
        }

    sporttery_raw_markets: dict[str, Any] = {}
    sporttery_1x2: dict[str, float] | None = None
    if match_input.sporttery_1x2 is not None:
        sporttery_1x2 = fair_1x2_probs(
            match_input.sporttery_1x2.home,
            match_input.sporttery_1x2.draw,
            match_input.sporttery_1x2.away,
        )
        sporttery_raw_markets["sporttery_1x2"] = {
            "type": "one_x_two",
            "home": match_input.sporttery_1x2.home,
            "draw": match_input.sporttery_1x2.draw,
            "away": match_input.sporttery_1x2.away,
        }

    sporttery_handicap_3way: dict[str, Any] | None = None
    if match_input.sporttery_handicap_3way is not None:
        fair = fair_1x2_probs(
            match_input.sporttery_handicap_3way.home,
            match_input.sporttery_handicap_3way.draw,
            match_input.sporttery_handicap_3way.away,
        )
        sporttery_handicap_3way = {
            "line": float(match_input.sporttery_handicap_3way.handicap),
            "home": fair["home"],
            "draw": fair["draw"],
            "away": fair["away"],
        }
        sporttery_raw_markets["sporttery_handicap_3way"] = {
            "type": "handicap_3way",
            "home": match_input.sporttery_handicap_3way.home,
            "draw": match_input.sporttery_handicap_3way.draw,
            "away": match_input.sporttery_handicap_3way.away,
        }

    sporttery_correct_score: dict[str, float] = {}
    if match_input.sporttery_correct_score_odds and _sporttery_used_in_calibration(match_input):
        sporttery_correct_score = devig_probabilities(match_input.sporttery_correct_score_odds)
        sporttery_raw_markets["sporttery_correct_score"] = {
            "type": "correct_score",
            "scores": {
                **dict(match_input.sporttery_correct_score_odds),
                **{
                    key: value
                    for key, value in match_input.correct_score_other_odds.items()
                    if key in SPORTTERY_OTHER_KEYS
                },
            },
        }

    sporttery_total_goals: dict[str, float] = {}
    if match_input.sporttery_total_goals_odds:
        sporttery_total_goals = devig_probabilities(match_input.sporttery_total_goals_odds)
        sporttery_raw_markets["sporttery_total_goals"] = {
            "type": "total_goals",
            "odds": dict(match_input.sporttery_total_goals_odds),
        }

    if match_input.half_full_time_odds:
        sporttery_raw_markets["sporttery_half_full"] = {
            "type": "half_full",
            "odds": dict(match_input.half_full_time_odds),
        }

    constraints = build_constraints_from_channels(
        {
            "one_x_two": one_x_two,
            "over_under": over_under,
            "spreads": spreads,
            "btts": btts,
            "correct_score": generic_correct_score,
            "raw_markets": international_raw_markets,
        },
        {
            "one_x_two": sporttery_1x2,
            "handicap_3way": sporttery_handicap_3way,
            "correct_score": sporttery_correct_score,
            "total_goals": sporttery_total_goals,
            "half_full": bool(match_input.half_full_time_odds),
            "raw_markets": sporttery_raw_markets,
        },
        match_input.settings,
    )
    qualities = constraints["market_quality"]
    consistency = constraints["channel_consistency"]

    one_x_two_quality = qualities.get("international.one_x_two") or score_market_quality(
        international_raw_markets["one_x_two"]
    )
    primary_consistency = {"score": 1.0}
    sporttery_consistency = consistency

    weights["one_x_two"] = final_weight(
        float(match_input.settings.x1x2_weight) * float(match_input.settings.h2h_weight),
        one_x_two_quality,
        primary_consistency,
    )
    weights["totals"] = float(match_input.settings.ou_weight) * float(
        match_input.settings.totals_weight
    )
    weights["alternate_totals"] = float(match_input.settings.ou_weight) * float(
        match_input.settings.alternate_totals_weight
    )
    weights["spreads"] = float(match_input.settings.spreads_weight)
    weights["btts"] = float(match_input.settings.btts_weight)
    weights["correct_score"] = final_weight(
        float(match_input.settings.correct_score_weight),
        qualities.get("international.correct_score")
        or score_market_quality({"type": "correct_score", "scores": dict(match_input.correct_score_odds)}),
        primary_consistency,
    ) if generic_correct_score else 0.0
    weights["sporttery_1x2"] = final_weight(
        float(match_input.settings.sporttery_1x2_weight),
        qualities.get("sporttery.sporttery_1x2")
        or score_market_quality(sporttery_raw_markets.get("sporttery_1x2")),
        sporttery_consistency,
    ) if sporttery_1x2 else 0.0
    weights["sporttery_handicap_3way"] = final_weight(
        float(match_input.settings.sporttery_handicap_3way_weight),
        qualities.get("sporttery.sporttery_handicap_3way")
        or score_market_quality(sporttery_raw_markets.get("sporttery_handicap_3way")),
        sporttery_consistency,
    ) if sporttery_handicap_3way else 0.0
    weights["sporttery_total_goals"] = final_weight(
        float(match_input.settings.sporttery_total_goals_weight),
        qualities.get("sporttery.sporttery_total_goals")
        or score_market_quality(sporttery_raw_markets.get("sporttery_total_goals")),
        sporttery_consistency,
    ) if sporttery_total_goals else 0.0
    weights["sporttery_correct_score"] = final_weight(
        float(match_input.settings.sporttery_correct_score_weight),
        qualities.get("sporttery.sporttery_correct_score")
        or score_market_quality(sporttery_raw_markets.get("sporttery_correct_score")),
        sporttery_consistency,
    ) if sporttery_correct_score else 0.0
    weights["sporttery_half_full"] = 0.0

    one_x_two_sources = [
        {
            "channel": "international",
            "market": "h2h_3_way",
            "probabilities": one_x_two,
            "weight": weights["one_x_two"],
        }
    ]
    if sporttery_1x2:
        one_x_two_sources.append(
            {
                "channel": "sporttery",
                "market": "sporttery_1x2",
                "probabilities": sporttery_1x2,
                "weight": weights["sporttery_1x2"],
            }
        )

    correct_score_sources = []
    if generic_correct_score:
        correct_score_sources.append(
            {
                "channel": "international",
                "market": "correct_score",
                "probabilities": generic_correct_score,
                "weight": weights["correct_score"],
            }
        )
    if sporttery_correct_score:
        correct_score_sources.append(
            {
                "channel": "sporttery",
                "market": "sporttery_correct_score",
                "probabilities": sporttery_correct_score,
                "weight": weights["sporttery_correct_score"],
            }
        )

    def status_row(name: str, base_weight: float, final: float, raw_market: dict[str, Any] | None, status: str) -> dict[str, Any]:
        quality = score_market_quality(raw_market or {})
        return {
            "market": name,
            "status": status,
            "base_weight": float(base_weight),
            "market_quality_score": float(quality.get("score", 0.25)),
            "market_quality_level": quality.get("level", "low"),
            "consistency_score": float(consistency.get("score", 1.0)),
            "final_weight": float(final),
            "payout_rate": quality.get("payout_rate"),
            "overround": quality.get("overround"),
            "warnings": quality.get("warnings", []),
        }

    sporttery_market_status = {
        "sporttery_1x2": status_row(
            "sporttery_1x2",
            match_input.settings.sporttery_1x2_weight,
            weights["sporttery_1x2"],
            sporttery_raw_markets.get("sporttery_1x2"),
            "soft_calibration" if sporttery_1x2 else "ignored",
        ),
        "sporttery_handicap_3way": status_row(
            "sporttery_handicap_3way",
            match_input.settings.sporttery_handicap_3way_weight,
            weights["sporttery_handicap_3way"],
            sporttery_raw_markets.get("sporttery_handicap_3way"),
            "soft_calibration" if sporttery_handicap_3way else "ignored",
        ),
        "sporttery_correct_score": status_row(
            "sporttery_correct_score",
            match_input.settings.sporttery_correct_score_weight,
            weights["sporttery_correct_score"],
            sporttery_raw_markets.get("sporttery_correct_score"),
            "soft_calibration" if sporttery_correct_score else "ignored",
        ),
        "sporttery_total_goals": status_row(
            "sporttery_total_goals",
            match_input.settings.sporttery_total_goals_weight,
            weights["sporttery_total_goals"],
            sporttery_raw_markets.get("sporttery_total_goals"),
            "soft_calibration" if sporttery_total_goals else "ignored",
        ),
        "sporttery_half_full": status_row(
            "sporttery_half_full",
            0.0,
            0.0,
            sporttery_raw_markets.get("sporttery_half_full"),
            "audit_only" if match_input.half_full_time_odds else "ignored",
        ),
    }

    combined_correct_score = dict(generic_correct_score)
    combined_correct_score.update(sporttery_correct_score)

    return {
        "one_x_two": one_x_two,
        "one_x_two_sources": one_x_two_sources,
        "over_under": over_under,
        "primary_over_under_line": primary_over_under_line,
        "spreads": spreads,
        "btts": btts,
        "correct_score": combined_correct_score,
        "correct_score_sources": correct_score_sources,
        "sporttery_total_goals": sporttery_total_goals,
        "sporttery_handicap_3way": sporttery_handicap_3way,
        "market_quality": qualities,
        "channel_consistency": consistency,
        "sporttery_market_status": sporttery_market_status,
        "odds_channels": {
            "international": match_input.odds_channels.international.dict(),
            "sporttery": match_input.odds_channels.sporttery.dict(),
        },
        "weights": weights,
    }


def weighted_market_loss(
    params: np.ndarray | list[float] | tuple[float, ...],
    market_probs: dict[str, Any],
    dc_enabled: bool = False,
    max_goals: int = 10,
) -> float:
    lambda_home = float(params[0])
    lambda_away = float(params[1])
    rho = float(params[2]) if dc_enabled and len(params) > 2 else 0.0
    if not (LAMBDA_BOUNDS[0] <= lambda_home <= LAMBDA_BOUNDS[1]):
        return 1e6
    if not (LAMBDA_BOUNDS[0] <= lambda_away <= LAMBDA_BOUNDS[1]):
        return 1e6
    if not (RHO_BOUNDS[0] <= rho <= RHO_BOUNDS[1]):
        return 1e6

    try:
        matrix = calibrated_score_matrix(
            lambda_home,
            lambda_away,
            rho=rho,
            dc_enabled=dc_enabled,
            max_goals=max_goals,
        )
    except (RuntimeError, ValueError):
        return 1e6

    over_under_lines = tuple(
        sorted(float(line) for line in (market_probs.get("over_under") or {}).keys())
    ) or (1.5, 2.5, 3.5, 4.5)
    model = summarize_score_matrix(matrix, over_under_lines=over_under_lines)
    loss = 0.0

    one_x_two_sources = market_probs.get("one_x_two_sources") or []
    if not one_x_two_sources and market_probs.get("one_x_two"):
        one_x_two_sources = [
            {
                "probabilities": market_probs["one_x_two"],
                "weight": market_probs["weights"].get("one_x_two", 1.0),
            }
        ]
    for source in one_x_two_sources:
        one_x_two = source.get("probabilities") or {}
        if one_x_two:
            errors = [
                model["one_x_two"][outcome] - one_x_two[outcome]
                for outcome in ("home", "draw", "away")
                if outcome in one_x_two
            ]
            loss += float(source.get("weight", 0.0)) * _mean_square(errors)

    for line, market in (market_probs.get("over_under") or {}).items():
        model_ou = model["over_under"].get(_line_key(line))
        if not model_ou:
            continue
        errors = [
            model_ou[outcome] - market[outcome]
            for outcome in ("over", "under")
            if outcome in market
        ]
        if _line_key(line) == market_probs.get("primary_over_under_line"):
            weight = market_probs["weights"].get("totals", 1.0)
        else:
            weight = market_probs["weights"].get("alternate_totals", 0.8)
        loss += weight * _mean_square(errors)

    spreads = market_probs.get("spreads")
    if spreads:
        model_spread = _model_spread_probs(matrix, float(spreads["line"]))
        errors = [
            model_spread[outcome] - spreads[outcome]
            for outcome in ("home", "away")
            if outcome in spreads
        ]
        loss += market_probs["weights"]["spreads"] * _mean_square(errors)

    btts = market_probs.get("btts")
    if btts:
        errors = [
            model["btts"][outcome] - btts[outcome]
            for outcome in ("yes", "no")
            if outcome in btts
        ]
        loss += market_probs["weights"]["btts"] * _mean_square(errors)

    correct_score_sources_for_loss = market_probs.get("correct_score_sources") or []
    if "correct_score" in market_probs and not market_probs.get("correct_score"):
        correct_score_sources_for_loss = []
    for source in correct_score_sources_for_loss:
        correct_score = source.get("probabilities") or {}
        if correct_score:
            errors = [
                model["correct_scores"].get(score, 0.0) - probability
                for score, probability in correct_score.items()
            ]
            loss += float(source.get("weight", 0.0)) * _mean_square(errors)

    sporttery_total_goals = market_probs.get("sporttery_total_goals") or {}
    if sporttery_total_goals:
        errors = [
            model["total_goals"].get(bucket, 0.0) - sporttery_total_goals[bucket]
            for bucket in TOTAL_GOALS_BUCKETS
            if bucket in sporttery_total_goals
        ]
        loss += market_probs["weights"]["sporttery_total_goals"] * _mean_square(errors)

    sporttery_handicap_3way = market_probs.get("sporttery_handicap_3way")
    if sporttery_handicap_3way:
        model_handicap = _model_handicap_3way_probs(
            matrix,
            float(sporttery_handicap_3way["line"]),
        )
        errors = [
            model_handicap[outcome] - sporttery_handicap_3way[outcome]
            for outcome in ("home", "draw", "away")
            if outcome in sporttery_handicap_3way
        ]
        loss += market_probs["weights"]["sporttery_handicap_3way"] * _mean_square(errors)

    return float(loss)


def _fit_errors(
    market_probs: dict[str, Any],
    model: dict[str, Any],
    score_df: Any | None = None,
) -> dict[str, Any]:
    errors: dict[str, Any] = {}

    one_x_two = market_probs.get("one_x_two") or {}
    if one_x_two:
        details = {}
        diffs = []
        for outcome in ("home", "draw", "away"):
            market_probability = float(one_x_two[outcome])
            model_probability = float(model["one_x_two"][outcome])
            difference = model_probability - market_probability
            diffs.append(difference)
            details[outcome] = {
                "market_probability": market_probability,
                "model_probability": model_probability,
                "difference": difference,
            }
        errors["one_x_two"] = {"rmse": _rmse(diffs), "details": details}

    source_errors = {}
    for source in market_probs.get("one_x_two_sources") or []:
        probabilities = source.get("probabilities") or {}
        if not probabilities:
            continue
        details = {}
        diffs = []
        for outcome in ("home", "draw", "away"):
            market_probability = float(probabilities[outcome])
            model_probability = float(model["one_x_two"][outcome])
            difference = model_probability - market_probability
            diffs.append(difference)
            details[outcome] = {
                "market_probability": market_probability,
                "model_probability": model_probability,
                "difference": difference,
            }
        source_errors[f"{source.get('channel', 'unknown')}.{source.get('market', 'one_x_two')}"] = {
            "rmse": _rmse(diffs),
            "weight": float(source.get("weight", 0.0)),
            "details": details,
        }
    if source_errors:
        errors["one_x_two_sources"] = source_errors

    over_under_errors = {}
    for line, market in (market_probs.get("over_under") or {}).items():
        model_ou = model["over_under"].get(_line_key(line))
        if not model_ou:
            continue
        details = {}
        diffs = []
        for outcome in ("over", "under"):
            market_probability = float(market[outcome])
            model_probability = float(model_ou[outcome])
            difference = model_probability - market_probability
            diffs.append(difference)
            details[outcome] = {
                "market_probability": market_probability,
                "model_probability": model_probability,
                "difference": difference,
            }
        over_under_errors[_line_key(line)] = {"rmse": _rmse(diffs), "details": details}
    if over_under_errors:
        errors["over_under"] = over_under_errors

    spreads = market_probs.get("spreads")
    if spreads and score_df is not None:
        model_spread = _model_spread_probs(score_df, float(spreads["line"]))
        details = {}
        diffs = []
        for outcome in ("home", "away"):
            market_probability = float(spreads[outcome])
            model_probability = float(model_spread[outcome])
            difference = model_probability - market_probability
            diffs.append(difference)
            details[outcome] = {
                "market_probability": market_probability,
                "model_probability": model_probability,
                "difference": difference,
            }
        errors["spreads"] = {
            "line": float(spreads["line"]),
            "rmse": _rmse(diffs),
            "details": details,
        }

    btts = market_probs.get("btts")
    if btts:
        details = {}
        diffs = []
        for outcome in ("yes", "no"):
            market_probability = float(btts[outcome])
            model_probability = float(model["btts"][outcome])
            difference = model_probability - market_probability
            diffs.append(difference)
            details[outcome] = {
                "market_probability": market_probability,
                "model_probability": model_probability,
                "difference": difference,
            }
        errors["btts"] = {"rmse": _rmse(diffs), "details": details}

    sporttery_total_goals = market_probs.get("sporttery_total_goals") or {}
    if sporttery_total_goals:
        details = {}
        diffs = []
        for bucket in TOTAL_GOALS_BUCKETS:
            if bucket not in sporttery_total_goals:
                continue
            market_probability = float(sporttery_total_goals[bucket])
            model_probability = float(model["total_goals"].get(bucket, 0.0))
            difference = model_probability - market_probability
            diffs.append(difference)
            details[bucket] = {
                "market_probability": market_probability,
                "model_probability": model_probability,
                "difference": difference,
            }
        errors["sporttery_total_goals"] = {
            "rmse": _rmse(diffs),
            "details": details,
        }

    sporttery_handicap_3way = market_probs.get("sporttery_handicap_3way")
    if sporttery_handicap_3way and score_df is not None:
        model_handicap = _model_handicap_3way_probs(
            score_df,
            float(sporttery_handicap_3way["line"]),
        )
        details = {}
        diffs = []
        for outcome in ("home", "draw", "away"):
            market_probability = float(sporttery_handicap_3way[outcome])
            model_probability = float(model_handicap[outcome])
            difference = model_probability - market_probability
            diffs.append(difference)
            details[outcome] = {
                "market_probability": market_probability,
                "model_probability": model_probability,
                "difference": difference,
            }
        errors["sporttery_handicap_3way"] = {
            "line": float(sporttery_handicap_3way["line"]),
            "rmse": _rmse(diffs),
            "details": details,
        }

    return errors


def _correct_score_table(
    market_probs: dict[str, Any],
    model: dict[str, Any],
) -> tuple[list[dict[str, float | str]], float | None]:
    sources = market_probs.get("correct_score_sources") or []
    if not sources and market_probs.get("correct_score"):
        sources = [
            {
                "channel": "unknown",
                "market": "correct_score",
                "probabilities": market_probs["correct_score"],
                "weight": market_probs["weights"].get("correct_score", 0.0),
            }
        ]
    if not sources:
        return [], None

    table = []
    diffs = []
    for source in sources:
        correct_score = source.get("probabilities") or {}
        for score, market_probability in correct_score.items():
            model_probability = float(model["correct_scores"].get(score, 0.0))
            difference = model_probability - float(market_probability)
            diffs.append(difference)
            table.append(
                {
                    "channel": str(source.get("channel", "unknown")),
                    "score": score,
                    "market_probability": float(market_probability),
                    "model_probability": model_probability,
                    "difference": difference,
                    "weight": float(source.get("weight", 0.0)),
                }
            )

    table.sort(key=lambda row: abs(float(row["difference"])), reverse=True)
    return table, _rmse(diffs)


def calibrate_markets(
    match_input: MatchInput,
    initial: tuple[float, float],
    dc_enabled: bool = False,
    max_goals: int = 10,
) -> dict[str, Any]:
    market_probs = build_market_probabilities(match_input)
    initial_home = min(max(float(initial[0]), LAMBDA_BOUNDS[0]), LAMBDA_BOUNDS[1])
    initial_away = min(max(float(initial[1]), LAMBDA_BOUNDS[0]), LAMBDA_BOUNDS[1])
    x0 = np.array([initial_home, initial_away, 0.0], dtype=float)
    bounds: tuple[tuple[float, float], ...] = (
        LAMBDA_BOUNDS,
        LAMBDA_BOUNDS,
        RHO_BOUNDS,
    )
    if not dc_enabled:
        x0 = x0[:2]
        bounds = (LAMBDA_BOUNDS, LAMBDA_BOUNDS)

    result = minimize(
        lambda x: weighted_market_loss(
            x,
            market_probs,
            dc_enabled=dc_enabled,
            max_goals=max_goals,
        ),
        x0=x0,
        method="L-BFGS-B",
        bounds=bounds,
    )

    if result.success:
        fitted = result.x
        optimizer_warning = None
    else:
        fitted = x0
        optimizer_warning = f"v3_market_calibration_optimizer_fallback:{result.message}"

    lambda_home = float(fitted[0])
    lambda_away = float(fitted[1])
    rho = float(fitted[2]) if dc_enabled and len(fitted) > 2 else 0.0
    matrix = calibrated_score_matrix(
        lambda_home,
        lambda_away,
        rho=rho,
        dc_enabled=dc_enabled,
        max_goals=max_goals,
    )
    over_under_lines = tuple(
        sorted(float(line) for line in (market_probs.get("over_under") or {}).keys())
    ) or (1.5, 2.5, 3.5, 4.5)
    model = summarize_score_matrix(matrix, over_under_lines=over_under_lines)
    fit_errors = _fit_errors(market_probs, model, matrix)
    correct_score_table, correct_score_fit_error = _correct_score_table(
        market_probs,
        model,
    )

    warnings = []
    if optimizer_warning:
        warnings.append(optimizer_warning)
    if not market_probs.get("over_under"):
        warnings.append("v3_over_under_markets_missing")
    if not market_probs.get("btts"):
        warnings.append("v3_btts_market_missing")
    correct_score_sources = market_probs.get("correct_score_sources") or []
    has_international_correct_score = any(
        source.get("channel") == "international" for source in correct_score_sources
    )
    has_sporttery_correct_score = any(
        source.get("channel") == "sporttery" for source in correct_score_sources
    )
    if has_sporttery_correct_score:
        warnings.append("sporttery_correct_score_soft_constraint")
        if not has_international_correct_score:
            warnings.append("sporttery_correct_score_supplemented_missing_international")
    if match_input.correct_score_other_odds:
        warnings.append("correct_score_other_not_used")
    if not correct_score_sources:
        warnings.append("v3_correct_score_market_missing")
    if market_probs.get("sporttery_total_goals"):
        warnings.append("sporttery_total_goals_used")
    for status in (market_probs.get("sporttery_market_status") or {}).values():
        for warning in status.get("warnings") or []:
            warnings.append(warning)
    warnings.extend((market_probs.get("channel_consistency") or {}).get("warnings") or [])
    warnings = list(dict.fromkeys(warnings))

    return {
        "market_probabilities": market_probs,
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "rho": rho,
        "dc_enabled": dc_enabled,
        "loss": weighted_market_loss(
            fitted,
            market_probs,
            dc_enabled=dc_enabled,
            max_goals=max_goals,
        ),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "market_fit_errors": fit_errors,
        "correct_score_fit_error": correct_score_fit_error,
        "correct_score_calibration": correct_score_table,
        "market_quality": market_probs.get("market_quality", {}),
        "channel_consistency": market_probs.get("channel_consistency", {}),
        "sporttery_market_status": market_probs.get("sporttery_market_status", {}),
        "btts_fit_error": fit_errors.get("btts", {}).get("rmse"),
        "warnings": warnings,
    }
