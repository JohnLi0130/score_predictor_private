from __future__ import annotations

from typing import Any

from score_predictor.schemas import MatchInput


REFERENCE_WARNING = "该市场已参与模型校准，价值判断不是独立 edge。"


def breakeven_probability(odds: float) -> float:
    if odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.")
    return 1.0 / float(odds)


def edge(model_probability: float, odds: float) -> float:
    return float(model_probability) - breakeven_probability(float(odds))


def expected_value(model_probability: float, odds: float) -> float:
    return float(model_probability) * float(odds) - 1.0


def value_row(
    *,
    market: str,
    outcome: str,
    model_probability: float,
    market_odds: float,
    used_in_calibration: bool = False,
) -> dict[str, Any]:
    reliability = "reference_only" if used_in_calibration else "independent_comparison"
    return {
        "market": market,
        "outcome": outcome,
        "model_probability": float(model_probability),
        "market_odds": float(market_odds),
        "breakeven_probability": breakeven_probability(float(market_odds)),
        "edge": edge(float(model_probability), float(market_odds)),
        "expected_value": expected_value(float(model_probability), float(market_odds)),
        "used_in_calibration": bool(used_in_calibration),
        "value_reliability": reliability,
        "warning": REFERENCE_WARNING if used_in_calibration else "",
    }


def _roles(match_input: MatchInput) -> tuple[set[str], set[str]]:
    calibration_sources = {
        source.lower().strip()
        for source in match_input.market_roles.calibration_sources
        if source
    }
    value_sources = {
        source.lower().strip()
        for source in match_input.market_roles.value_comparison_sources
        if source
    }
    return calibration_sources, value_sources


def _used_in_calibration(match_input: MatchInput, source: str = "sporttery") -> bool:
    if not match_input.market_roles.roles_configured:
        return True
    calibration_sources, value_sources = _roles(match_input)
    normalized_source = source.lower().strip()
    if normalized_source in calibration_sources:
        return True
    return bool(calibration_sources.intersection(value_sources))


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _market_source(market: dict[str, Any], fallback: str = "") -> str:
    source = market.get("source") or market.get("odds_source")
    if not source and isinstance(market.get("odds_1x2"), dict):
        source = market["odds_1x2"].get("source")
    return str(source or fallback).strip()


def _calibration_source_label(match_input: MatchInput, fallback_sources: set[str]) -> str:
    market = _as_mapping(getattr(match_input, "calibration_market", {}))
    source = _market_source(market)
    provider = _as_mapping(market.get("provider"))
    bookmaker = provider.get("bookmaker")
    if source and bookmaker:
        return f"{source} / {bookmaker}"
    if source:
        return source
    return ", ".join(sorted(fallback_sources)) or "calibration"


def _valid_odds(value: Any) -> float | None:
    try:
        odds = float(value)
    except (TypeError, ValueError):
        return None
    return odds if odds > 1.0 else None


def _extract_1x2_odds(market: dict[str, Any]) -> dict[str, float]:
    odds = _as_mapping(market.get("odds_1x2") or market.get("spf"))
    values = {
        "home": _valid_odds(odds.get("home", odds.get("win"))),
        "draw": _valid_odds(odds.get("draw")),
        "away": _valid_odds(odds.get("away", odds.get("loss"))),
    }
    return {key: value for key, value in values.items() if value is not None}


def _extract_correct_score_odds(market: dict[str, Any]) -> dict[str, float]:
    raw = market.get("correct_score_odds") or market.get("correct_score") or {}
    if isinstance(raw, dict) and isinstance(raw.get("scores"), dict):
        raw = raw["scores"]
    if not isinstance(raw, dict):
        return {}
    rows: dict[str, float] = {}
    for score, odds in raw.items():
        parsed = _valid_odds(odds)
        if parsed is not None:
            rows[str(score)] = parsed
    return rows


def _extract_total_goals_odds(market: dict[str, Any]) -> dict[str, float]:
    raw = (
        _as_mapping(market.get("sporttery_total_goals")).get("odds")
        or market.get("sporttery_total_goals_odds")
        or market.get("total_goals_odds")
        or _as_mapping(market.get("total_goals")).get("odds")
        or {}
    )
    if not isinstance(raw, dict):
        return {}
    rows: dict[str, float] = {}
    for outcome, odds in raw.items():
        parsed = _valid_odds(odds)
        if parsed is not None:
            rows[str(outcome)] = parsed
    return rows


def _extract_half_full_time_odds(market: dict[str, Any]) -> dict[str, float]:
    raw = (
        _as_mapping(market.get("half_full_time")).get("odds")
        or market.get("half_full_time_odds")
        or {}
    )
    if not isinstance(raw, dict):
        return {}
    rows: dict[str, float] = {}
    for outcome, odds in raw.items():
        parsed = _valid_odds(odds)
        if parsed is not None:
            rows[str(outcome)] = parsed
    return rows


def _selected_value_market(match_input: MatchInput) -> tuple[dict[str, Any], str, bool]:
    value_market = _as_mapping(getattr(match_input, "value_comparison_market", {}))
    if value_market and (
        _extract_1x2_odds(value_market)
        or _extract_correct_score_odds(value_market)
        or _extract_total_goals_odds(value_market)
        or _extract_half_full_time_odds(value_market)
    ):
        return value_market, _market_source(value_market, "Sporttery") or "Sporttery", False

    fallback_market = {
        "source": "calibration",
        "odds_1x2": {
            "home": match_input.odds_1x2.home,
            "draw": match_input.odds_1x2.draw,
            "away": match_input.odds_1x2.away,
        },
        "correct_score_odds": dict(match_input.correct_score_odds),
        "sporttery_total_goals_odds": dict(match_input.sporttery_total_goals_odds),
        "half_full_time_odds": dict(match_input.half_full_time_odds),
    }
    return fallback_market, "calibration", True


def _source_used_in_calibration(
    match_input: MatchInput,
    source: str,
    fallback_to_calibration: bool,
) -> bool:
    if fallback_to_calibration:
        return True
    calibration_sources, _ = _roles(match_input)
    normalized = source.lower().strip()
    if normalized in calibration_sources:
        return True
    if normalized == "sporttery":
        return _used_in_calibration(match_input, "sporttery")
    return False


def _score_probability_lookup(score_matrix: list[dict[str, Any]]) -> dict[str, float]:
    lookup: dict[str, float] = {}
    for row in score_matrix or []:
        score = row.get("score")
        if not score:
            score = f"{int(row['home_goals'])}-{int(row['away_goals'])}"
        lookup[str(score)] = float(row.get("prob", 0.0))
    return lookup


def _add_1x2_market_rows(
    rows: list[dict[str, Any]],
    odds: dict[str, float],
    probabilities: dict[str, float],
    used_in_calibration: bool,
    source: str,
) -> None:
    odds_map = {
        "主胜": odds.get("home"),
        "平局": odds.get("draw"),
        "客胜": odds.get("away"),
    }
    probability_map = {
        "主胜": probabilities.get("home", 0.0),
        "平局": probabilities.get("draw", 0.0),
        "客胜": probabilities.get("away", 0.0),
    }
    for outcome, market_odds in odds_map.items():
        if market_odds is None:
            continue
        row = value_row(
            market="胜平负",
            outcome=outcome,
            model_probability=probability_map[outcome],
            market_odds=market_odds,
            used_in_calibration=used_in_calibration,
        )
        row["market_source"] = source
        rows.append(row)


def _add_correct_score_market_rows(
    rows: list[dict[str, Any]],
    audit_only: list[dict[str, Any]],
    odds: dict[str, float],
    score_lookup: dict[str, float],
    used_in_calibration: bool,
    source: str,
) -> None:
    for score, market_odds in odds.items():
        if score in {"home_other", "draw_other", "away_other"}:
            audit_only.append(
                {
                    "market": "比分固定奖金",
                    "outcome": score,
                    "market_odds": market_odds,
                    "market_source": source,
                    "used_in_calibration": used_in_calibration,
                    "value_reliability": "audit_only",
                    "warning": "其他比分当前仅审计展示，暂不计算 EV。",
                }
            )
            continue
        if "-" not in score:
            continue
        row = value_row(
            market="比分固定奖金",
            outcome=score,
            model_probability=score_lookup.get(score, 0.0),
            market_odds=market_odds,
            used_in_calibration=used_in_calibration,
        )
        row["market_source"] = source
        rows.append(row)


def _add_total_goals_market_rows(
    rows: list[dict[str, Any]],
    odds: dict[str, float],
    total_goal_lookup: dict[str, float],
    used_in_calibration: bool,
    source: str,
) -> None:
    for outcome, market_odds in odds.items():
        key = str(outcome)
        row = value_row(
            market="总进球",
            outcome=key,
            model_probability=total_goal_lookup.get(key, 0.0),
            market_odds=market_odds,
            used_in_calibration=used_in_calibration,
        )
        row["market_source"] = source
        rows.append(row)


def _total_goal_probability_lookup(score_matrix: list[dict[str, Any]]) -> dict[str, float]:
    lookup: dict[str, float] = {}
    for row in score_matrix or []:
        total_goals = int(row.get("home_goals", 0)) + int(row.get("away_goals", 0))
        key = str(total_goals)
        lookup[key] = lookup.get(key, 0.0) + float(row.get("prob", 0.0))
        if total_goals >= 7:
            lookup["7+"] = lookup.get("7+", 0.0) + float(row.get("prob", 0.0))
    return lookup


def _add_1x2_rows(
    rows: list[dict[str, Any]],
    match_input: MatchInput,
    probabilities: dict[str, float],
    used_in_calibration: bool,
) -> None:
    odds_map = {
        "主胜": match_input.odds_1x2.home,
        "平局": match_input.odds_1x2.draw,
        "客胜": match_input.odds_1x2.away,
    }
    probability_map = {
        "主胜": probabilities.get("home", 0.0),
        "平局": probabilities.get("draw", 0.0),
        "客胜": probabilities.get("away", 0.0),
    }
    for outcome, odds in odds_map.items():
        rows.append(
            value_row(
                market="胜平负",
                outcome=outcome,
                model_probability=probability_map[outcome],
                market_odds=odds,
                used_in_calibration=used_in_calibration,
            )
        )


def _add_correct_score_rows(
    rows: list[dict[str, Any]],
    match_input: MatchInput,
    score_lookup: dict[str, float],
    used_in_calibration: bool,
) -> None:
    for score, odds in match_input.correct_score_odds.items():
        rows.append(
            value_row(
                market="比分固定奖金",
                outcome=score,
                model_probability=score_lookup.get(score, 0.0),
                market_odds=odds,
                used_in_calibration=used_in_calibration,
            )
        )


def _add_over_under_rows(
    rows: list[dict[str, Any]],
    match_input: MatchInput,
    probabilities: dict[str, dict[str, float]],
    used_in_calibration: bool,
) -> None:
    seen_lines: set[str] = set()
    for market in match_input.over_under_markets:
        line = f"{float(market.line):g}"
        if line in seen_lines:
            continue
        seen_lines.add(line)
        model = probabilities.get(line)
        if not model:
            continue
        rows.append(
            value_row(
                market="大小球",
                outcome=f"大 {line}",
                model_probability=model.get("over", 0.0),
                market_odds=market.over_odds,
                used_in_calibration=used_in_calibration,
            )
        )
        rows.append(
            value_row(
                market="大小球",
                outcome=f"小 {line}",
                model_probability=model.get("under", 0.0),
                market_odds=market.under_odds,
                used_in_calibration=used_in_calibration,
            )
        )


def _add_btts_rows(
    rows: list[dict[str, Any]],
    match_input: MatchInput,
    probabilities: dict[str, float],
    used_in_calibration: bool,
) -> None:
    if match_input.btts is None:
        return
    rows.append(
        value_row(
            market="双方进球",
            outcome="是",
            model_probability=probabilities.get("yes", 0.0),
            market_odds=match_input.btts.yes,
            used_in_calibration=used_in_calibration,
        )
    )
    rows.append(
        value_row(
            market="双方进球",
            outcome="否",
            model_probability=probabilities.get("no", 0.0),
            market_odds=match_input.btts.no,
            used_in_calibration=used_in_calibration,
        )
    )


def _add_total_goals_rows(
    rows: list[dict[str, Any]],
    match_input: MatchInput,
    total_goal_lookup: dict[str, float],
    used_in_calibration: bool,
) -> None:
    for outcome, odds in match_input.sporttery_total_goals_odds.items():
        rows.append(
            value_row(
                market="体彩总进球",
                outcome=str(outcome),
                model_probability=total_goal_lookup.get(str(outcome), 0.0),
                market_odds=odds,
                used_in_calibration=used_in_calibration,
            )
        )


def _audit_half_full_time(match_input: MatchInput, used_in_calibration: bool) -> list[dict[str, Any]]:
    reliability = "reference_only" if used_in_calibration else "audit_only"
    warning = (
        REFERENCE_WARNING
        if used_in_calibration
        else "半全场第一版仅做审计展示，未生成模型概率和 EV。"
    )
    return [
        {
            "market": "半全场",
            "outcome": outcome,
            "market_odds": odds,
            "used_in_calibration": used_in_calibration,
            "value_reliability": reliability,
            "warning": warning,
        }
        for outcome, odds in match_input.half_full_time_odds.items()
    ]


def _audit_half_full_time_odds(
    odds: dict[str, float],
    *,
    source: str,
    used_in_calibration: bool,
) -> list[dict[str, Any]]:
    reliability = "reference_only" if used_in_calibration else "audit_only"
    warning = (
        REFERENCE_WARNING
        if used_in_calibration
        else "半全场当前仅审计展示，暂不计算 EV。"
    )
    return [
        {
            "market": "半全场",
            "outcome": outcome,
            "market_odds": market_odds,
            "market_source": source,
            "used_in_calibration": used_in_calibration,
            "value_reliability": reliability,
            "warning": warning,
        }
        for outcome, market_odds in odds.items()
    ]


def negative_ev_popular_tips(rows: list[dict[str, Any]], limit: int = 5) -> list[str]:
    popular = sorted(rows, key=lambda row: row["model_probability"], reverse=True)
    tips: list[str] = []
    for row in popular:
        if row["expected_value"] < 0:
            tips.append(
                f"{row['outcome']} 是高概率结果之一，但当前赔率对应的盈亏平衡概率高于模型概率，因此没有价值优势。"
            )
        if len(tips) >= limit:
            break
    return tips


def build_value_analysis(match_input: MatchInput, v3_result: dict[str, Any]) -> dict[str, Any]:
    probabilities = v3_result.get("probabilities") or {}
    score_matrix = v3_result.get("final_score_matrix") or []
    score_lookup = _score_probability_lookup(score_matrix)
    total_goal_lookup = _total_goal_probability_lookup(score_matrix)
    value_market, value_source, used_calibration_fallback = _selected_value_market(match_input)
    used_in_calibration = _source_used_in_calibration(
        match_input,
        value_source,
        used_calibration_fallback,
    )

    rows: list[dict[str, Any]] = []
    audit_only: list[dict[str, Any]] = []
    sporttery_value_used_in_calibration = (
        used_in_calibration
        if not used_calibration_fallback
        else _used_in_calibration(match_input, "sporttery")
    )
    comparison_1x2 = _extract_1x2_odds(value_market)
    _add_1x2_market_rows(
        rows,
        comparison_1x2,
        probabilities.get("one_x_two") or {},
        used_in_calibration,
        value_source,
    )
    _add_correct_score_market_rows(
        rows,
        audit_only,
        _extract_correct_score_odds(value_market),
        score_lookup,
        sporttery_value_used_in_calibration,
        value_source,
    )
    _add_total_goals_market_rows(
        rows,
        _extract_total_goals_odds(value_market),
        total_goal_lookup,
        sporttery_value_used_in_calibration,
        value_source,
    )
    audit_only.extend(
        _audit_half_full_time_odds(
            _extract_half_full_time_odds(value_market),
            source=value_source,
            used_in_calibration=sporttery_value_used_in_calibration,
        )
    )

    if used_calibration_fallback:
        _add_over_under_rows(
            rows,
            match_input,
            probabilities.get("over_under") or {},
            used_in_calibration,
        )
        _add_btts_rows(
            rows,
            match_input,
            probabilities.get("btts") or {},
            used_in_calibration,
        )

    probability_rank = sorted(
        rows,
        key=lambda row: row["model_probability"],
        reverse=True,
    )
    value_rank = sorted(
        rows,
        key=lambda row: row["expected_value"],
        reverse=True,
    )
    warnings = sorted(
        {
            row["warning"]
            for row in rows
            if row.get("warning")
        }
    )
    warnings.extend(
        sorted({row["warning"] for row in audit_only if row.get("warning")})
    )
    if used_calibration_fallback:
        warnings.append(REFERENCE_WARNING)

    calibration_sources, value_sources = _roles(match_input)
    reliability = "reference_only" if used_in_calibration else "independent_comparison"
    return {
        "market_roles": {
            "calibration_sources": sorted(calibration_sources),
            "value_comparison_sources": sorted(value_sources),
        },
        "source_check": {
            "calibration_source": _calibration_source_label(match_input, calibration_sources),
            "value_comparison_source": value_source,
            "comparison_odds_1x2": comparison_1x2,
            "used_calibration_fallback": used_calibration_fallback,
            "used_in_calibration": used_in_calibration,
            "value_reliability": reliability,
        },
        "probability_rank": probability_rank,
        "value_rank": value_rank,
        "negative_ev_popular_tips": negative_ev_popular_tips(rows),
        "audit_only": audit_only,
        "warnings": list(dict.fromkeys(warnings)),
        "disclaimer": "价值分析不是投注建议；EV 排名不等于推荐。",
    }
