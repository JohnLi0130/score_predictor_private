from __future__ import annotations

import json
from typing import Any

import pandas as pd


def summarize_prediction(
    score_df: pd.DataFrame,
    over_under_line: float = 2.5,
    top_n: int = 5,
) -> dict[str, Any]:
    total_goals = score_df["home_goals"] + score_df["away_goals"]
    top_scores = score_df.head(top_n)[["score", "prob"]].to_dict(orient="records")

    home_win = score_df.loc[score_df["home_goals"] > score_df["away_goals"], "prob"].sum()
    draw = score_df.loc[score_df["home_goals"] == score_df["away_goals"], "prob"].sum()
    away_win = score_df.loc[score_df["home_goals"] < score_df["away_goals"], "prob"].sum()
    over = score_df.loc[total_goals > over_under_line, "prob"].sum()
    btts_yes = score_df.loc[
        (score_df["home_goals"] > 0) & (score_df["away_goals"] > 0),
        "prob",
    ].sum()

    probabilities = {
        "home_win": float(home_win),
        "draw": float(draw),
        "away_win": float(away_win),
        "over": float(over),
        "under": float(1.0 - over),
        "btts_yes": float(btts_yes),
        "btts_no": float(1.0 - btts_yes),
    }

    return {
        "max_probability_score": top_scores[0],
        "top_scores": top_scores,
        "probabilities": probabilities,
    }


def format_probability(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_v3_section(result: dict[str, Any]) -> list[str]:
    v3 = result.get("v3")
    if not v3 or not v3.get("enabled"):
        return []

    fit = v3["joint_fit"]
    probs = v3["probabilities"]
    confidence = v3["confidence"]
    lines = [
        "[V3 multi-market calibration]",
        (
            "joint fit lambda/rho: "
            f"home {fit['lambda_home']:.3f}, "
            f"away {fit['lambda_away']:.3f}, "
            f"rho {fit['rho']:.3f}, "
            f"dc_enabled={fit['dc_enabled']}"
        ),
        (
            "lambda flow: "
            f"market {v3['lambda_flow']['market_prior_lambda_home']:.3f}/"
            f"{v3['lambda_flow']['market_prior_lambda_away']:.3f}, "
            f"team_adjusted {v3['lambda_flow']['team_adjusted_lambda_home']:.3f}/"
            f"{v3['lambda_flow']['team_adjusted_lambda_away']:.3f}, "
            f"final {v3['lambda_flow']['final_lambda_home']:.3f}/"
            f"{v3['lambda_flow']['final_lambda_away']:.3f}"
        ),
        (
            "V3 1X2: "
            f"home {format_probability(probs['one_x_two']['home'])}, "
            f"draw {format_probability(probs['one_x_two']['draw'])}, "
            f"away {format_probability(probs['one_x_two']['away'])}"
        ),
        (
            "V3 O/U 2.5: "
            f"over {format_probability(probs['over_under']['2.5']['over'])}, "
            f"under {format_probability(probs['over_under']['2.5']['under'])}"
        ),
        (
            "V3 BTTS: "
            f"yes {format_probability(probs['btts']['yes'])}, "
            f"no {format_probability(probs['btts']['no'])}"
        ),
        "V3 Top 10 scores:",
    ]
    for index, row in enumerate(v3["top_scores"], start=1):
        lines.append(f"{index}. {row['score']}  {format_probability(row['prob'])}")

    lines.append(
        "V3 confidence split: "
        f"result={confidence['result_confidence']:.2f}, "
        f"score={confidence['score_confidence']:.2f}, "
        f"market_consistency={confidence['market_consistency_score']:.2f}, "
        f"data_quality={confidence['data_quality_score']:.2f}, "
        f"sensitivity={confidence['sensitivity_stability_score']:.2f}, "
        f"final={confidence['final_confidence_score']:.2f}"
    )
    if v3.get("risk_warnings"):
        lines.append("V3 risk warnings: " + ", ".join(v3["risk_warnings"]))
    return lines


def format_human_report(result: dict[str, Any]) -> str:
    lines = [
        "仅娱乐预测，不构成投注建议。",
        f"比赛：{result['match']}",
        f"目标：{result['target']} ({result['prediction_time']})",
        (
            "最终 λ："
            f"主队 {result['final_lambda']['home']:.2f}，"
            f"客队 {result['final_lambda']['away']:.2f}，"
            f"总计 {result['final_lambda']['total']:.2f}"
        ),
        (
            "最大概率比分："
            f"{result['max_probability_score']['score']} "
            f"({format_probability(result['max_probability_score']['prob'])})"
        ),
        "Top 5 比分：",
    ]

    for index, row in enumerate(result["top_scores"], start=1):
        lines.append(f"{index}. {row['score']}  {format_probability(row['prob'])}")

    probs = result["probabilities"]
    lines.extend(
        [
            (
                "胜平负："
                f"主胜 {format_probability(probs['home_win'])}，"
                f"平局 {format_probability(probs['draw'])}，"
                f"客胜 {format_probability(probs['away_win'])}"
            ),
            (
                f"大小球 {result['over_under_line']:g}："
                f"大 {format_probability(probs['over'])}，"
                f"小 {format_probability(probs['under'])}"
            ),
            (
                "双方进球："
                f"是 {format_probability(probs['btts_yes'])}，"
                f"否 {format_probability(probs['btts_no'])}"
            ),
            f"置信度：{result['confidence']}",
        ]
    )
    if "intelligence" in result:
        intel = result["intelligence"]
        lines.extend(
            [
                "情报诊断：",
                (
                    "LSI："
                    f"主队 {intel['lineup_strength']['home'].get('level')}，"
                    f"客队 {intel['lineup_strength']['away'].get('level')}"
                ),
                (
                    "MII："
                    f"{intel['match_intensity']['level']} "
                    f"({intel['match_intensity']['score']:.0f})"
                ),
                (
                    "Narrative Heat："
                    f"{intel['narrative_heat']['level']} "
                    f"({intel['narrative_heat']['score']:.0f})"
                ),
                (
                    "数据质量："
                    f"{result['data_quality']['level']} "
                    f"({result['data_quality']['score']:.0f})"
                ),
            ]
        )
    lines.extend(
        [
            "关键驱动：" + ", ".join(result["main_drivers"]),
            "风险提示：" + ", ".join(result["warnings"]),
        ]
    )
    lines.extend(_format_v3_section(result))
    return "\n".join(lines)


def to_pretty_json(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)
