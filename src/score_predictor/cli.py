from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import yaml

from .market.market_snapshot import compute_snapshot_movements
from .intelligence.manual_loader import load_intelligence
from .predictor import load_match_input, predict
from .report import format_human_report, to_pretty_json
from .research.match_research import build_match_research_bundle
from .research.research_report import format_research_report
from .connectors.sporttery_manual import normalize_sporttery_manual


MARKET_DISCLAIMER_ZH = "\u4ec5\u5a31\u4e50\u9884\u6d4b\uff0c\u4e0d\u6784\u6210\u6295\u6ce8\u5efa\u8bae\u3002"


def _optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML must contain a mapping/object.")
    return data


def build_market_payload(data: dict[str, Any]) -> dict[str, Any]:
    sporttery = normalize_sporttery_manual(data)
    movement: dict[str, Any] = {}
    if data.get("market_snapshots"):
        movement = compute_snapshot_movements(data["market_snapshots"])

    warnings = list(sporttery.get("warnings", []))
    warnings.extend(movement.get("warnings", []))
    return {
        "disclaimer": "Only for entertainment and probability modeling; not betting advice.",
        "market": sporttery["market"],
        "market_features": sporttery["features"],
        "movement": movement,
        "warnings": list(dict.fromkeys(warnings)),
        "audit": {
            "source_policy": "facts_only_no_prediction_articles",
            "source": "manual_sporttery",
            "sporttery_entry_mode": "manual",
            "used_prediction_sources": False,
        },
    }


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_probability_rows(table: dict) -> list[str]:
    rows: list[str] = []
    for outcome, data in table.get("outcomes", {}).items():
        rows.append(
            "  "
            f"{outcome}: odds {data['odds']:.2f}, "
            f"raw {data['raw_prob_pct']:.1f}%, "
            f"de-vig {_pct(data['fair_prob'])}, "
            f"hidden_multiplier {data['hidden_multiplier']:.3f}"
        )
    rows.append(f"  payout_rate {_pct(table['payout_rate'])}")
    rows.append(f"  bookmaker_margin {_pct(table['bookmaker_margin'])}")
    return rows


def format_market_report(payload: dict[str, Any]) -> str:
    features = payload["market_features"]
    lines = [
        MARKET_DISCLAIMER_ZH,
        "[Market probabilities]",
        "1X2 raw implied probability / de-vig market implied probability:",
    ]
    lines.extend(_format_probability_rows(features["spf"]))

    rqspf = features.get("rqspf")
    if rqspf:
        lines.append("[Sporttery handicap / rqspf]")
        lines.append(f"  handicap: {rqspf.get('handicap')}")
        lines.extend(_format_probability_rows(rqspf))

    total_goals = features.get("total_goals")
    if total_goals:
        lines.append("[Total goals]")
        for row in total_goals.get("top_total_goals", []):
            lines.append(
                f"  {row['outcome']}: odds {row['odds']:.2f}, de-vig {_pct(row['fair_prob'])}"
            )
        lines.append(
            "  implied mean total goals: "
            f"{total_goals['expected_total_goals_from_distribution']:.2f}"
        )

    correct_score = features.get("correct_score")
    if correct_score:
        lines.append("[Correct score awards]")
        for row in correct_score.get("top_scores_by_market", []):
            lines.append(
                f"  {row['outcome']}: odds {row['odds']:.2f}, de-vig {_pct(row['fair_prob'])}"
            )
        lines.append(
            "  incomplete: "
            f"{'correct_score_odds_incomplete' in correct_score.get('warnings', [])}"
        )

    movement = payload.get("movement", {}).get("markets", {})
    if movement:
        lines.append("[Odds movement]")
        for market_name, market_payload in movement.items():
            heat = market_payload.get("heat", {})
            lines.append(
                f"  {market_name}: heated={heat.get('heated_outcome')}, "
                f"level={heat.get('heat_level')}"
            )
            for outcome, row in market_payload.get("movement", {}).get("outcomes", {}).items():
                lines.append(
                    "    "
                    f"{outcome}: {row['opening_odds']:.2f} -> {row['current_odds']:.2f}, "
                    f"fair_prob_change {_pct(row['fair_prob_change'])}"
                )

    if payload.get("warnings"):
        lines.append("warnings: " + ", ".join(payload["warnings"]))
    lines.append("audit: manual_sporttery, facts_only_no_prediction_articles")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="score_predictor",
        description="Pre-match 90-minute football score probability engine.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser("predict", help="Predict one match.")
    predict_parser.add_argument("input_yaml", type=Path, help="Path to match YAML.")
    predict_parser.add_argument(
        "--intel",
        type=Path,
        default=None,
        help="Optional separate facts-only intelligence YAML.",
    )
    predict_parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only machine-readable JSON.",
    )
    predict_parser.add_argument(
        "--research",
        type=Path,
        default=None,
        help="Optional V2 research config to attach as an audited bundle.",
    )
    predict_parser.add_argument(
        "--dc-enabled",
        type=_optional_bool,
        default=None,
        metavar="{true,false}",
        help="Enable or disable V3 Dixon-Coles low-score correction.",
    )

    market_parser = subparsers.add_parser(
        "market", help="Analyze manual Sporttery market odds."
    )
    market_parser.add_argument("input_yaml", type=Path, help="Path to V2 market YAML.")
    market_parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only machine-readable JSON.",
    )

    research_parser = subparsers.add_parser(
        "research", help="Build a facts-only V2 match research bundle."
    )
    research_parser.add_argument("input_yaml", type=Path, help="Path to research YAML.")
    research_parser.add_argument(
        "--write-yaml",
        type=Path,
        default=None,
        help="Optional output path for a normalized research bundle YAML.",
    )
    research_parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only machine-readable JSON.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "predict":
        match_input = load_match_input(args.input_yaml)
        intelligence = load_intelligence(args.intel) if args.intel else None
        result = predict(
            match_input,
            intelligence=intelligence,
            dc_enabled=args.dc_enabled,
        )
        if args.research:
            result["research_bundle"] = build_match_research_bundle(_load_yaml(args.research))
        if args.json_only:
            print(to_pretty_json(result))
        else:
            print(format_human_report(result))
            if args.research:
                print()
                print(format_research_report(result["research_bundle"]))
            print()
            print(to_pretty_json(result))
        return 0

    if args.command == "market":
        payload = build_market_payload(_load_yaml(args.input_yaml))
        if args.json_only:
            print(to_pretty_json(payload))
        else:
            print(format_market_report(payload))
            print()
            print(to_pretty_json(payload))
        return 0

    if args.command == "research":
        bundle = build_match_research_bundle(_load_yaml(args.input_yaml))
        if args.write_yaml:
            args.write_yaml.parent.mkdir(parents=True, exist_ok=True)
            with args.write_yaml.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(bundle, handle, sort_keys=False, allow_unicode=True)
        if args.json_only:
            print(to_pretty_json(bundle))
        else:
            print(format_research_report(bundle))
            print()
            print(to_pretty_json(bundle))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
