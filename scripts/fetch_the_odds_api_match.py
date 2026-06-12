from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from score_predictor.connectors.odds_api_normalizer import normalize_event_odds_to_v3_input
from score_predictor.connectors.the_odds_api import (
    MISSING_API_KEY_MESSAGE,
    fetch_event_markets,
    fetch_event_odds,
    fetch_events,
    fetch_sports,
    find_world_cup_sport_key,
    load_provider_config,
    match_event_by_teams,
)


def _load_aliases(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    aliases: dict[str, str] = {}
    for canonical, values in (data.get("aliases") or {}).items():
        aliases[str(canonical)] = str(canonical)
        for value in values or []:
            aliases[str(value)] = str(canonical)
    return aliases


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch The Odds API match odds.")
    parser.add_argument("--sport-key")
    parser.add_argument("--home")
    parser.add_argument("--away")
    parser.add_argument("--event-id")
    parser.add_argument("--regions", default=None)
    parser.add_argument("--markets", default=None)
    parser.add_argument("--bookmaker", default="auto")
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-sports", action="store_true")
    parser.add_argument("--print-events", action="store_true")
    parser.add_argument("--print-markets", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_provider_config(PROJECT_ROOT / "config" / "provider_the_odds_api.example.yaml")
    sport_key = args.sport_key

    if args.dry_run:
        _print_json(
            {
                "sport_key": sport_key or "auto",
                "home": args.home,
                "away": args.away,
                "event_id": args.event_id,
                "regions": args.regions or config["default_regions"],
                "markets": args.markets or config["default_markets"],
                "bookmaker": args.bookmaker,
                "note": "dry-run 不访问 The Odds API。",
            }
        )
        return 0

    try:
        if args.print_sports or not sport_key:
            sports = fetch_sports(config=config)
            if args.print_sports:
                _print_json({"sports": sports.data, "metadata": sports.metadata})
                return 0
            sport_key = find_world_cup_sport_key(sports)
            if not sport_key:
                raise RuntimeError("未自动找到世界杯 sport_key，请使用 --sport-key 手动指定。")

        if args.print_events:
            events = fetch_events(sport_key, config=config)
            _print_json({"events": events.data, "metadata": events.metadata})
            return 0

        event_id = args.event_id
        selected_event = None
        if not event_id:
            if not args.home or not args.away:
                raise RuntimeError("缺少 --home/--away 或 --event-id。")
            events = fetch_events(sport_key, config=config)
            aliases = _load_aliases(PROJECT_ROOT / "config" / "team_aliases_worldcup_2026.yaml")
            match = match_event_by_teams(
                events,
                args.home,
                args.away,
                team_aliases=aliases,
            )
            if len(match["candidates"]) != 1:
                _print_json(match)
                return 2
            selected_event = match["candidates"][0]
            event_id = str(selected_event["id"])

        if args.print_markets:
            markets = fetch_event_markets(sport_key, event_id, args.regions, config=config)
            _print_json({"markets": markets.data, "metadata": markets.metadata})
            return 0

        odds = fetch_event_odds(
            sport_key,
            event_id,
            markets=args.markets,
            regions=args.regions,
            bookmakers=None if args.bookmaker == "auto" else args.bookmaker,
            config=config,
        )
        event_odds = odds.data
        if selected_event:
            event_odds = {**selected_event, **event_odds}
        normalized = normalize_event_odds_to_v3_input(
            event_odds,
            sport_key=sport_key,
            event_id=event_id,
            bookmaker=args.bookmaker,
            preferred_bookmakers=config.get("preferred_bookmakers"),
            metadata=odds.metadata,
        )
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                yaml.safe_dump(
                    normalized["payload"],
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )
        _print_json(normalized["summary"])
        return 0
    except RuntimeError as exc:
        if str(exc) == MISSING_API_KEY_MESSAGE:
            print(MISSING_API_KEY_MESSAGE, file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
