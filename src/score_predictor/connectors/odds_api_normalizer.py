from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from score_predictor.connectors.the_odds_api import normalize_team_name


BOOKMAKER_PRIORITY = [
    "pinnacle",
    "betfair_exchange",
    "betfair_ex_eu",
    "bet365",
    "matchbook",
]

PRIMARY_MODEL_MARKETS = {
    "h2h",
    "h2h_3_way",
    "spreads",
    "totals",
    "alternate_totals",
    "btts",
}

SECONDARY_AUDIT_MARKETS = {
    "alternate_spreads",
    "draw_no_bet",
    "team_totals",
}

IGNORED_MARKET_KEYS = {
    "player_first_goal_scorer",
    "player_goal_scorer_anytime",
    "player_to_receive_card",
    "alternate_spreads_cards",
    "alternate_totals_cards",
    "alternate_spreads_corners",
    "alternate_totals_corners",
    "totals_h1",
    "spreads_h1",
    "team_totals_h1",
    "h2h_3_way_h1",
}

MARKET_MODE_KEYS = {
    "省额度模式": ["h2h", "spreads", "totals"],
    "完整建模模式": [
        "h2h",
        "h2h_3_way",
        "spreads",
        "totals",
        "alternate_totals",
        "btts",
    ],
    "扩展审计模式": [
        "h2h",
        "h2h_3_way",
        "spreads",
        "totals",
        "alternate_totals",
        "btts",
        "alternate_spreads",
        "draw_no_bet",
        "team_totals",
    ],
}


def _line_key(line: float | str) -> str:
    return f"{float(line):g}"


def market_keys_for_mode(mode: str) -> list[str]:
    return list(MARKET_MODE_KEYS.get(mode, MARKET_MODE_KEYS["省额度模式"]))


def selectable_market_keys(available_keys: list[str]) -> list[str]:
    available = [str(key) for key in available_keys]
    allowed = PRIMARY_MODEL_MARKETS | SECONDARY_AUDIT_MARKETS | {"correct_score"}
    return [
        key
        for key in available
        if (key in allowed or "correct_score" in key) and key not in IGNORED_MARKET_KEYS
    ]


def available_bookmakers(event_odds: dict[str, Any]) -> list[str]:
    return [
        str(bookmaker.get("key"))
        for bookmaker in event_odds.get("bookmakers", [])
        if bookmaker.get("key")
    ]


def select_bookmaker(
    event_odds: dict[str, Any],
    bookmaker: str = "auto",
    preferred: list[str] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    bookmakers = event_odds.get("bookmakers", []) or []
    warnings: list[str] = []
    if not bookmakers:
        return None, ["The Odds API 未返回 bookmaker。"]

    by_key = {str(item.get("key")): item for item in bookmakers if item.get("key")}
    if bookmaker and bookmaker != "auto":
        selected = by_key.get(bookmaker)
        if selected is not None:
            return selected, warnings
        warnings.append(f"所选 bookmaker {bookmaker} 不可用，已尝试自动优先级。")

    priority = preferred or BOOKMAKER_PRIORITY
    for key in priority:
        if key in by_key:
            return by_key[key], warnings
    return bookmakers[0], warnings


def _market_by_key(bookmaker: dict[str, Any], market_key: str) -> dict[str, Any] | None:
    for market in bookmaker.get("markets", []) or []:
        if market.get("key") == market_key:
            return market
    return None


def _correct_score_market(bookmaker: dict[str, Any]) -> dict[str, Any] | None:
    exact = _market_by_key(bookmaker, "correct_score")
    if exact:
        return exact
    for market in bookmaker.get("markets", []) or []:
        if "correct_score" in str(market.get("key") or ""):
            return market
    return None


def _parse_h2h(
    market: dict[str, Any] | None,
    *,
    home_team: str,
    away_team: str,
) -> dict[str, float] | None:
    if not market:
        return None
    prices: dict[str, float] = {}
    normalized_home = normalize_team_name(home_team)
    normalized_away = normalize_team_name(away_team)
    for outcome in market.get("outcomes", []) or []:
        name = str(outcome.get("name", "")).strip()
        price = outcome.get("price")
        if price is None:
            continue
        normalized_name = normalize_team_name(name)
        if normalized_name == normalized_home:
            prices["home"] = float(price)
        elif normalized_name == normalized_away:
            prices["away"] = float(price)
        elif name.lower() == "draw":
            prices["draw"] = float(price)
    if {"home", "draw", "away"}.issubset(prices):
        return prices
    return None


def _parse_totals(
    market: dict[str, Any] | None,
    *,
    bookmaker: str | None = None,
    last_update: str | None = None,
) -> list[dict[str, Any]]:
    if not market:
        return []
    by_line: dict[str, dict[str, float]] = {}
    for outcome in market.get("outcomes", []) or []:
        if outcome.get("point") is None or outcome.get("price") is None:
            continue
        line = _line_key(outcome["point"])
        row = by_line.setdefault(
            line,
            {
                "line": float(outcome["point"]),
                "bookmaker": bookmaker,
                "last_update": last_update,
            },
        )
        name = str(outcome.get("name", "")).lower()
        if name == "over":
            row["over_odds"] = float(outcome["price"])
            row["over"] = float(outcome["price"])
        elif name == "under":
            row["under_odds"] = float(outcome["price"])
            row["under"] = float(outcome["price"])
    return [
        row
        for row in by_line.values()
        if "over_odds" in row and "under_odds" in row
    ]


def _parse_spreads(
    market: dict[str, Any] | None,
    *,
    home_team: str,
    away_team: str,
    bookmaker: str | None = None,
    last_update: str | None = None,
) -> list[dict[str, Any]]:
    if not market:
        return []
    normalized_home = normalize_team_name(home_team)
    normalized_away = normalize_team_name(away_team)
    home_by_line: dict[str, dict[str, float]] = {}
    away_by_line: dict[str, dict[str, float]] = {}
    for outcome in market.get("outcomes", []) or []:
        if outcome.get("point") is None or outcome.get("price") is None:
            continue
        name = str(outcome.get("name", "")).strip()
        normalized_name = normalize_team_name(name)
        line = _line_key(outcome["point"])
        item = {
            "line": float(outcome["point"]),
            "odds": float(outcome["price"]),
            "bookmaker": bookmaker,
            "last_update": last_update,
        }
        if normalized_name == normalized_home:
            home_by_line[line] = item
        elif normalized_name == normalized_away:
            away_by_line[_line_key(-float(outcome["point"]))] = item
    rows = []
    for line, home in home_by_line.items():
        away = away_by_line.get(line)
        if away:
            rows.append(
                {
                    "line": home["line"],
                    "home_odds": home["odds"],
                    "away_odds": away["odds"],
                    "bookmaker": bookmaker,
                    "last_update": last_update,
                }
            )
    return rows


def _parse_btts(market: dict[str, Any] | None) -> dict[str, float] | None:
    if not market:
        return None
    prices: dict[str, float] = {}
    for outcome in market.get("outcomes", []) or []:
        name = str(outcome.get("name", "")).strip().lower()
        price = outcome.get("price")
        if price is None:
            continue
        if name in {"yes", "y"}:
            prices["yes"] = float(price)
        elif name in {"no", "n"}:
            prices["no"] = float(price)
    return prices if {"yes", "no"}.issubset(prices) else None


def _parse_draw_no_bet(
    market: dict[str, Any] | None,
    *,
    home_team: str,
    away_team: str,
    bookmaker: str | None = None,
    last_update: str | None = None,
) -> dict[str, Any]:
    if not market:
        return {}
    normalized_home = normalize_team_name(home_team)
    normalized_away = normalize_team_name(away_team)
    row: dict[str, Any] = {"bookmaker": bookmaker, "last_update": last_update}
    for outcome in market.get("outcomes", []) or []:
        price = outcome.get("price")
        if price is None:
            continue
        normalized_name = normalize_team_name(str(outcome.get("name", "")))
        if normalized_name == normalized_home:
            row["home"] = float(price)
        elif normalized_name == normalized_away:
            row["away"] = float(price)
    return row if "home" in row and "away" in row else {}


def _parse_team_totals(
    market: dict[str, Any] | None,
    *,
    home_team: str,
    away_team: str,
    bookmaker: str | None = None,
    last_update: str | None = None,
) -> list[dict[str, Any]]:
    if not market:
        return []
    normalized_home = normalize_team_name(home_team)
    normalized_away = normalize_team_name(away_team)
    by_team_line: dict[tuple[str, str], dict[str, Any]] = {}
    for outcome in market.get("outcomes", []) or []:
        if outcome.get("point") is None or outcome.get("price") is None:
            continue
        descriptor = str(
            outcome.get("description")
            or outcome.get("participant")
            or outcome.get("team")
            or ""
        )
        normalized_team = normalize_team_name(descriptor)
        if normalized_team == normalized_home:
            team = "home"
        elif normalized_team == normalized_away:
            team = "away"
        else:
            continue
        line = _line_key(outcome["point"])
        row = by_team_line.setdefault(
            (team, line),
            {
                "team": team,
                "line": float(outcome["point"]),
                "bookmaker": bookmaker,
                "last_update": last_update,
            },
        )
        name = str(outcome.get("name", "")).lower()
        if name == "over":
            row["over_odds"] = float(outcome["price"])
            row["over"] = float(outcome["price"])
        elif name == "under":
            row["under_odds"] = float(outcome["price"])
            row["under"] = float(outcome["price"])
    return [
        row
        for row in by_team_line.values()
        if "over_odds" in row and "under_odds" in row
    ]


def _parse_correct_score(market: dict[str, Any] | None) -> dict[str, float]:
    if not market:
        return {}
    scores: dict[str, float] = {}
    for outcome in market.get("outcomes", []) or []:
        price = outcome.get("price")
        if price is None:
            continue
        name = str(outcome.get("name", "")).strip()
        match = re.search(r"\b(\d{1,2})\s*[-:]\s*(\d{1,2})\b", name)
        if match:
            scores[f"{int(match.group(1))}-{int(match.group(2))}"] = float(price)
    return scores


def _fallback_market(
    event_odds: dict[str, Any],
    market_key: str,
    selected_bookmaker_key: str,
) -> tuple[dict[str, Any] | None, str | None]:
    for bookmaker in event_odds.get("bookmakers", []) or []:
        if bookmaker.get("key") == selected_bookmaker_key:
            continue
        market = _market_by_key(bookmaker, market_key)
        if market:
            return market, str(bookmaker.get("key"))
    return None, None


def normalize_event_odds_to_v3_input(
    event_odds: dict[str, Any],
    *,
    sport_key: str,
    event_id: str | None = None,
    bookmaker: str = "auto",
    preferred_bookmakers: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    selected_bookmaker, selection_warnings = select_bookmaker(
        event_odds,
        bookmaker=bookmaker,
        preferred=preferred_bookmakers,
    )
    warnings.extend(selection_warnings)
    if selected_bookmaker is None:
        raise ValueError("The Odds API 未返回可用 bookmaker。")

    selected_key = str(selected_bookmaker.get("key"))
    home_team = str(event_odds.get("home_team", "Home"))
    away_team = str(event_odds.get("away_team", "Away"))
    markets_found = [
        str(market.get("key"))
        for market in selected_bookmaker.get("markets", []) or []
        if market.get("key")
    ]

    selected_last_update = selected_bookmaker.get("last_update")
    h2h_market = _market_by_key(selected_bookmaker, "h2h") or _market_by_key(
        selected_bookmaker,
        "h2h_3_way",
    )
    totals_market = _market_by_key(selected_bookmaker, "totals")
    alternate_totals_market = _market_by_key(selected_bookmaker, "alternate_totals")
    spreads_market = _market_by_key(selected_bookmaker, "spreads")
    alternate_spreads_market = _market_by_key(selected_bookmaker, "alternate_spreads")
    btts_market = _market_by_key(selected_bookmaker, "btts")
    draw_no_bet_market = _market_by_key(selected_bookmaker, "draw_no_bet")
    team_totals_market = _market_by_key(selected_bookmaker, "team_totals")
    correct_score_market = _correct_score_market(selected_bookmaker)
    fallback_used: dict[str, str] = {}

    if h2h_market is None:
        h2h_market, fallback_key = _fallback_market(event_odds, "h2h", selected_key)
        if fallback_key:
            fallback_used["h2h"] = fallback_key
            warnings.append(f"{selected_key} 缺少 h2h，已使用 {fallback_key}。")
    if h2h_market is None:
        h2h_market, fallback_key = _fallback_market(event_odds, "h2h_3_way", selected_key)
        if fallback_key:
            fallback_used["h2h_3_way"] = fallback_key
            warnings.append(f"{selected_key} missing h2h_3_way; used {fallback_key}.")
    if totals_market is None:
        totals_market, fallback_key = _fallback_market(event_odds, "totals", selected_key)
        if fallback_key:
            fallback_used["totals"] = fallback_key
            warnings.append(f"{selected_key} 缺少 totals，已使用 {fallback_key}。")
    if spreads_market is None:
        spreads_market, fallback_key = _fallback_market(event_odds, "spreads", selected_key)
        if fallback_key:
            fallback_used["spreads"] = fallback_key
            warnings.append(f"{selected_key} 缺少 spreads，已使用 {fallback_key}。")

    if correct_score_market is None:
        correct_score_market, fallback_key = _fallback_market(event_odds, "correct_score", selected_key)
        if fallback_key:
            fallback_used["correct_score"] = fallback_key
            warnings.append(f"{selected_key} missing correct_score; used {fallback_key}.")

    odds_1x2 = _parse_h2h(h2h_market, home_team=home_team, away_team=away_team)
    last_update_text = str(selected_last_update) if selected_last_update else None
    over_under = _parse_totals(
        totals_market,
        bookmaker=selected_key,
        last_update=last_update_text,
    )
    alternate_totals = _parse_totals(
        alternate_totals_market,
        bookmaker=selected_key,
        last_update=last_update_text,
    )
    asian_handicap = _parse_spreads(
        spreads_market,
        home_team=home_team,
        away_team=away_team,
        bookmaker=selected_key,
        last_update=last_update_text,
    )
    alternate_spreads = _parse_spreads(
        alternate_spreads_market,
        home_team=home_team,
        away_team=away_team,
        bookmaker=selected_key,
        last_update=last_update_text,
    )
    btts = _parse_btts(btts_market)
    draw_no_bet = _parse_draw_no_bet(
        draw_no_bet_market,
        home_team=home_team,
        away_team=away_team,
        bookmaker=selected_key,
        last_update=last_update_text,
    )
    team_totals = _parse_team_totals(
        team_totals_market,
        home_team=home_team,
        away_team=away_team,
        bookmaker=selected_key,
        last_update=last_update_text,
    )
    correct_score = _parse_correct_score(correct_score_market)
    all_over_under = over_under + alternate_totals
    if odds_1x2 is None:
        raise ValueError("The Odds API 返回数据中缺少可用 h2h 胜平负赔率。")
    if not over_under:
        warnings.append("未解析到 totals 大小球盘口。")
    if not asian_handicap:
        warnings.append("未解析到 spreads 让球盘口。")

    snapshot_time = (
        selected_bookmaker.get("last_update")
        or event_odds.get("commence_time")
        or datetime.now(timezone.utc).isoformat()
    )
    event_id = event_id or str(event_odds.get("id", ""))
    over_under_odds = {
        _line_key(row["line"]): {
            "over_odds": row["over_odds"],
            "under_odds": row["under_odds"],
        }
        for row in all_over_under
    }
    primary_spread = asian_handicap[0] if asian_handicap else None
    payload = {
        "match": {
            "match_id": event_id,
            "date": event_odds.get("commence_time") or "",
            "kickoff_time": event_odds.get("commence_time") or "",
            "home_team": home_team,
            "away_team": away_team,
            "competition": "FIFA World Cup",
            "stage": "Group stage",
            "timezone": "UTC",
            "venue": {"venue_type": "neutral"},
            "neutral_site": True,
            "target": "90min_score",
        },
        "prediction_time": snapshot_time,
        "odds_channels": {
            "international": {
                "role": "primary_calibration",
                "source": "the_odds_api",
                "provider": selected_key,
                "weight": 1.0,
            },
            "sporttery": {
                "role": "supplemental_calibration",
                "source": "yaml",
                "provider": "sporttery",
                "weight": 0.35,
            },
        },
        "market_roles": {
            "calibration_sources": ["international", "the_odds_api"],
            "value_comparison_sources": ["sporttery"],
        },
        "markets": {
            "international": {
                "source": "The Odds API",
                "weight": 1.0,
                "provider": {
                    "sport_key": sport_key,
                    "event_id": event_id,
                    "bookmaker": selected_key,
                    "snapshot_time": snapshot_time,
                    "metadata": metadata or {},
                },
                "odds_1x2": odds_1x2,
                "over_under": over_under,
                "alternate_totals": alternate_totals,
                "asian_handicap": asian_handicap,
                "btts": btts,
                "audit_markets": {
                    "alternate_spreads": alternate_spreads,
                    "draw_no_bet": draw_no_bet,
                    "team_totals": team_totals,
                    "ignored_market_keys": [
                        key for key in markets_found if key in IGNORED_MARKET_KEYS
                    ],
                },
            }
        },
        "market": {
            "odds_source": "The Odds API",
            "odds_1x2": {"source": "The Odds API", **odds_1x2},
            "over_under": over_under,
            "over_under_markets": all_over_under,
            "over_under_odds": over_under_odds,
        },
        "settings": {
            "market_only_mode": True,
            "market_weight": 1.0,
            "h2h_weight": 1.0,
            "totals_weight": 1.0,
            "alternate_totals_weight": 0.8,
            "spreads_weight": 0.5,
            "btts_weight": 0.6,
            "max_goals": 8,
            "dc_enabled": True,
        },
        "notes": [
            "Generated from The Odds API international market feed.",
            "API key is not stored in this file.",
        ],
    }
    if primary_spread:
        payload["market"]["asian_handicap"] = primary_spread
    if btts:
        payload["market"]["btts"] = btts
    if correct_score:
        payload["markets"]["international"]["correct_score_odds"] = correct_score
        payload["market"]["correct_score_odds"] = correct_score
    summary = {
        "event_id": event_id,
        "commence_time": event_odds.get("commence_time"),
        "api_home_team": home_team,
        "api_away_team": away_team,
        "bookmakers": available_bookmakers(event_odds),
        "selected_bookmaker": selected_key,
        "markets_found": markets_found,
        "selected_1x2": odds_1x2,
        "selected_over_under": over_under,
        "selected_alternate_totals": alternate_totals,
        "selected_asian_handicap": asian_handicap,
        "selected_btts": btts,
        "audit_markets": {
            "alternate_spreads": alternate_spreads,
            "draw_no_bet": draw_no_bet,
            "team_totals": team_totals,
        },
        "selected_correct_score": correct_score,
        "fallback_used": fallback_used,
        "warnings": warnings,
        "quota_headers": {
            key: value
            for key, value in (metadata or {}).items()
            if key.startswith("x_requests_")
        },
    }
    return {"payload": payload, "summary": summary, "warnings": warnings}
