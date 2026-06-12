from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TeamMatch:
    points: int
    goals_for: int
    goals_against: int
    win: int
    draw: int
    loss: int


class NationalTeamStatisticsEngine:
    """Compute no-leakage rolling features for international matches."""

    def __init__(self, windows: Iterable[int] = (5, 10, 20), mode: str = "stats_only"):
        self.windows = tuple(sorted({int(window) for window in windows}))
        if not self.windows:
            raise ValueError("At least one rolling window is required.")
        if mode not in {"stats_only", "stats_plus_odds"}:
            raise ValueError('mode must be "stats_only" or "stats_plus_odds".')
        self.mode = mode

    def build_history_features(self, matches: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        matches = self._prepare_matches(matches)
        rows = []
        history: Dict[str, Deque[TeamMatch]] = defaultdict(deque)

        for _, day_df in matches.groupby("Date", sort=True):
            day_rows = []
            for _, match in day_df.iterrows():
                feature_row = self._build_feature_row(match=match, history=history)
                feature_row["Result"] = match["Result"]
                feature_row["HG"] = match["HG"]
                feature_row["AG"] = match["AG"]
                day_rows.append(feature_row)

            rows.extend(day_rows)

            for _, match in day_df.iterrows():
                self._append_completed_match(history=history, match=match)

        feature_df = pd.DataFrame(rows)
        feature_columns = self.get_feature_columns(feature_df)
        return feature_df, feature_columns

    def build_fixture_features(
        self,
        history_matches: pd.DataFrame,
        fixtures: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        history_matches = self._prepare_matches(history_matches)
        fixtures = fixtures.sort_values(["Date", "Home", "Away"]).reset_index(drop=True)

        history: Dict[str, Deque[TeamMatch]] = defaultdict(deque)
        for _, match in history_matches.iterrows():
            self._append_completed_match(history=history, match=match)

        rows = []
        for _, fixture in fixtures.iterrows():
            rows.append(self._build_feature_row(match=fixture, history=history))

        feature_df = pd.DataFrame(rows)
        if feature_columns is None:
            feature_columns = self.get_feature_columns(feature_df)
        for column in feature_columns:
            if column not in feature_df:
                feature_df[column] = np.nan
        return feature_df, feature_columns

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        non_features = {
            "Date",
            "Season",
            "Home",
            "Away",
            "HG",
            "AG",
            "Result",
            "Tournament",
            "City",
            "Country",
            "Neutral",
            "IsWorldCup",
            "IsQualifier",
            "IsFriendly",
            "IsContinentalCup",
            "HostTeamFlag",
            "MatchImportance",
            "warning",
        }
        return [column for column in df.columns if column not in non_features]

    def _prepare_matches(self, matches: pd.DataFrame) -> pd.DataFrame:
        prepared = matches.copy()
        prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")
        prepared = prepared.dropna(subset=["Date", "Home", "Away", "HG", "AG", "Result"])
        prepared["HG"] = prepared["HG"].astype(int)
        prepared["AG"] = prepared["AG"].astype(int)
        return prepared.sort_values(["Date", "Home", "Away"]).reset_index(drop=True)

    def _build_feature_row(self, match: pd.Series, history: Dict[str, Deque[TeamMatch]]) -> Dict[str, object]:
        home = match["Home"]
        away = match["Away"]
        row = {
            "Date": match.get("Date"),
            "Season": match.get("Season", pd.to_datetime(match.get("Date")).year),
            "Home": home,
            "Away": away,
            "Tournament": match.get("Tournament", ""),
            "City": match.get("City", ""),
            "Country": match.get("Country", ""),
            "Neutral": bool(match.get("Neutral", False)),
            "is_world_cup": int(bool(match.get("IsWorldCup", False))),
            "is_qualifier": int(bool(match.get("IsQualifier", False))),
            "is_friendly": int(bool(match.get("IsFriendly", False))),
            "is_continental_cup": int(bool(match.get("IsContinentalCup", False))),
            "host_team_home": int(str(match.get("Country", "")) == str(home) and not bool(match.get("Neutral", False))),
            "host_team_away": int(str(match.get("Country", "")) == str(away) and not bool(match.get("Neutral", False))),
            "match_importance": int(match.get("MatchImportance", 1) if not pd.isna(match.get("MatchImportance", 1)) else 1),
        }

        home_history = list(history.get(home, []))
        away_history = list(history.get(away, []))
        for window in self.windows:
            home_stats = self._summarize(home_history, window)
            away_stats = self._summarize(away_history, window)
            self._add_team_window_features(row, "home", window, home_stats)
            self._add_team_window_features(row, "away", window, away_stats)

            row[f"diff_last{window}_points_per_game"] = home_stats["points_per_game"] - away_stats["points_per_game"]
            row[f"diff_last{window}_goal_diff"] = home_stats["goal_diff"] - away_stats["goal_diff"]
            row[f"diff_last{window}_win_rate"] = home_stats["win_rate"] - away_stats["win_rate"]

        if self.mode == "stats_plus_odds":
            self._add_odds_features(row, match)

        return row

    @staticmethod
    def _add_team_window_features(row: Dict[str, object], side: str, window: int, stats: Dict[str, float]) -> None:
        row[f"{side}_last{window}_points_per_game"] = stats["points_per_game"]
        row[f"{side}_last{window}_goals_for"] = stats["goals_for"]
        row[f"{side}_last{window}_goals_against"] = stats["goals_against"]
        row[f"{side}_last{window}_goal_diff"] = stats["goal_diff"]
        row[f"{side}_last{window}_win_rate"] = stats["win_rate"]
        row[f"{side}_last{window}_draw_rate"] = stats["draw_rate"]
        row[f"{side}_last{window}_loss_rate"] = stats["loss_rate"]

    @staticmethod
    def _summarize(history: List[TeamMatch], window: int) -> Dict[str, float]:
        last_matches = history[-window:]
        if not last_matches:
            return {
                "points_per_game": np.nan,
                "goals_for": np.nan,
                "goals_against": np.nan,
                "goal_diff": np.nan,
                "win_rate": np.nan,
                "draw_rate": np.nan,
                "loss_rate": np.nan,
            }

        count = len(last_matches)
        goals_for = sum(match.goals_for for match in last_matches)
        goals_against = sum(match.goals_against for match in last_matches)
        return {
            "points_per_game": sum(match.points for match in last_matches) / count,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "goal_diff": goals_for - goals_against,
            "win_rate": sum(match.win for match in last_matches) / count,
            "draw_rate": sum(match.draw for match in last_matches) / count,
            "loss_rate": sum(match.loss for match in last_matches) / count,
        }

    @staticmethod
    def _add_odds_features(row: Dict[str, object], match: pd.Series) -> None:
        odds_home = pd.to_numeric(match.get("1", np.nan), errors="coerce")
        odds_draw = pd.to_numeric(match.get("X", np.nan), errors="coerce")
        odds_away = pd.to_numeric(match.get("2", np.nan), errors="coerce")
        row["odds_home_win"] = odds_home
        row["odds_draw"] = odds_draw
        row["odds_away_win"] = odds_away

        if pd.isna(odds_home) or pd.isna(odds_draw) or pd.isna(odds_away):
            row["implied_prob_home"] = np.nan
            row["implied_prob_draw"] = np.nan
            row["implied_prob_away"] = np.nan
            row["bookmaker_margin"] = np.nan
            row["norm_prob_home"] = np.nan
            row["norm_prob_draw"] = np.nan
            row["norm_prob_away"] = np.nan
            return

        implied_home = 1.0 / odds_home
        implied_draw = 1.0 / odds_draw
        implied_away = 1.0 / odds_away
        margin = implied_home + implied_draw + implied_away
        row["implied_prob_home"] = implied_home
        row["implied_prob_draw"] = implied_draw
        row["implied_prob_away"] = implied_away
        row["bookmaker_margin"] = margin
        row["norm_prob_home"] = implied_home / margin
        row["norm_prob_draw"] = implied_draw / margin
        row["norm_prob_away"] = implied_away / margin

    @staticmethod
    def _append_completed_match(history: Dict[str, Deque[TeamMatch]], match: pd.Series) -> None:
        home_goals = int(match["HG"])
        away_goals = int(match["AG"])
        if home_goals > away_goals:
            home_points, away_points = 3, 0
            home_win, draw, away_win = 1, 0, 0
        elif home_goals < away_goals:
            home_points, away_points = 0, 3
            home_win, draw, away_win = 0, 0, 1
        else:
            home_points, away_points = 1, 1
            home_win, draw, away_win = 0, 1, 0

        history[match["Home"]].append(
            TeamMatch(
                points=home_points,
                goals_for=home_goals,
                goals_against=away_goals,
                win=home_win,
                draw=draw,
                loss=away_win,
            )
        )
        history[match["Away"]].append(
            TeamMatch(
                points=away_points,
                goals_for=away_goals,
                goals_against=home_goals,
                win=away_win,
                draw=draw,
                loss=home_win,
            )
        )


def target_from_result(result: pd.Series) -> pd.Series:
    return result.replace({"H": 0, "D": 1, "A": 2}).astype(int)


def result_from_target(target: int) -> str:
    return {0: "H", 1: "D", 2: "A"}[int(target)]

