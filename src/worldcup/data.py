import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.worldcup.team_names import normalize_team_columns


INTERNATIONAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

DEFAULT_DATA_PATHS = {
    "international_results": Path("data/external/international_results/results.csv"),
    "worldcup_xlsx": Path("data/external/worldcup/WorldCup2026.xlsx"),
    "fixtures": Path("data/external/worldcup/worldcup_2026_fixtures_odds.csv"),
    "processed_history": Path("data/processed/worldcup/national_team_matches.csv"),
    "training_dataset": Path("data/processed/worldcup/worldcup_training_dataset.csv"),
    "features_for_prediction": Path("data/processed/worldcup/worldcup_fixtures_for_prediction.csv"),
    "predictions": Path("data/processed/worldcup/predictions.csv"),
    "model_report": Path("data/processed/worldcup/model_report.json"),
    "model_dir": Path("data/models/worldcup"),
    "classifier": Path("data/models/worldcup/classifier.pkl"),
    "metadata": Path("data/models/worldcup/metadata.json"),
}


def ensure_worldcup_directories() -> None:
    for path in DEFAULT_DATA_PATHS.values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "neutral"}


def _result_from_scores(home_goals: object, away_goals: object) -> object:
    if pd.isna(home_goals) or pd.isna(away_goals):
        return np.nan
    home_goals = int(home_goals)
    away_goals = int(away_goals)
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def add_match_context_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add tournament/context flags used by national-team feature engineering."""

    tournament = df.get("Tournament", pd.Series([""] * df.shape[0], index=df.index)).fillna("").astype(str)
    tournament_lower = tournament.str.lower()

    df["IsWorldCup"] = tournament_lower.str.contains("world cup", regex=False) & ~tournament_lower.str.contains(
        "qualif", regex=False
    )
    df["IsQualifier"] = tournament_lower.str.contains("qualif", regex=False)
    df["IsFriendly"] = tournament_lower.str.contains("friendly", regex=False)
    continental_tokens = [
        "euro",
        "copa am",
        "africa cup",
        "asian cup",
        "gold cup",
        "nations league",
        "confederations",
        "oceania",
    ]
    df["IsContinentalCup"] = tournament_lower.apply(
        lambda value: any(token in value for token in continental_tokens)
    )

    if "Neutral" not in df:
        df["Neutral"] = False
    df["Neutral"] = df["Neutral"].apply(_coerce_bool)

    country = df.get("Country", pd.Series([""] * df.shape[0], index=df.index)).fillna("").astype(str)
    df["HostTeamFlag"] = ((country == df["Home"]) | (country == df["Away"])) & ~df["Neutral"]
    df["MatchImportance"] = np.select(
        [
            df["IsWorldCup"],
            df["IsQualifier"],
            df["IsContinentalCup"],
            df["IsFriendly"],
        ],
        [4, 3, 2, 0],
        default=1,
    )
    return df


def normalize_international_results(df: pd.DataFrame, allow_missing_odds: bool = True) -> pd.DataFrame:
    """Convert martj42 international_results rows to the ProphitBet-compatible schema."""

    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"international_results is missing required columns: {missing}")

    normalized = pd.DataFrame()
    normalized["Date"] = pd.to_datetime(df["date"], errors="coerce")
    normalized["Season"] = normalized["Date"].dt.year.astype("Int64")
    normalized["Home"] = df["home_team"]
    normalized["Away"] = df["away_team"]
    normalized["HG"] = pd.to_numeric(df["home_score"], errors="coerce")
    normalized["AG"] = pd.to_numeric(df["away_score"], errors="coerce")
    normalized["Result"] = [
        _result_from_scores(home_goals, away_goals)
        for home_goals, away_goals in zip(normalized["HG"], normalized["AG"])
    ]

    odds_columns = {"1": np.nan, "X": np.nan, "2": np.nan}
    for column, default_value in odds_columns.items():
        normalized[column] = pd.to_numeric(df[column], errors="coerce") if column in df else default_value

    if not allow_missing_odds and normalized[["1", "X", "2"]].isna().any().any():
        raise ValueError("Historical data contains missing odds and allow_missing_odds is false.")

    optional_map = {
        "tournament": "Tournament",
        "city": "City",
        "country": "Country",
        "neutral": "Neutral",
    }
    for source, target in optional_map.items():
        normalized[target] = df[source] if source in df else (False if target == "Neutral" else "")

    normalized = normalize_team_columns(normalized, ["Home", "Away", "Country"])
    normalized = add_match_context_flags(normalized)
    normalized = normalized.dropna(subset=["Date", "Season", "Home", "Away", "HG", "AG", "Result"])
    normalized["Season"] = normalized["Season"].astype(int)
    normalized["HG"] = normalized["HG"].astype(int)
    normalized["AG"] = normalized["AG"].astype(int)
    return normalized.sort_values(["Date", "Home", "Away"]).reset_index(drop=True)


def load_international_results(
    path: Optional[str] = None,
    download_if_missing: bool = True,
    allow_missing_odds: bool = True,
) -> pd.DataFrame:
    """Load local international results or try the martj42 GitHub raw CSV."""

    path = Path(path) if path else DEFAULT_DATA_PATHS["international_results"]
    if path.exists():
        df = pd.read_csv(path)
    elif download_if_missing:
        try:
            df = pd.read_csv(INTERNATIONAL_RESULTS_URL)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
        except Exception as exc:
            raise RuntimeError(
                "Could not load local international results or download martj42 results.csv. "
                f"Place the file at {path}. Original error: {exc}"
            ) from exc
    else:
        raise FileNotFoundError(f"International results file not found: {path}")

    return normalize_international_results(df=df, allow_missing_odds=allow_missing_odds)


def _find_first_column(columns, candidates) -> Optional[str]:
    normalized = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def _read_table(path_or_url: str) -> pd.DataFrame:
    suffix = Path(str(path_or_url).split("?", 1)[0]).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path_or_url)
    return pd.read_csv(path_or_url)


def load_worldcup_football_data_xlsx(path_or_url: str) -> pd.DataFrame:
    """Read a Football-Data World Cup spreadsheet and normalize its columns."""

    if not path_or_url.startswith(("http://", "https://")) and not Path(path_or_url).exists():
        raise FileNotFoundError(f"World Cup Football-Data file not found: {path_or_url}")

    raw = _read_table(path_or_url)
    columns = raw.columns
    mapping: Dict[str, Optional[str]] = {
        "Date": _find_first_column(columns, ["Date", "MatchDate", "Match Date"]),
        "Home": _find_first_column(columns, ["HomeTeam", "Home", "Home Team"]),
        "Away": _find_first_column(columns, ["AwayTeam", "Away", "Away Team"]),
        "HG": _find_first_column(columns, ["FTHG", "HG", "HomeGoals", "Home Goals"]),
        "AG": _find_first_column(columns, ["FTAG", "AG", "AwayGoals", "Away Goals"]),
        "Result": _find_first_column(columns, ["FTR", "Result", "Res"]),
        "AvgH": _find_first_column(columns, ["AvgH", "AvgCH", "B365H"]),
        "AvgD": _find_first_column(columns, ["AvgD", "AvgCD", "B365D"]),
        "AvgA": _find_first_column(columns, ["AvgA", "AvgCA", "B365A"]),
    }

    required = ["Date", "Home", "Away"]
    missing = [column for column in required if mapping[column] is None]
    if missing:
        raise ValueError(f"World Cup file is missing required columns after mapping: {missing}")

    normalized = pd.DataFrame()
    normalized["Date"] = pd.to_datetime(raw[mapping["Date"]], errors="coerce")
    normalized["Season"] = normalized["Date"].dt.year.astype("Int64")
    normalized["Home"] = raw[mapping["Home"]]
    normalized["Away"] = raw[mapping["Away"]]
    normalized["HG"] = pd.to_numeric(raw[mapping["HG"]], errors="coerce") if mapping["HG"] else np.nan
    normalized["AG"] = pd.to_numeric(raw[mapping["AG"]], errors="coerce") if mapping["AG"] else np.nan

    if mapping["Result"]:
        normalized["Result"] = raw[mapping["Result"]].replace({"H": "H", "D": "D", "A": "A", "1": "H", "X": "D", "2": "A"})
    else:
        normalized["Result"] = [
            _result_from_scores(home_goals, away_goals)
            for home_goals, away_goals in zip(normalized["HG"], normalized["AG"])
        ]

    normalized["1"] = pd.to_numeric(raw[mapping["AvgH"]], errors="coerce") if mapping["AvgH"] else np.nan
    normalized["X"] = pd.to_numeric(raw[mapping["AvgD"]], errors="coerce") if mapping["AvgD"] else np.nan
    normalized["2"] = pd.to_numeric(raw[mapping["AvgA"]], errors="coerce") if mapping["AvgA"] else np.nan
    normalized["Tournament"] = "FIFA World Cup"
    normalized["City"] = raw[_find_first_column(columns, ["City", "Venue"])] if _find_first_column(columns, ["City", "Venue"]) else ""
    normalized["Country"] = raw[_find_first_column(columns, ["Country", "Host"])] if _find_first_column(columns, ["Country", "Host"]) else ""
    normalized["Neutral"] = True

    normalized = normalize_team_columns(normalized, ["Home", "Away", "Country"])
    normalized = add_match_context_flags(normalized)
    return normalized.dropna(subset=["Date", "Season", "Home", "Away"]).sort_values(["Date", "Home", "Away"]).reset_index(drop=True)


def normalize_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize future fixtures used by the prediction CLI."""

    required = {"Date", "Home", "Away"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Fixtures file is missing required columns: {missing}")

    fixtures = df.copy()
    fixtures["Date"] = pd.to_datetime(fixtures["Date"], errors="coerce")
    fixtures["Season"] = fixtures["Date"].dt.year.astype("Int64")
    for column in ["1", "X", "2"]:
        fixtures[column] = pd.to_numeric(fixtures[column], errors="coerce") if column in fixtures else np.nan
    for column, default in {
        "Tournament": "FIFA World Cup",
        "City": "",
        "Country": "",
        "Neutral": True,
    }.items():
        if column not in fixtures:
            fixtures[column] = default
    fixtures = normalize_team_columns(fixtures, ["Home", "Away", "Country"])
    fixtures = add_match_context_flags(fixtures)
    return fixtures.dropna(subset=["Date", "Season", "Home", "Away"]).reset_index(drop=True)

