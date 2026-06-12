import numpy as np
import pandas as pd

from src.worldcup.data import load_worldcup_football_data_xlsx, normalize_international_results
from src.worldcup.features import NationalTeamStatisticsEngine


def test_normalize_international_results_maps_required_columns():
    raw = pd.DataFrame(
        {
            "date": ["2020-01-01"],
            "home_team": ["USA"],
            "away_team": ["Korea Republic"],
            "home_score": [2],
            "away_score": [1],
            "tournament": ["Friendly"],
            "city": ["Austin"],
            "country": ["United States"],
            "neutral": [False],
        }
    )

    normalized = normalize_international_results(raw)

    assert normalized.loc[0, "Home"] == "United States"
    assert normalized.loc[0, "Away"] == "South Korea"
    assert normalized.loc[0, "Result"] == "H"
    assert normalized.loc[0, "Season"] == 2020
    assert np.isnan(normalized.loc[0, "1"])


def test_football_data_xlsx_mapping_uses_average_odds(tmp_path):
    path = tmp_path / "WorldCup2026.xlsx"
    raw = pd.DataFrame(
        {
            "Date": ["2026-06-11"],
            "HomeTeam": ["Mexico"],
            "AwayTeam": ["South Africa"],
            "FTHG": [np.nan],
            "FTAG": [np.nan],
            "AvgH": [1.8],
            "AvgD": [3.4],
            "AvgA": [4.5],
        }
    )
    raw.to_excel(path, index=False)

    normalized = load_worldcup_football_data_xlsx(str(path))

    assert normalized.loc[0, "Home"] == "Mexico"
    assert normalized.loc[0, "Away"] == "South Africa"
    assert normalized.loc[0, "1"] == 1.8
    assert normalized.loc[0, "X"] == 3.4
    assert normalized.loc[0, "2"] == 4.5


def test_rolling_features_do_not_use_same_day_results():
    matches = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
            "Season": [2020, 2020, 2020],
            "Home": ["A", "C", "A"],
            "Away": ["B", "D", "C"],
            "HG": [1, 2, 0],
            "AG": [0, 1, 0],
            "Result": ["H", "H", "D"],
            "Tournament": ["Friendly", "Friendly", "Friendly"],
            "City": ["", "", ""],
            "Country": ["", "", ""],
            "Neutral": [True, True, True],
            "IsWorldCup": [False, False, False],
            "IsQualifier": [False, False, False],
            "IsFriendly": [True, True, True],
            "IsContinentalCup": [False, False, False],
            "MatchImportance": [0, 0, 0],
        }
    )

    engine = NationalTeamStatisticsEngine(windows=[1], mode="stats_only")
    features, _ = engine.build_history_features(matches)

    same_day_c = features[(features["Date"] == pd.Timestamp("2020-01-01")) & (features["Home"] == "C")].iloc[0]
    next_day_a = features[(features["Date"] == pd.Timestamp("2020-01-02")) & (features["Home"] == "A")].iloc[0]

    assert np.isnan(same_day_c["home_last1_points_per_game"])
    assert next_day_a["home_last1_points_per_game"] == 3.0

