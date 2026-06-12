import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.worldcup.data import DEFAULT_DATA_PATHS, normalize_fixtures
from src.worldcup.features import NationalTeamStatisticsEngine, result_from_target
from src.worldcup.modeling import load_model_artifact
from src.worldcup.team_names import find_unknown_teams


def predict_worldcup_fixtures(
    fixtures: pd.DataFrame,
    history_matches: pd.DataFrame,
    model_path: Path = DEFAULT_DATA_PATHS["classifier"],
    output_path: Path = DEFAULT_DATA_PATHS["predictions"],
    feature_snapshot_path: Path = DEFAULT_DATA_PATHS["features_for_prediction"],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model, metadata = load_model_artifact(model_path=model_path)
    mode = metadata.get("mode", "stats_only")
    windows = metadata.get("windows", [5, 10, 20])
    feature_columns: List[str] = metadata.get("feature_columns", [])
    known_teams = metadata.get("known_teams", [])

    fixtures = normalize_fixtures(fixtures)
    if mode == "stats_plus_odds" and fixtures[["1", "X", "2"]].isna().any().any():
        raise ValueError("stats_plus_odds model requires fixture odds columns 1/X/2.")

    engine = NationalTeamStatisticsEngine(windows=windows, mode=mode)
    feature_df, feature_columns = engine.build_fixture_features(
        history_matches=history_matches,
        fixtures=fixtures,
        feature_columns=feature_columns,
    )

    unknown_by_row: Dict[int, List[str]] = {}
    for index, row in fixtures.iterrows():
        unknown = sorted(find_unknown_teams([row["Home"], row["Away"]], known_teams))
        unknown_by_row[index] = unknown

    y_prob = model.predict_proba(feature_df[feature_columns])
    y_pred = y_prob.argmax(axis=1)
    warnings = []
    for index, unknown in unknown_by_row.items():
        row_warnings = []
        if unknown:
            row_warnings.append(f"unknown teams in training history: {', '.join(unknown)}")
        if feature_df.loc[index, feature_columns].isna().any():
            row_warnings.append("some rolling features were imputed from training medians")
        warnings.append("; ".join(row_warnings))

    predictions = pd.DataFrame(
        {
            "Date": fixtures["Date"].dt.strftime("%Y-%m-%d"),
            "Home": fixtures["Home"],
            "Away": fixtures["Away"],
            "prob_1": y_prob[:, 0].round(4),
            "prob_X": y_prob[:, 1].round(4),
            "prob_2": y_prob[:, 2].round(4),
            "predicted_result": [result_from_target(value) for value in y_pred],
            "confidence": y_prob.max(axis=1).round(4),
            "odds_1": fixtures["1"],
            "odds_X": fixtures["X"],
            "odds_2": fixtures["2"],
            "model_mode": mode,
            "features_used": json.dumps(feature_columns, ensure_ascii=False),
            "warning": warnings,
        }
    )

    feature_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(feature_snapshot_path, index=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return predictions, feature_df

