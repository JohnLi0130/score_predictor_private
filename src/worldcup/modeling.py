import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.worldcup.data import DEFAULT_DATA_PATHS
from src.worldcup.features import NationalTeamStatisticsEngine, target_from_result


LABELS = [0, 1, 2]
LABEL_NAMES = ["H", "D", "A"]


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    one_hot = np.zeros_like(y_prob)
    one_hot[np.arange(y_true.shape[0]), y_true] = 1.0
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def probability_calibration(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 5) -> Dict[str, List[Dict[str, float]]]:
    calibration: Dict[str, List[Dict[str, float]]] = {}
    edges = np.linspace(0.0, 1.0, bins + 1)
    for class_id, class_name in enumerate(LABEL_NAMES):
        class_rows = []
        observed = (y_true == class_id).astype(float)
        for low, high in zip(edges[:-1], edges[1:]):
            if high == 1.0:
                mask = (y_prob[:, class_id] >= low) & (y_prob[:, class_id] <= high)
            else:
                mask = (y_prob[:, class_id] >= low) & (y_prob[:, class_id] < high)
            if not mask.any():
                continue
            class_rows.append(
                {
                    "bin_low": float(low),
                    "bin_high": float(high),
                    "count": int(mask.sum()),
                    "mean_predicted": float(y_prob[mask, class_id].mean()),
                    "observed_rate": float(observed[mask].mean()),
                }
            )
        calibration[class_name] = class_rows
    return calibration


def build_candidate_models(random_state: int = 0) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=5000, class_weight="balanced", random_state=random_state),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", GradientBoostingClassifier(random_state=random_state)),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    XGBClassifier(
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        n_estimators=250,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    except Exception:
        pass

    return models


def evaluate_model(model: Pipeline, x_eval: pd.DataFrame, y_eval: np.ndarray) -> Dict[str, object]:
    y_prob = model.predict_proba(x_eval)
    y_pred = y_prob.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "log_loss": float(log_loss(y_eval, y_prob, labels=LABELS)),
        "brier_score": multiclass_brier_score(y_eval, y_prob),
        "confusion_matrix": confusion_matrix(y_eval, y_pred, labels=LABELS).tolist(),
        "classification_report": classification_report(
            y_eval,
            y_pred,
            labels=LABELS,
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0.0,
        ),
        "probability_calibration": probability_calibration(y_eval, y_prob),
    }


def time_based_split(df: pd.DataFrame, eval_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("Date").reset_index(drop=True)
    eval_size = max(1, int(round(df.shape[0] * eval_ratio)))
    train_size = df.shape[0] - eval_size
    if train_size < 30 or eval_size < 5:
        raise ValueError(
            f"Not enough samples for time-based split. Need at least 35 usable rows, got {df.shape[0]}."
        )
    return df.iloc[:train_size].reset_index(drop=True), df.iloc[train_size:].reset_index(drop=True)


def train_worldcup_model(
    matches: pd.DataFrame,
    mode: str,
    from_year: int,
    windows: Iterable[int],
    output_dataset_path: Path = DEFAULT_DATA_PATHS["training_dataset"],
    model_dir: Path = DEFAULT_DATA_PATHS["model_dir"],
    model_report_path: Path = DEFAULT_DATA_PATHS["model_report"],
    model_name: str = "auto",
) -> Tuple[Pipeline, Dict[str, object]]:
    matches = matches.copy()
    matches["Date"] = pd.to_datetime(matches["Date"], errors="coerce")
    matches = matches[matches["Season"] >= from_year].sort_values("Date").reset_index(drop=True)
    if matches.shape[0] < 35:
        raise ValueError(
            f"Not enough historical matches after from_year={from_year}. "
            "Use an earlier --from-year or provide more data."
        )

    if mode == "stats_plus_odds" and matches[["1", "X", "2"]].isna().any().any():
        raise ValueError(
            "stats_plus_odds requires 1/X/2 odds in the training data. "
            "Use --mode stats_only or provide historical odds."
        )

    engine = NationalTeamStatisticsEngine(windows=windows, mode=mode)
    dataset, feature_columns = engine.build_history_features(matches=matches)
    dataset = dataset.dropna(subset=["Result"]).reset_index(drop=True)

    output_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_dataset_path, index=False)

    train_df, eval_df = time_based_split(dataset, eval_ratio=0.2)
    x_train = train_df[feature_columns]
    y_train = target_from_result(train_df["Result"]).to_numpy()
    x_eval = eval_df[feature_columns]
    y_eval = target_from_result(eval_df["Result"]).to_numpy()

    candidates = build_candidate_models()
    if model_name != "auto":
        if model_name not in candidates:
            raise ValueError(f"Unknown model_name '{model_name}'. Available: {sorted(candidates)}")
        candidates = {model_name: candidates[model_name]}

    reports = {}
    best_name: Optional[str] = None
    best_model: Optional[Pipeline] = None
    best_score: Optional[float] = None
    for candidate_name, candidate in candidates.items():
        candidate.fit(x_train, y_train)
        report = evaluate_model(candidate, x_eval=x_eval, y_eval=y_eval)
        reports[candidate_name] = report
        score = report["log_loss"]
        if best_score is None or score < best_score:
            best_name = candidate_name
            best_model = candidate
            best_score = score

    if best_model is None or best_name is None:
        raise RuntimeError("No candidate model was trained.")

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "from_year": from_year,
        "windows": list(windows),
        "model_type": best_name,
        "candidate_reports": reports,
        "metrics": reports[best_name],
        "feature_columns": feature_columns,
        "label_mapping": {"H": 0, "D": 1, "A": 2},
        "training_rows": int(train_df.shape[0]),
        "validation_rows": int(eval_df.shape[0]),
        "training_start_date": str(train_df["Date"].min()),
        "training_end_date": str(train_df["Date"].max()),
        "validation_start_date": str(eval_df["Date"].min()),
        "validation_end_date": str(eval_df["Date"].max()),
        "known_teams": sorted(set(matches["Home"]).union(set(matches["Away"]))),
        "training_dataset": str(output_dataset_path),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": best_model,
        "metadata": metadata,
    }
    with (model_dir / "classifier.pkl").open("wb") as fp:
        pickle.dump(artifact, fp)
    with (model_dir / "metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2, ensure_ascii=False)
    model_report_path.parent.mkdir(parents=True, exist_ok=True)
    with model_report_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata["metrics"], fp, indent=2, ensure_ascii=False)

    return best_model, metadata


def load_model_artifact(model_path: Path) -> Tuple[Pipeline, Dict[str, object]]:
    with model_path.open("rb") as fp:
        artifact = pickle.load(fp)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"], artifact.get("metadata", {})
    metadata_path = model_path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return artifact, metadata
