import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.worldcup.data import DEFAULT_DATA_PATHS, ensure_worldcup_directories  # noqa: E402
from src.worldcup.predict import predict_worldcup_fixtures  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict World Cup fixtures with a trained national-team model.")
    parser.add_argument(
        "--fixtures",
        default=str(DEFAULT_DATA_PATHS["fixtures"]),
        help="Fixtures CSV with Date, Home, Away, optional 1/X/2, Tournament, City, Country, Neutral.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_DATA_PATHS["classifier"]),
        help="Path to trained classifier.pkl.",
    )
    parser.add_argument(
        "--history",
        default=str(DEFAULT_DATA_PATHS["processed_history"]),
        help="Canonical national-team history CSV used to build rolling features.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_DATA_PATHS["predictions"]),
        help="Output predictions CSV path.",
    )
    parser.add_argument(
        "--feature-output",
        default=str(DEFAULT_DATA_PATHS["features_for_prediction"]),
        help="Output feature snapshot CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_worldcup_directories()

    fixtures_path = Path(args.fixtures)
    model_path = Path(args.model)
    history_path = Path(args.history)
    if not fixtures_path.exists():
        raise FileNotFoundError(f"Fixtures file not found: {fixtures_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not history_path.exists():
        raise FileNotFoundError(
            f"Prepared history not found: {history_path}. Run scripts/prepare_worldcup_data.py first."
        )

    fixtures = pd.read_csv(fixtures_path)
    history = pd.read_csv(history_path)
    predictions, _ = predict_worldcup_fixtures(
        fixtures=fixtures,
        history_matches=history,
        model_path=model_path,
        output_path=Path(args.output),
        feature_snapshot_path=Path(args.feature_output),
    )
    print(f"Predicted {predictions.shape[0]} fixtures -> {Path(args.output)}")


if __name__ == "__main__":
    main()
