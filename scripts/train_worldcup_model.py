import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.worldcup.data import DEFAULT_DATA_PATHS, ensure_worldcup_directories  # noqa: E402
from src.worldcup.modeling import train_worldcup_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a World Cup / national-team 1/X/2 model.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_DATA_PATHS["processed_history"]),
        help="Canonical national-team matches CSV from prepare_worldcup_data.py.",
    )
    parser.add_argument(
        "--mode",
        choices=["stats_only", "stats_plus_odds"],
        default="stats_only",
        help="Feature mode. stats_plus_odds requires 1/X/2 in training and prediction data.",
    )
    parser.add_argument("--from-year", type=int, default=2010, help="First season/year to include.")
    parser.add_argument("--windows", nargs="+", type=int, default=[5, 10, 20], help="Rolling windows.")
    parser.add_argument(
        "--output-dataset",
        default=str(DEFAULT_DATA_PATHS["training_dataset"]),
        help="Output feature dataset CSV.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_DATA_PATHS["model_dir"]),
        help="Directory for classifier.pkl and metadata.json.",
    )
    parser.add_argument(
        "--model-report",
        default=str(DEFAULT_DATA_PATHS["model_report"]),
        help="Output model evaluation report JSON.",
    )
    parser.add_argument(
        "--model",
        default="auto",
        help="Candidate to train: auto, logistic_regression, random_forest, gradient_boosting, xgboost.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_worldcup_directories()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Prepared history not found: {input_path}. Run scripts/prepare_worldcup_data.py first."
        )

    matches = pd.read_csv(input_path)
    _, metadata = train_worldcup_model(
        matches=matches,
        mode=args.mode,
        from_year=args.from_year,
        windows=args.windows,
        output_dataset_path=Path(args.output_dataset),
        model_dir=Path(args.model_dir),
        model_report_path=Path(args.model_report),
        model_name=args.model,
    )

    print(f"Trained {metadata['model_type']} model in {args.mode} mode.")
    print(f"Validation accuracy: {metadata['metrics']['accuracy']:.4f}")
    print(f"Validation log_loss: {metadata['metrics']['log_loss']:.4f}")
    print(f"Saved model -> {Path(args.model_dir) / 'classifier.pkl'}")
    print(f"Saved metadata -> {Path(args.model_dir) / 'metadata.json'}")


if __name__ == "__main__":
    main()
