import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.worldcup.data import (  # noqa: E402
    DEFAULT_DATA_PATHS,
    ensure_worldcup_directories,
    load_international_results,
    load_worldcup_football_data_xlsx,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare national-team match data for World Cup models.")
    parser.add_argument(
        "--international-results",
        default=str(DEFAULT_DATA_PATHS["international_results"]),
        help="Path to martj42 international_results results.csv.",
    )
    parser.add_argument(
        "--worldcup-xlsx",
        default=None,
        help="Optional Football-Data World Cup XLSX/CSV file to normalize and append when results are available.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_DATA_PATHS["processed_history"]),
        help="Output canonical national-team CSV path.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not try to download martj42 results.csv when the local file is missing.",
    )
    parser.add_argument(
        "--strict-odds",
        action="store_true",
        help="Fail if historical rows have missing 1/X/2 odds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_worldcup_directories()

    matches = load_international_results(
        path=args.international_results,
        download_if_missing=not args.no_download,
        allow_missing_odds=not args.strict_odds,
    )

    if args.worldcup_xlsx:
        worldcup_rows = load_worldcup_football_data_xlsx(args.worldcup_xlsx)
        completed_worldcup_rows = worldcup_rows.dropna(subset=["HG", "AG", "Result"])
        if not completed_worldcup_rows.empty:
            matches = pd.concat([matches, completed_worldcup_rows], ignore_index=True)
            matches = (
                matches.drop_duplicates(subset=["Date", "Home", "Away"], keep="last")
                .sort_values(["Date", "Home", "Away"])
                .reset_index(drop=True)
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(output, index=False)
    print(f"Prepared {matches.shape[0]} national-team matches -> {output}")


if __name__ == "__main__":
    main()
