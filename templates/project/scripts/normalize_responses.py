from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "scripts"))

from common import read_csv
from normalize_common import build_normalized_dataframe as build_standard_normalized_dataframe
from normalize_common import write_processed_outputs


SOURCE_COLUMN_MAP = {
    # "元データの列名": "response_id",
    # "元データの列名": "question_id",
    # "元データの列名": "question_text",
    # "元データの列名": "answer_text",
}

OUTPUT_COLUMNS = [
    "response_id",
    "question_id",
    "question_text",
    "answer_text",
]


def build_normalized_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if not SOURCE_COLUMN_MAP:
        raise ValueError("SOURCE_COLUMN_MAP is empty. Edit templates/project/scripts/normalize_responses.py for this project.")
    return build_standard_normalized_dataframe(df, SOURCE_COLUMN_MAP)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project-specific template for 00_raw -> 01_processed conversion")
    parser.add_argument("--input", required=True, type=Path, help="Path to raw CSV")
    parser.add_argument("--output", required=True, type=Path, help="Path to responses_normalized.csv")
    parser.add_argument("--mapping-log", required=True, type=Path, help="Path to raw_to_processed_mapping.md")
    parser.add_argument("--run-log", required=True, type=Path, help="Path to raw_to_processed.log")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_csv(args.input)
    normalized = build_normalized_dataframe(df)
    write_processed_outputs(
        normalized,
        input_path=args.input,
        output_path=args.output,
        mapping_log_path=args.mapping_log,
        run_log_path=args.run_log,
        source_column_map=SOURCE_COLUMN_MAP,
    )


if __name__ == "__main__":
    main()
