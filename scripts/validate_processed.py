from __future__ import annotations

import argparse
from pathlib import Path

from normalize import validate_dataframe
from common import read_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 01_processed/responses_normalized.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to responses_normalized.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append validation results as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_csv(args.input)
    errors = validate_dataframe(df, input_path=args.input, log_path=args.log)
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
