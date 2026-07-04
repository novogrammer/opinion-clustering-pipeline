from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import REQUIRED_RESPONSE_COLUMNS, append_jsonl, read_csv, utc_now_iso, validate_required_columns, write_csv
from screening_common import SCREENED_COLUMNS, classify_answer
from validate_processed import run_validations as run_processed_validations
from validate_screened_responses import run_validations as run_screened_validations


def build_screened_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    results = working["answer_text"].map(classify_answer)
    working["is_target"] = results.map(lambda item: item[0])
    working["screening_reason"] = results.map(lambda item: item[1])
    return working[SCREENED_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply initial screening rules to normalized responses.")
    parser.add_argument("--input", required=True, type=Path, help="Path to responses_normalized.csv")
    parser.add_argument("--output", required=True, type=Path, help="Path to screened_responses.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_csv(args.input)
    validate_required_columns(df, REQUIRED_RESPONSE_COLUMNS)
    processed_errors = run_processed_validations(df)
    if processed_errors:
        raise SystemExit("\n".join(processed_errors))
    screened = build_screened_dataframe(df)
    screened_errors = run_screened_validations(screened)
    if screened_errors:
        raise SystemExit("\n".join(screened_errors))
    write_csv(screened, args.output)
    if args.log is not None:
        reason_counts = screened["screening_reason"].value_counts(dropna=False).to_dict()
        append_jsonl(
            {
                "event": "screening",
                "input": str(args.input),
                "output": str(args.output),
                "row_count": int(len(screened)),
                "target_count": int((screened["is_target"] == True).sum()),
                "reason_counts": {str(key): int(value) for key, value in reason_counts.items()},
                "created_at": utc_now_iso(),
            },
            args.log,
        )


if __name__ == "__main__":
    main()
