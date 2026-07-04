from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns
from review_common import REVIEW_SAMPLE_COLUMNS as OUTPUT_COLUMNS


def validate_required_text(df: pd.DataFrame, columns: list[str]) -> list[str]:
    errors: list[str] = []
    for column in columns:
        blank_mask = df[column].map(lambda value: str(value).strip() == "")
        count = int(blank_mask.sum())
        if count > 0:
            errors.append(f"Blank values found in {column}: {count}")
    return errors


def validate_priority_values(df: pd.DataFrame) -> list[str]:
    allowed = {"high", "medium", "low"}
    invalid_rows = [
        str(index + 1)
        for index, value in enumerate(df["review_priority"].astype(str).str.lower())
        if value not in allowed
    ]
    if not invalid_rows:
        return []
    return [f"Invalid review_priority values at rows: {', '.join(invalid_rows)}"]


def validate_sample_bucket(df: pd.DataFrame) -> list[str]:
    allowed = {"trigger", "high", "medium", "low"}
    invalid_rows = [
        str(index + 1)
        for index, value in enumerate(df["sample_bucket"].astype(str).str.lower())
        if value not in allowed
    ]
    if not invalid_rows:
        return []
    return [f"Invalid sample_bucket values at rows: {', '.join(invalid_rows)}"]


def validate_no_duplicate_response_ids(df: pd.DataFrame) -> list[str]:
    duplicate_mask = df["response_id"].duplicated(keep=False)
    duplicates = df.loc[duplicate_mask, "response_id"].tolist()
    if not duplicates:
        return []
    joined = ", ".join(dict.fromkeys(duplicates))
    return [f"Duplicate response_id values found: {joined}"]


def run_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if len(df) == 0:
        return errors
    errors.extend(
        validate_required_text(
            df,
            ["response_id", "question_id", "sample_reason", "sample_bucket"],
        )
    )
    errors.extend(validate_priority_values(df))
    errors.extend(validate_sample_bucket(df))
    errors.extend(validate_no_duplicate_response_ids(df))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 06_review/review_samples.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to review_samples.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append validation results as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_csv(args.input)
    validate_required_columns(df, OUTPUT_COLUMNS)
    errors = run_validations(df)

    log_payload = {
        "event": "validate_review_samples",
        "input": str(args.input),
        "row_count": int(len(df)),
        "success": len(errors) == 0,
        "errors": errors,
        "created_at": utc_now_iso(),
    }
    if args.log is not None:
        append_jsonl(log_payload, args.log)

    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
