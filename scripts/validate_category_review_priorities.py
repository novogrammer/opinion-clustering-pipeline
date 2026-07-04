from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns
from review_common import CATEGORY_REVIEW_PRIORITY_COLUMNS as OUTPUT_COLUMNS


INTEGER_COLUMNS = [
    "corrected_count",
    "high_priority_count",
    "conflict_pair_count",
    "high_conflict_pair_count",
    "priority_rank",
]

FLOAT_COLUMNS = [
    "correction_rate",
    "priority_score",
]


def validate_boolean_column(df: pd.DataFrame, column: str) -> list[str]:
    allowed = {"true", "false"}
    invalid_rows = [
        str(index + 1)
        for index, value in enumerate(df[column].astype(str).str.lower())
        if value not in allowed
    ]
    if not invalid_rows:
        return []
    return [f"Invalid boolean values in {column} at rows: {', '.join(invalid_rows)}"]


def validate_integer_columns(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for column in INTEGER_COLUMNS:
        for idx, value in enumerate(df[column], start=1):
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                errors.append(f"Invalid {column} at row {idx}: {value}")
                continue
            if parsed < 0:
                errors.append(f"Negative {column} at row {idx}: {value}")
    return errors


def validate_float_columns(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for column in FLOAT_COLUMNS:
        for idx, value in enumerate(df[column], start=1):
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                errors.append(f"Invalid {column} at row {idx}: {value}")
                continue
            if parsed < 0:
                errors.append(f"Negative {column} at row {idx}: {value}")
    return errors


def validate_required_text(df: pd.DataFrame, columns: list[str]) -> list[str]:
    errors: list[str] = []
    for column in columns:
        blank_mask = df[column].map(lambda value: str(value).strip() == "")
        count = int(blank_mask.sum())
        if count > 0:
            errors.append(f"Blank values found in {column}: {count}")
    return errors


def validate_rank_sequence(df: pd.DataFrame) -> list[str]:
    if len(df) == 0:
        return []
    actual = [int(value) for value in df["priority_rank"].tolist()]
    expected = list(range(1, len(actual) + 1))
    if actual == expected:
        return []
    return ["priority_rank is not a contiguous sequence starting from 1"]


def run_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if len(df) == 0:
        return errors
    errors.extend(
        validate_required_text(
            df,
            ["question_id", "category_id", "category_name", "needs_definition_review", "definition_review_reason"],
        )
    )
    errors.extend(validate_boolean_column(df, "needs_definition_review"))
    errors.extend(validate_integer_columns(df))
    errors.extend(validate_float_columns(df))
    errors.extend(validate_rank_sequence(df))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 06_review/category_review_priorities.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to category_review_priorities.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append validation results as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_csv(args.input)
    validate_required_columns(df, OUTPUT_COLUMNS)
    errors = run_validations(df)

    log_payload = {
        "event": "validate_category_review_priorities",
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
