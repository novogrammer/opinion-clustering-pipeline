from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns
from review_common import REVIEW_SUMMARY_COLUMNS as SUMMARY_COLUMNS


COUNT_COLUMNS = [
    "total_count",
    "reviewed_count",
    "corrected_count",
    "pending_count",
    "skipped_count",
    "high_priority_count",
    "medium_priority_count",
    "low_priority_count",
]


def validate_integer_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    errors: list[str] = []
    for column in columns:
        for idx, value in enumerate(df[column], start=1):
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                errors.append(f"Invalid {column} at row {idx}: {value}")
                continue
            if parsed < 0:
                errors.append(f"Negative {column} at row {idx}: {value}")
    return errors


def validate_correction_rate(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for idx, value in enumerate(df["correction_rate"], start=1):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            errors.append(f"Invalid correction_rate at row {idx}: {value}")
            continue
        if parsed < 0 or parsed > 1:
            errors.append(f"Out-of-range correction_rate at row {idx}: {value}")
    return errors


def validate_required_text(df: pd.DataFrame, columns: list[str]) -> list[str]:
    errors: list[str] = []
    for column in columns:
        blank_mask = df[column].map(lambda value: str(value).strip() == "")
        count = int(blank_mask.sum())
        if count > 0:
            errors.append(f"Blank values found in {column}: {count}")
    return errors


def validate_row_consistency(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for idx, row in df.iterrows():
        total_count = int(row["total_count"])
        reviewed_count = int(row["reviewed_count"])
        pending_count = int(row["pending_count"])
        skipped_count = int(row["skipped_count"])
        priority_total = int(row["high_priority_count"]) + int(row["medium_priority_count"]) + int(row["low_priority_count"])
        corrected_count = int(row["corrected_count"])
        if reviewed_count + pending_count + skipped_count != total_count:
            errors.append(f"Status counts do not match total_count at row {idx + 1}")
        if priority_total != total_count:
            errors.append(f"Priority counts do not match total_count at row {idx + 1}")
        if corrected_count > reviewed_count:
            errors.append(f"corrected_count exceeds reviewed_count at row {idx + 1}")
    return errors


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


def run_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    errors.extend(
        validate_required_text(
            df,
            ["question_id", "predicted_category_id", "predicted_category_name", "needs_definition_review", "definition_review_reason"],
        )
    )
    errors.extend(validate_integer_columns(df, COUNT_COLUMNS))
    errors.extend(validate_correction_rate(df))
    errors.extend(validate_boolean_column(df, "needs_definition_review"))
    errors.extend(validate_row_consistency(df))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 06_review/review_summary.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to review_summary.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append validation results as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_csv(args.input)
    validate_required_columns(df, SUMMARY_COLUMNS)
    errors = run_validations(df)

    log_payload = {
        "event": "validate_review_summary",
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
