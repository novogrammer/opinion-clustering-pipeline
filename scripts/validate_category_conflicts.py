from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from category_master_conflicts import OUTPUT_COLUMNS
from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns


def validate_conflict_level(df: pd.DataFrame) -> list[str]:
    allowed = {"high", "medium", "low"}
    invalid_rows = [
        str(index + 1)
        for index, value in enumerate(df["conflict_level"].astype(str).str.lower())
        if value not in allowed
    ]
    if not invalid_rows:
        return []
    return [f"Invalid conflict_level values at rows: {', '.join(invalid_rows)}"]


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


def validate_required_text(df: pd.DataFrame, columns: list[str]) -> list[str]:
    errors: list[str] = []
    for column in columns:
        blank_mask = df[column].map(lambda value: str(value).strip() == "")
        count = int(blank_mask.sum())
        if count > 0:
            errors.append(f"Blank values found in {column}: {count}")
    return errors


def validate_pair_uniqueness(df: pd.DataFrame) -> list[str]:
    duplicate_mask = df[["left_category_id", "right_category_id"]].duplicated(keep=False)
    if not duplicate_mask.any():
        return []
    duplicates = df.loc[duplicate_mask, ["left_category_id", "right_category_id"]]
    pairs = [f"{left}-{right}" for left, right in duplicates.itertuples(index=False, name=None)]
    joined = ", ".join(dict.fromkeys(pairs))
    return [f"Duplicate category pairs found: {joined}"]


def run_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if len(df) == 0:
        return errors
    errors.extend(
        validate_required_text(
            df,
            ["left_category_id", "left_category_name", "right_category_id", "right_category_name", "conflict_level"],
        )
    )
    errors.extend(validate_boolean_column(df, "name_overlap"))
    errors.extend(validate_conflict_level(df))
    errors.extend(validate_pair_uniqueness(df))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 05_classification/category_conflicts.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to category_conflicts.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append validation results as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_csv(args.input)
    validate_required_columns(df, OUTPUT_COLUMNS)
    errors = run_validations(df)

    log_payload = {
        "event": "validate_category_conflicts",
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
