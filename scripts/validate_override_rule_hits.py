from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns
from override_common import OVERRIDE_RULE_HIT_COLUMNS as OUTPUT_COLUMNS


def validate_required_text(df: pd.DataFrame, columns: list[str]) -> list[str]:
    errors: list[str] = []
    for column in columns:
        blank_mask = df[column].map(lambda value: str(value).strip() == "")
        count = int(blank_mask.sum())
        if count > 0:
            errors.append(f"Blank values found in {column}: {count}")
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


def validate_priority(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for idx, value in enumerate(df["priority"], start=1):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            errors.append(f"Invalid priority at row {idx}: {value}")
            continue
        if parsed < 0:
            errors.append(f"Negative priority at row {idx}: {value}")
    return errors


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
            ["response_id", "question_id", "answer_text", "rule_id", "match_type", "pattern", "override_category_id", "override_category_name", "needs_human_review", "priority"],
        )
    )
    errors.extend(validate_boolean_column(df, "needs_human_review"))
    errors.extend(validate_priority(df))
    errors.extend(validate_no_duplicate_response_ids(df))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 05_classification/override_rule_hits.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to override_rule_hits.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append validation results as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_csv(args.input)
    validate_required_columns(df, OUTPUT_COLUMNS)
    errors = run_validations(df)

    log_payload = {
        "event": "validate_override_rule_hits",
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
