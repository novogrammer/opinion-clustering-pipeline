from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns
from override_common import OVERRIDE_CANDIDATE_COLUMNS as OUTPUT_COLUMNS
from validate_override_rules import run_validations as run_override_rule_validations


def validate_required_text(df: pd.DataFrame, columns: list[str]) -> list[str]:
    errors: list[str] = []
    for column in columns:
        blank_mask = df[column].map(lambda value: str(value).strip() == "")
        count = int(blank_mask.sum())
        if count > 0:
            errors.append(f"Blank values found in {column}: {count}")
    return errors


def validate_correction_type(df: pd.DataFrame) -> list[str]:
    allowed = {"fallback_to_category", "category_change"}
    invalid_rows = [
        str(index + 1)
        for index, value in enumerate(df["source_correction_type"].astype(str).str.lower())
        if value not in allowed
    ]
    if not invalid_rows:
        return []
    return [f"Invalid source_correction_type values at rows: {', '.join(invalid_rows)}"]


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
    errors = run_override_rule_validations(df)
    if len(df) == 0:
        return errors
    errors.extend(
        validate_required_text(
            df,
            ["approved", "source_response_id", "source_predicted_category_id", "source_predicted_category_name", "source_correction_type"],
        )
    )
    errors.extend(validate_boolean_column(df, "approved"))
    errors.extend(validate_correction_type(df))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 05_classification/manual_override_candidates.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to manual_override_candidates.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append validation results as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_csv(args.input)
    validate_required_columns(df, OUTPUT_COLUMNS)
    errors = run_validations(df)

    log_payload = {
        "event": "validate_override_candidates",
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
