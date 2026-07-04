from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns
from duplicate_common import DUPLICATE_OUTPUT_COLUMNS, normalize_answer_text


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


def validate_duplicate_count(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for idx, value in enumerate(df["duplicate_count"], start=1):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            errors.append(f"Invalid duplicate_count at row {idx}: {value}")
            continue
        if parsed <= 1:
            errors.append(f"duplicate_count must be greater than 1 at row {idx}: {value}")
    return errors


def validate_group_consistency(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for group_id, group in df.groupby("duplicate_group_id", sort=False):
        canonical_rows = group[group["is_canonical"].astype(str).str.lower() == "true"]
        if len(canonical_rows) != 1:
            errors.append(f"{group_id}: expected exactly one canonical row")
        stated_count = {int(value) for value in group["duplicate_count"].astype(int).tolist()}
        if len(stated_count) != 1:
            errors.append(f"{group_id}: duplicate_count values differ inside group")
        elif next(iter(stated_count)) != len(group):
            errors.append(f"{group_id}: duplicate_count does not match actual group size")
        canonical_ids = set(group["canonical_response_id"].astype(str).tolist())
        if len(canonical_ids) != 1:
            errors.append(f"{group_id}: canonical_response_id values differ inside group")
    return errors


def validate_normalized_answer_consistency(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for group_id, group in df.groupby("duplicate_group_id", sort=False):
        normalized_answers = {normalize_answer_text(value) for value in group["answer_text"].astype(str).tolist()}
        if len(normalized_answers) != 1:
            errors.append(f"{group_id}: answer_text values do not normalize to one value")
    return errors


def run_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if len(df) == 0:
        return errors
    errors.extend(
        validate_required_text(
            df,
            ["duplicate_group_id", "question_id", "response_id", "canonical_response_id", "duplicate_count", "is_canonical", "answer_text"],
        )
    )
    errors.extend(validate_boolean_column(df, "is_canonical"))
    errors.extend(validate_duplicate_count(df))
    errors.extend(validate_group_consistency(df))
    errors.extend(validate_normalized_answer_consistency(df))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 02_screening/duplicate_responses.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to duplicate_responses.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append validation results as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_csv(args.input)
    validate_required_columns(df, DUPLICATE_OUTPUT_COLUMNS)
    errors = run_validations(df)
    log_payload = {
        "event": "validate_duplicate_responses",
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
