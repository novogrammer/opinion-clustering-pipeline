from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns
from review_common import REVIEW_COLUMNS

ALLOWED_REVIEW_TRIGGER_TOKENS = {
    "fallback_category",
    "other_category",
    "needs_human_review",
    "low_confidence",
    "short_answer",
    "negation",
    "ambiguous_match",
    "multi_topic",
    "pii_detected",
    "aggressive_expression",
    "duplicate_response",
    "typical_match",
}


def validate_status_values(df: pd.DataFrame) -> list[str]:
    allowed = {"pending", "reviewed", "skipped"}
    invalid_rows = [
        str(index + 1)
        for index, value in enumerate(df["review_status"].astype(str).str.lower())
        if value not in allowed
    ]
    if not invalid_rows:
        return []
    return [f"Invalid review_status values at rows: {', '.join(invalid_rows)}"]


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


def validate_reviewed_rows(df: pd.DataFrame) -> list[str]:
    reviewed_mask = df["review_status"].astype(str).str.lower() == "reviewed"
    if int(reviewed_mask.sum()) == 0:
        return []

    errors: list[str] = []
    reviewed = df.loc[reviewed_mask]
    for column in ["reviewed_category_id", "reviewed_category_name", "reviewer", "reviewed_at"]:
        blank_mask = reviewed[column].map(lambda value: str(value).strip() == "")
        count = int(blank_mask.sum())
        if count > 0:
            errors.append(f"Blank values found in {column} for reviewed rows: {count}")
    return errors


def validate_skipped_rows(df: pd.DataFrame) -> list[str]:
    skipped_mask = df["review_status"].astype(str).str.lower() == "skipped"
    if int(skipped_mask.sum()) == 0:
        return []

    skipped = df.loc[skipped_mask]
    blank_trigger_mask = skipped["review_trigger"].map(lambda value: str(value).strip() == "")
    if int(blank_trigger_mask.sum()) > 0:
        return ["Blank review_trigger found for skipped rows"]
    return []


def validate_review_trigger_tokens(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for idx, value in enumerate(df["review_trigger"].astype(str), start=1):
        tokens = [token.strip() for token in value.split("|") if token.strip()]
        if not tokens:
            errors.append(f"Blank review_trigger at row {idx}")
            continue
        invalid_tokens = [token for token in tokens if token not in ALLOWED_REVIEW_TRIGGER_TOKENS]
        if invalid_tokens:
            errors.append(f"Invalid review_trigger tokens at row {idx}: {', '.join(invalid_tokens)}")
    return errors


def validate_no_duplicate_response_ids(df: pd.DataFrame) -> list[str]:
    duplicate_mask = df["response_id"].duplicated(keep=False)
    duplicates = df.loc[duplicate_mask, "response_id"].tolist()
    if not duplicates:
        return []
    joined = ", ".join(dict.fromkeys(duplicates))
    return [f"Duplicate response_id values found: {joined}"]


def validate_required_text(df: pd.DataFrame, columns: list[str]) -> list[str]:
    errors: list[str] = []
    for column in columns:
        blank_mask = df[column].map(lambda value: str(value).strip() == "")
        count = int(blank_mask.sum())
        if count > 0:
            errors.append(f"Blank values found in {column}: {count}")
    return errors


def run_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_no_duplicate_response_ids(df))
    errors.extend(validate_priority_values(df))
    errors.extend(validate_status_values(df))
    errors.extend(validate_boolean_column(df, "needs_human_review"))
    errors.extend(validate_review_trigger_tokens(df))
    errors.extend(
        validate_required_text(
            df,
            ["response_id", "question_id", "answer_text", "predicted_category_id", "predicted_category_name", "review_priority", "review_trigger", "review_status"],
        )
    )
    errors.extend(validate_reviewed_rows(df))
    errors.extend(validate_skipped_rows(df))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 06_review/review_log.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to review_log.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append validation results as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_csv(args.input)
    validate_required_columns(df, REVIEW_COLUMNS)
    errors = run_validations(df)

    log_payload = {
        "event": "validate_review_log",
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
