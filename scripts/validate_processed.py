from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import REQUIRED_RESPONSE_COLUMNS, append_jsonl, read_csv, utc_now_iso, validate_required_columns


def normalize_whitespace(value: str) -> str:
    return " ".join(str(value).replace("\t", " ").split())


def validate_no_duplicate_response_ids(df: pd.DataFrame) -> list[str]:
    duplicate_mask = df["response_id"].duplicated(keep=False)
    duplicates = df.loc[duplicate_mask, "response_id"].tolist()
    if not duplicates:
        return []
    joined = ", ".join(dict.fromkeys(duplicates))
    return [f"Duplicate response_id values found: {joined}"]


def validate_no_blank_core_values(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for column in ["response_id", "question_id", "question_text"]:
        blank_mask = df[column].map(lambda value: str(value).strip() == "")
        count = int(blank_mask.sum())
        if count > 0:
            errors.append(f"Blank values found in {column}: {count}")
    return errors


def validate_normalized_answer_text(df: pd.DataFrame) -> list[str]:
    non_normalized = df["answer_text"].map(lambda value: str(value) != normalize_whitespace(value))
    count = int(non_normalized.sum())
    if count == 0:
        return []
    return [f"Non-normalized answer_text values found: {count}"]


def validate_question_ids(df: pd.DataFrame) -> list[str]:
    blank_mask = df["question_id"].map(lambda value: str(value).strip() == "")
    if int(blank_mask.sum()) == 0:
        return []
    return ["question_id contains blank values"]


def run_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_no_duplicate_response_ids(df))
    errors.extend(validate_no_blank_core_values(df))
    errors.extend(validate_normalized_answer_text(df))
    errors.extend(validate_question_ids(df))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 01_processed/responses_normalized.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to responses_normalized.csv")
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional path to append validation results as JSONL",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_csv(args.input)
    validate_required_columns(df, REQUIRED_RESPONSE_COLUMNS)
    errors = run_validations(df)

    log_payload = {
        "event": "validate_processed",
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
