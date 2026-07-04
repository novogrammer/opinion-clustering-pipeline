from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns


EMBEDDING_REQUEST_COLUMNS = ["response_id", "question_id", "embedding_input_text"]


def validate_no_duplicate_response_ids(df: pd.DataFrame) -> list[str]:
    duplicate_mask = df["response_id"].duplicated(keep=False)
    duplicates = df.loc[duplicate_mask, "response_id"].tolist()
    if not duplicates:
        return []
    joined = ", ".join(dict.fromkeys(duplicates))
    return [f"Duplicate response_id values found: {joined}"]


def validate_no_blank_values(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for column in EMBEDDING_REQUEST_COLUMNS:
        blank_mask = df[column].map(lambda value: str(value).strip() == "")
        count = int(blank_mask.sum())
        if count > 0:
            errors.append(f"Blank values found in {column}: {count}")
    return errors


def run_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_no_duplicate_response_ids(df))
    errors.extend(validate_no_blank_values(df))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 03_embeddings/embedding_requests.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to embedding_requests.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append validation results as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_csv(args.input)
    validate_required_columns(df, EMBEDDING_REQUEST_COLUMNS)
    errors = run_validations(df)
    log_payload = {
        "event": "validate_embedding_requests",
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
