from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from classification_common import CATEGORY_MASTER_COLUMNS
from classification_keywords import build_categories_df
from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns


def validate_no_duplicate_category_ids(df: pd.DataFrame) -> list[str]:
    duplicate_mask = df["category_id"].duplicated(keep=False)
    duplicates = df.loc[duplicate_mask, "category_id"].tolist()
    if not duplicates:
        return []
    joined = ", ".join(dict.fromkeys(duplicates))
    return [f"Duplicate category_id values found: {joined}"]


def validate_no_blank_core_values(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for column in ["category_id", "category_name", "category_definition"]:
        blank_mask = df[column].map(lambda value: str(value).strip() == "")
        count = int(blank_mask.sum())
        if count > 0:
            errors.append(f"Blank values found in {column}: {count}")
    return errors


def validate_keyword_coverage(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    categories = build_categories_df(df)
    for category in categories:
        if len(category["keywords"]) == 0:
            errors.append(f"Category has no usable keywords: {category['category_id']}")
    return errors


def run_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_no_duplicate_category_ids(df))
    errors.extend(validate_no_blank_core_values(df))
    errors.extend(validate_keyword_coverage(df))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 05_classification/category_master.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to category_master.csv")
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
    validate_required_columns(df, CATEGORY_MASTER_COLUMNS)
    errors = run_validations(df)

    log_payload = {
        "event": "validate_category_master",
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
