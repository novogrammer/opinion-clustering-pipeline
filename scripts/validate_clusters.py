from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from clustering_common import CLUSTER_COLUMNS
from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns


def validate_no_duplicate_response_ids(df: pd.DataFrame) -> list[str]:
    duplicate_mask = df["response_id"].duplicated(keep=False)
    duplicates = df.loc[duplicate_mask, "response_id"].tolist()
    if not duplicates:
        return []
    joined = ", ".join(dict.fromkeys(duplicates))
    return [f"Duplicate response_id values found: {joined}"]


def validate_probability_values(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for idx, value in enumerate(df["topic_probability"], start=1):
        if str(value).strip() == "":
            continue
        try:
            probability = float(value)
        except (TypeError, ValueError):
            errors.append(f"Invalid topic_probability at row {idx}: {value}")
            continue
        if probability < 0 or probability > 1:
            errors.append(f"Out-of-range topic_probability at row {idx}: {value}")
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


def validate_topic_outlier_consistency(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for idx, row in df.iterrows():
        topic_id = str(row["topic_id"])
        is_outlier = str(row["is_outlier"]).lower() == "true"
        if is_outlier and topic_id != "-1":
            errors.append(f"Row {idx + 1}: is_outlier=true requires topic_id=-1")
        if not is_outlier and topic_id == "-1":
            errors.append(f"Row {idx + 1}: topic_id=-1 requires is_outlier=true")
    return errors


def run_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_no_duplicate_response_ids(df))
    errors.extend(validate_probability_values(df))
    errors.extend(validate_boolean_column(df, "is_outlier"))
    errors.extend(validate_topic_outlier_consistency(df))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 04_clustering/clusters.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to clusters.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append validation results as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_csv(args.input)
    validate_required_columns(df, CLUSTER_COLUMNS)
    errors = run_validations(df)
    log_payload = {
        "event": "validate_clusters",
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
