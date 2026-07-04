from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from clustering_common import SUMMARY_COLUMNS
from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns


def parse_representative_answers(value: object) -> list[str]:
    return [part.strip() for part in str(value).split(" || ") if part.strip()]


def validate_topic_ids(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for idx, value in enumerate(df["topic_id"], start=1):
        try:
            int(value)
        except (TypeError, ValueError):
            errors.append(f"Invalid topic_id at row {idx}: {value}")
    return errors


def validate_cluster_sizes(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for idx, value in enumerate(df["cluster_size"], start=1):
        try:
            cluster_size = int(value)
        except (TypeError, ValueError):
            errors.append(f"Invalid cluster_size at row {idx}: {value}")
            continue
        if cluster_size <= 0:
            errors.append(f"cluster_size must be greater than 0 at row {idx}: {value}")
    return errors


def validate_confidence_values(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for idx, value in enumerate(df["confidence"], start=1):
        if str(value).strip() == "":
            continue
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            errors.append(f"Invalid confidence at row {idx}: {value}")
            continue
        if confidence < 0 or confidence > 1:
            errors.append(f"Out-of-range confidence at row {idx}: {value}")
    return errors


def validate_required_text(df: pd.DataFrame, columns: list[str]) -> list[str]:
    errors: list[str] = []
    for column in columns:
        blank_mask = df[column].map(lambda value: str(value).strip() == "")
        count = int(blank_mask.sum())
        if count > 0:
            errors.append(f"Blank values found in {column}: {count}")
    return errors


def validate_representative_answers(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for idx, value in enumerate(df["representative_answers"], start=1):
        answers = parse_representative_answers(value)
        if not answers:
            errors.append(f"representative_answers is blank at row {idx}")
            continue
        if len(answers) != len(set(answers)):
            errors.append(f"representative_answers contains duplicates at row {idx}")
        if len(answers) > 5:
            errors.append(f"representative_answers has more than 5 entries at row {idx}")
    return errors


def validate_against_clusters(summary_df: pd.DataFrame, clusters_df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    cluster_counts = (
        clusters_df.groupby(["question_id", "topic_id"], sort=False)
        .size()
        .reset_index(name="actual_cluster_size")
    )
    merged = summary_df.merge(
        cluster_counts,
        on=["question_id", "topic_id"],
        how="outer",
        indicator=True,
    )
    for idx, row in merged.iterrows():
        merge_state = str(row["_merge"])
        if merge_state != "both":
            errors.append(f"topic pair missing between cluster_summary.csv and clusters.csv at merged row {idx + 1}")
            continue
        if int(row["cluster_size"]) != int(row["actual_cluster_size"]):
            errors.append(
                f"cluster_size does not match clusters.csv for question_id={row['question_id']} topic_id={row['topic_id']}"
            )
    return errors


def run_validations(df: pd.DataFrame, clusters_df: pd.DataFrame | None = None) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_required_text(df, ["question_id", "candidate_label", "representative_answers"]))
    errors.extend(validate_topic_ids(df))
    errors.extend(validate_cluster_sizes(df))
    errors.extend(validate_confidence_values(df))
    errors.extend(validate_representative_answers(df))
    if clusters_df is not None:
        errors.extend(validate_against_clusters(df, clusters_df))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 04_clustering/cluster_summary.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to cluster_summary.csv")
    parser.add_argument("--clusters", type=Path, default=None, help="Optional path to clusters.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append validation results as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_csv(args.input)
    validate_required_columns(df, SUMMARY_COLUMNS)
    clusters_df = read_csv(args.clusters) if args.clusters is not None else None
    errors = run_validations(df, clusters_df=clusters_df)
    log_payload = {
        "event": "validate_cluster_summary",
        "input": str(args.input),
        "clusters": None if args.clusters is None else str(args.clusters),
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
