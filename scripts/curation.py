from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns, write_csv, write_json
from screening import SCREENED_COLUMNS, run_validations as run_screened_validations
from clustering import CLUSTER_COLUMNS, run_cluster_validations


REPRESENTATIVE_COLUMNS = [
    "topic_id",
    "topic_size",
    "response_id",
    "question_id",
    "answer_text",
    "topic_probability",
    "representative_rank",
]
CURATION_METADATA_KEYS = [
    "created_at",
    "question_id",
    "row_count",
    "cluster_count",
    "input_screened_path",
    "input_clusters_path",
]


def validate_required_text(df: pd.DataFrame, columns: list[str]) -> list[str]:
    errors: list[str] = []
    for column in columns:
        blank_mask = df[column].map(lambda value: str(value).strip() == "")
        count = int(blank_mask.sum())
        if count > 0:
            errors.append(f"Blank values found in {column}: {count}")
    return errors


def validate_no_duplicate_response_ids(df: pd.DataFrame) -> list[str]:
    duplicate_mask = df["response_id"].duplicated(keep=False)
    duplicates = df.loc[duplicate_mask, "response_id"].tolist()
    if not duplicates:
        return []
    joined = ", ".join(dict.fromkeys(duplicates))
    return [f"Duplicate response_id values found: {joined}"]


def run_representative_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if len(df) == 0:
        return []
    errors.extend(validate_no_duplicate_response_ids(df))
    errors.extend(validate_required_text(df, ["topic_id", "topic_size", "response_id", "question_id", "answer_text", "representative_rank"]))
    return errors


def build_representatives_df(clusters_df: pd.DataFrame, target_rows: pd.DataFrame, per_topic: int = 3) -> pd.DataFrame:
    if len(clusters_df) == 0:
        return pd.DataFrame(columns=REPRESENTATIVE_COLUMNS)
    merged = clusters_df.merge(
        target_rows[["response_id", "question_id", "answer_text"]],
        on=["response_id", "question_id"],
        how="left",
    )
    representative_frames: list[pd.DataFrame] = []
    for topic_id, group in merged.groupby("topic_id", sort=True):
        ordered = group.sort_values(
            by=["topic_probability", "response_id"],
            ascending=[False, True],
            na_position="last",
        ).head(per_topic).copy()
        ordered["topic_size"] = int(len(group))
        ordered["representative_rank"] = range(1, len(ordered) + 1)
        representative_frames.append(ordered[REPRESENTATIVE_COLUMNS])
    if not representative_frames:
        return pd.DataFrame(columns=REPRESENTATIVE_COLUMNS)
    return pd.concat(representative_frames, ignore_index=True)


def build_metadata_payload(
    *,
    question_id: str,
    row_count: int,
    cluster_count: int,
    screened_path: Path,
    clusters_path: Path,
) -> dict[str, object]:
    return {
        "created_at": utc_now_iso(),
        "question_id": question_id,
        "row_count": row_count,
        "cluster_count": cluster_count,
        "input_screened_path": str(screened_path),
        "input_clusters_path": str(clusters_path),
    }


def run_curation_metadata_validations(payload: dict[str, object], representatives_path: Path) -> list[str]:
    errors: list[str] = []
    for key in CURATION_METADATA_KEYS:
        if key not in payload:
            errors.append(f"Missing key: {key}")
    if payload.get("question_id", "") == "":
        errors.append("question_id must not be blank")
    if not isinstance(payload.get("row_count"), int) or payload["row_count"] < 0:
        errors.append("row_count must be a non-negative integer")
    if not isinstance(payload.get("cluster_count"), int) or payload["cluster_count"] < 0:
        errors.append("cluster_count must be a non-negative integer")
    if not representatives_path.exists():
        errors.append("cluster_representatives.csv must exist when writing curation metadata")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare curation artifacts for one question.")
    parser.add_argument("--input", required=True, type=Path, help="Path to screened_responses.csv")
    parser.add_argument("--clusters", required=True, type=Path, help="Path to 04_clustering/clusters.csv")
    parser.add_argument("--question-id", required=True, help="Target question_id")
    parser.add_argument("--output-dir", required=True, type=Path, help="Path to 05_curation directory")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    responses_df = read_csv(args.input)
    validate_required_columns(responses_df, SCREENED_COLUMNS)
    screened_errors = run_screened_validations(responses_df)
    if screened_errors:
        raise SystemExit("\n".join(screened_errors))

    clusters_df = read_csv(args.clusters)
    validate_required_columns(clusters_df, CLUSTER_COLUMNS)
    cluster_errors = run_cluster_validations(clusters_df)
    if cluster_errors:
        raise SystemExit("\n".join(cluster_errors))

    target_rows = responses_df[
        (responses_df["question_id"] == args.question_id)
        & (responses_df["is_target"].astype(str).str.lower() == "true")
    ].copy()
    if len(target_rows) != len(clusters_df):
        raise SystemExit("clusters.csv row count does not match screened target row count")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    representatives_path = args.output_dir / "cluster_representatives.csv"
    metadata_path = args.output_dir / "curation_metadata.json"

    representatives_df = build_representatives_df(clusters_df, target_rows)
    representative_errors = run_representative_validations(representatives_df)
    if representative_errors:
        raise SystemExit("\n".join(representative_errors))

    write_csv(representatives_df, representatives_path)

    metadata_payload = build_metadata_payload(
        question_id=args.question_id,
        row_count=int(len(representatives_df)),
        cluster_count=int(clusters_df["topic_id"].nunique()),
        screened_path=args.input,
        clusters_path=args.clusters,
    )
    metadata_errors = run_curation_metadata_validations(metadata_payload, representatives_path)
    if metadata_errors:
        raise SystemExit("\n".join(metadata_errors))
    write_json(metadata_payload, metadata_path)
    if args.log is not None:
        append_jsonl(
            {
                "event": "curation",
                "input": str(args.input),
                "clusters": str(args.clusters),
                "question_id": args.question_id,
                "output_dir": str(args.output_dir),
                "row_count": int(len(representatives_df)),
                "cluster_count": int(clusters_df["topic_id"].nunique()),
                "created_at": utc_now_iso(),
            },
            args.log,
        )


if __name__ == "__main__":
    main()
