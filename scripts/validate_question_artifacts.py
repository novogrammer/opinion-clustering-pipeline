from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from classification import (
    CATEGORY_MASTER_COLUMNS,
    FINAL_LABEL_COLUMNS,
    run_category_master_validations,
    run_final_label_validations,
)
from clustering import (
    CLUSTER_COLUMNS,
    run_cluster_validations,
    run_clustering_metadata_validations,
)
from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns
from embeddings import (
    FAILURE_COLUMNS,
    run_embedding_metadata_validations,
    run_embeddings_array_validations,
    run_failure_validations,
)
from review_prep import REVIEW_COLUMNS, run_review_log_validations
from validate_jsonl_log import run_validations as run_jsonl_log_validations


def validate_question_id_column(df, expected_question_id: str, label: str) -> list[str]:
    if "question_id" not in df.columns:
        return []
    actual_question_ids = {str(value).strip() for value in df["question_id"].tolist()}
    if actual_question_ids == {expected_question_id}:
        return []
    return [f"{label}: question_id values do not match directory name {expected_question_id}"]


def collect_question_errors(question_dir: Path) -> list[str]:
    expected_question_id = question_dir.name
    project_dir = question_dir.parent.parent
    errors: list[str] = []

    embeddings_dir = question_dir / "03_embeddings"
    clustering_dir = question_dir / "04_clustering"
    classification_dir = question_dir / "05_classification"
    review_dir = question_dir / "06_review"

    embeddings_path = embeddings_dir / "embeddings.npy"
    embedding_metadata_path = embeddings_dir / "embedding_metadata.json"
    embedding_failures_path = embeddings_dir / "embedding_failures.csv"
    clusters_path = clustering_dir / "clusters.csv"
    clustering_metadata_path = clustering_dir / "clustering_metadata.json"
    category_master_path = classification_dir / "category_master.csv"
    final_labels_path = classification_dir / "final_labels.csv"
    review_log_path = review_dir / "review_log.csv"
    screened_path = project_dir / "02_screening" / "screened_responses.csv"

    stage_03_started = any(path.exists() for path in [embeddings_path, embedding_metadata_path, embedding_failures_path, embeddings_dir / "embedding.log"])
    stage_04_started = any(path.exists() for path in [clusters_path, clustering_metadata_path, clustering_dir / "clustering.log"])
    stage_05_started = any(path.exists() for path in [category_master_path, final_labels_path, classification_dir / "classification.log"])
    stage_06_started = any(path.exists() for path in [review_log_path, review_dir / "review.log"])

    if stage_03_started or stage_04_started or stage_05_started or stage_06_started:
        if not embeddings_path.exists():
            errors.append(f"Missing file: embeddings.npy ({embeddings_path})")
        if not embedding_metadata_path.exists():
            errors.append(f"Missing file: embedding_metadata.json ({embedding_metadata_path})")
    if stage_04_started or stage_05_started or stage_06_started:
        if not clusters_path.exists():
            errors.append(f"Missing file: clusters.csv ({clusters_path})")
        if not clustering_metadata_path.exists():
            errors.append(f"Missing file: clustering_metadata.json ({clustering_metadata_path})")
    if stage_05_started or stage_06_started:
        if not category_master_path.exists():
            errors.append(f"Missing file: category_master.csv ({category_master_path})")
        if not final_labels_path.exists():
            errors.append(f"Missing file: final_labels.csv ({final_labels_path})")
    if stage_06_started and not review_log_path.exists():
        errors.append(f"Missing file: review_log.csv ({review_log_path})")

    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
        metadata_payload = json.loads(embedding_metadata_path.read_text(encoding="utf-8")) if embedding_metadata_path.exists() else None
        errors.extend(
            f"embeddings.npy: {message}"
            for message in run_embeddings_array_validations(
                embeddings,
                metadata_row_count=None if metadata_payload is None else metadata_payload.get("row_count"),
                metadata_status=None if metadata_payload is None else metadata_payload.get("status"),
            )
        )

    if embedding_metadata_path.exists():
        payload = json.loads(embedding_metadata_path.read_text(encoding="utf-8"))
        errors.extend(
            f"embedding_metadata.json: {message}"
            for message in run_embedding_metadata_validations(
                payload,
                screened_path=screened_path if screened_path.exists() else None,
                embeddings_path=embeddings_path if embeddings_path.exists() else None,
                failures_path=embedding_failures_path if embedding_failures_path.exists() else None,
            )
        )
        if str(payload.get("question_id", "")).strip() != expected_question_id:
            errors.append(f"embedding_metadata.json: question_id does not match directory name {expected_question_id}")

    if embedding_failures_path.exists():
        failures_df = read_csv(embedding_failures_path)
        validate_required_columns(failures_df, FAILURE_COLUMNS)
        errors.extend(f"embedding_failures.csv: {message}" for message in run_failure_validations(failures_df))
        errors.extend(validate_question_id_column(failures_df, expected_question_id, "embedding_failures.csv"))

    if clusters_path.exists():
        clusters_df = read_csv(clusters_path)
        validate_required_columns(clusters_df, CLUSTER_COLUMNS)
        errors.extend(f"clusters.csv: {message}" for message in run_cluster_validations(clusters_df))
        errors.extend(validate_question_id_column(clusters_df, expected_question_id, "clusters.csv"))

    if clustering_metadata_path.exists():
        payload = json.loads(clustering_metadata_path.read_text(encoding="utf-8"))
        errors.extend(
            f"clustering_metadata.json: {message}"
            for message in run_clustering_metadata_validations(
                payload,
                screened_path=screened_path if screened_path.exists() else None,
                embeddings_path=embeddings_path if embeddings_path.exists() else None,
                clusters_path=clusters_path if clusters_path.exists() else None,
            )
        )

    if category_master_path.exists():
        category_master_df = read_csv(category_master_path)
        validate_required_columns(category_master_df, CATEGORY_MASTER_COLUMNS)
        errors.extend(f"category_master.csv: {message}" for message in run_category_master_validations(category_master_df))

    if final_labels_path.exists():
        final_labels_df = read_csv(final_labels_path)
        validate_required_columns(final_labels_df, FINAL_LABEL_COLUMNS)
        errors.extend(f"final_labels.csv: {message}" for message in run_final_label_validations(final_labels_df))
        errors.extend(validate_question_id_column(final_labels_df, expected_question_id, "final_labels.csv"))

    if review_log_path.exists():
        review_log_df = read_csv(review_log_path)
        validate_required_columns(review_log_df, REVIEW_COLUMNS)
        errors.extend(f"review_log.csv: {message}" for message in run_review_log_validations(review_log_df))
        errors.extend(validate_question_id_column(review_log_df, expected_question_id, "review_log.csv"))

    for log_path in [
        embeddings_dir / "embedding.log",
        clustering_dir / "clustering.log",
        classification_dir / "classification.log",
        review_dir / "review.log",
    ]:
        if not log_path.exists():
            continue
        log_errors = run_jsonl_log_validations(log_path.read_text(encoding="utf-8").splitlines())
        errors.extend(f"{log_path.name}: {message}" for message in log_errors)

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate all question-level artifacts under questions/{question_id}")
    parser.add_argument("--question-dir", required=True, type=Path, help="Path to projects/{project_name}/questions/{question_id}")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append validation results as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question_dir = args.question_dir
    errors = collect_question_errors(question_dir)

    log_payload = {
        "event": "validate_question_artifacts",
        "question_dir": str(question_dir),
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
