from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from classification_common import CATEGORY_MASTER_COLUMNS, FINAL_LABEL_COLUMNS, OVERRIDE_RULE_COLUMNS
from category_master_conflicts import OUTPUT_COLUMNS as CATEGORY_CONFLICT_COLUMNS
from category_review_priorities import OUTPUT_COLUMNS as CATEGORY_REVIEW_PRIORITY_COLUMNS
from clustering_common import CLUSTER_COLUMNS, SUMMARY_COLUMNS
from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns
from duplicate_common import DUPLICATE_OUTPUT_COLUMNS as DUPLICATE_COLUMNS
from override_rule_hits import OUTPUT_COLUMNS as OVERRIDE_RULE_HIT_COLUMNS
from override_rule_summary import OUTPUT_COLUMNS as OVERRIDE_RULE_SUMMARY_COLUMNS
from review_common import REVIEW_COLUMNS, REVIEW_CORRECTION_COLUMNS, REVIEW_SAMPLE_COLUMNS, REVIEW_SUMMARY_COLUMNS
from review_override_candidates import OUTPUT_COLUMNS as OVERRIDE_CANDIDATE_COLUMNS
from validate_category_conflicts import run_validations as run_category_conflict_validations
from validate_category_master import run_validations as run_category_master_validations
from validate_category_review_priorities import run_validations as run_category_review_priority_validations
from validate_cluster_summary import run_validations as run_cluster_summary_validations
from validate_clustering_metadata import run_validations as run_clustering_metadata_validations
from validate_clusters import run_validations as run_clusters_validations
from validate_duplicate_responses import run_validations as run_duplicate_validations
from validate_embedding_failures import EMBEDDING_FAILURE_COLUMNS, run_validations as run_embedding_failure_validations
from validate_embedding_metadata import run_validations as run_embedding_metadata_validations
from validate_embedding_requests import EMBEDDING_REQUEST_COLUMNS, run_validations as run_embedding_request_validations
from validate_embeddings_array import run_validations as run_embeddings_array_validations
from validate_final_labels import run_validations as run_final_labels_validations
from validate_jsonl_log import run_validations as run_jsonl_log_validations
from validate_override_candidates import run_validations as run_override_candidate_validations
from validate_override_rule_hits import run_validations as run_override_rule_hit_validations
from validate_override_rule_summary import run_validations as run_override_rule_summary_validations
from validate_override_rules import run_validations as run_override_rule_validations
from validate_review_corrections import run_validations as run_review_correction_validations
from validate_review_log import run_validations as run_review_log_validations
from validate_review_samples import run_validations as run_review_sample_validations
from validate_review_summary import run_validations as run_review_summary_validations


def stage_has_any(paths: list[Path]) -> bool:
    return any(path.exists() for path in paths)


def validate_question_id_column(df, expected_question_id: str, label: str) -> list[str]:
    if "question_id" not in df.columns:
        return []
    actual_question_ids = {str(value).strip() for value in df["question_id"].tolist()}
    if actual_question_ids == {expected_question_id}:
        return []
    return [f"{label}: question_id values do not match directory name {expected_question_id}"]


def validate_csv_artifact(
    path: Path,
    *,
    label: str,
    required_columns: list[str],
    validation_fn,
    expected_question_id: str | None = None,
    validation_kwargs: dict | None = None,
) -> list[str]:
    df = read_csv(path)
    validate_required_columns(df, required_columns)
    errors = validation_fn(df, **(validation_kwargs or {}))
    if expected_question_id is not None:
        errors.extend(validate_question_id_column(df, expected_question_id, label))
    return [f"{label}: {message}" for message in errors]


def validate_json_artifact(
    path: Path,
    *,
    label: str,
    validation_fn,
    validation_kwargs: dict | None = None,
) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = validation_fn(payload, **(validation_kwargs or {}))
    return [f"{label}: {message}" for message in errors]


def collect_question_errors(question_dir: Path) -> list[str]:
    expected_question_id = question_dir.name
    project_dir = question_dir.parent.parent
    errors: list[str] = []

    embedding_dir = question_dir / "03_embeddings"
    clustering_dir = question_dir / "04_clustering"
    classification_dir = question_dir / "05_classification"
    review_dir = question_dir / "06_review"

    requests_path = embedding_dir / "embedding_requests.csv"
    metadata_path = embedding_dir / "embedding_metadata.json"
    failures_path = embedding_dir / "embedding_failures.csv"
    embeddings_path = embedding_dir / "embeddings.npy"
    clusters_path = clustering_dir / "clusters.csv"
    summary_path = clustering_dir / "cluster_summary.csv"
    clustering_metadata_path = clustering_dir / "clustering_metadata.json"
    category_master_path = classification_dir / "category_master.csv"
    category_conflicts_path = classification_dir / "category_conflicts.csv"
    override_rules_path = classification_dir / "manual_override_rules.csv"
    override_candidates_path = classification_dir / "manual_override_candidates.csv"
    override_rule_hits_path = classification_dir / "override_rule_hits.csv"
    override_rule_summary_path = classification_dir / "override_rule_summary.csv"
    final_labels_path = classification_dir / "final_labels.csv"
    review_log_path = review_dir / "review_log.csv"
    review_summary_path = review_dir / "review_summary.csv"
    category_review_priorities_path = review_dir / "category_review_priorities.csv"
    review_samples_path = review_dir / "review_samples.csv"
    review_corrections_path = review_dir / "review_corrections.csv"

    embedding_stage_started = stage_has_any(
        [
            requests_path,
            metadata_path,
            failures_path,
            embeddings_path,
            embedding_dir / "embedding.log",
        ]
    )
    clustering_stage_started = embedding_stage_started or stage_has_any(
        [
            clusters_path,
            summary_path,
            clustering_metadata_path,
            clustering_dir / "clustering.log",
        ]
    )
    classification_stage_started = clustering_stage_started or stage_has_any(
        [
            category_master_path,
            category_conflicts_path,
            override_rules_path,
            override_candidates_path,
            override_rule_hits_path,
            override_rule_summary_path,
            final_labels_path,
            classification_dir / "classification.log",
        ]
    )
    review_stage_started = classification_stage_started or stage_has_any(
        [
            review_log_path,
            review_summary_path,
            category_review_priorities_path,
            review_samples_path,
            review_corrections_path,
            review_dir / "review.log",
        ]
    )

    required_paths: list[tuple[Path, str]] = []
    if embedding_stage_started:
        required_paths.extend(
            [
                (requests_path, "embedding_requests.csv"),
                (metadata_path, "embedding_metadata.json"),
            ]
        )
    if clustering_stage_started:
        required_paths.extend(
            [
                (clusters_path, "clusters.csv"),
                (summary_path, "cluster_summary.csv"),
                (clustering_metadata_path, "clustering_metadata.json"),
            ]
        )
    if classification_stage_started:
        required_paths.append((category_master_path, "category_master.csv"))
    if review_stage_started or final_labels_path.exists():
        required_paths.append((final_labels_path, "final_labels.csv"))
    if review_stage_started:
        required_paths.append((review_log_path, "review_log.csv"))

    for path, label in required_paths:
        if not path.exists():
            errors.append(f"Missing file: {label} ({path})")

    if requests_path.exists():
        errors.extend(
            validate_csv_artifact(
                requests_path,
                label="embedding_requests.csv",
                required_columns=EMBEDDING_REQUEST_COLUMNS,
                validation_fn=run_embedding_request_validations,
                expected_question_id=expected_question_id,
            )
        )

    if metadata_path.exists():
        metadata_kwargs = {
            "screened_path": project_dir / "02_screening" / "screened_responses.csv"
            if (project_dir / "02_screening" / "screened_responses.csv").exists()
            else None,
            "requests_path": requests_path if requests_path.exists() else None,
            "embeddings_path": embeddings_path if embeddings_path.exists() else None,
            "failures_path": failures_path if failures_path.exists() else None,
        }
        errors.extend(
            validate_json_artifact(
                metadata_path,
                label="embedding_metadata.json",
                validation_fn=run_embedding_metadata_validations,
                validation_kwargs=metadata_kwargs,
            )
        )

    if failures_path.exists():
        errors.extend(
            validate_csv_artifact(
                failures_path,
                label="embedding_failures.csv",
                required_columns=EMBEDDING_FAILURE_COLUMNS,
                validation_fn=run_embedding_failure_validations,
                expected_question_id=expected_question_id,
            )
        )

    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
        metadata_payload = None
        if metadata_path.exists():
            metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        errors.extend(
            f"embeddings.npy: {message}"
            for message in run_embeddings_array_validations(
                embeddings,
                request_row_count=int(len(read_csv(requests_path))) if requests_path.exists() else None,
                metadata_row_count=None if metadata_payload is None else metadata_payload.get("row_count"),
                metadata_status=None if metadata_payload is None else metadata_payload.get("status"),
            )
        )

    if clusters_path.exists():
        errors.extend(
            validate_csv_artifact(
                clusters_path,
                label="clusters.csv",
                required_columns=CLUSTER_COLUMNS,
                validation_fn=run_clusters_validations,
                expected_question_id=expected_question_id,
            )
        )

    if summary_path.exists():
        clusters_df = read_csv(clusters_path) if clusters_path.exists() else None
        errors.extend(
            validate_csv_artifact(
                summary_path,
                label="cluster_summary.csv",
                required_columns=SUMMARY_COLUMNS,
                validation_fn=run_cluster_summary_validations,
                expected_question_id=expected_question_id,
                validation_kwargs={"clusters_df": clusters_df} if clusters_df is not None else None,
            )
        )

    if clustering_metadata_path.exists():
        clustering_kwargs = {
            "requests_path": requests_path if requests_path.exists() else None,
            "embeddings_path": embeddings_path if embeddings_path.exists() else None,
            "clusters_path": clusters_path if clusters_path.exists() else None,
            "summary_path": summary_path if summary_path.exists() else None,
        }
        errors.extend(
            validate_json_artifact(
                clustering_metadata_path,
                label="clustering_metadata.json",
                validation_fn=run_clustering_metadata_validations,
                validation_kwargs=clustering_kwargs,
            )
        )

    if category_master_path.exists():
        errors.extend(
            validate_csv_artifact(
                category_master_path,
                label="category_master.csv",
                required_columns=CATEGORY_MASTER_COLUMNS,
                validation_fn=run_category_master_validations,
            )
        )

    if category_conflicts_path.exists():
        errors.extend(
            validate_csv_artifact(
                category_conflicts_path,
                label="category_conflicts.csv",
                required_columns=CATEGORY_CONFLICT_COLUMNS,
                validation_fn=run_category_conflict_validations,
            )
        )

    if override_rules_path.exists():
        errors.extend(
            validate_csv_artifact(
                override_rules_path,
                label="manual_override_rules.csv",
                required_columns=OVERRIDE_RULE_COLUMNS,
                validation_fn=run_override_rule_validations,
            )
        )

    if override_candidates_path.exists():
        errors.extend(
            validate_csv_artifact(
                override_candidates_path,
                label="manual_override_candidates.csv",
                required_columns=OVERRIDE_CANDIDATE_COLUMNS,
                validation_fn=run_override_candidate_validations,
            )
        )

    if override_rule_hits_path.exists():
        errors.extend(
            validate_csv_artifact(
                override_rule_hits_path,
                label="override_rule_hits.csv",
                required_columns=OVERRIDE_RULE_HIT_COLUMNS,
                validation_fn=run_override_rule_hit_validations,
                expected_question_id=expected_question_id,
            )
        )

    if override_rule_summary_path.exists():
        errors.extend(
            validate_csv_artifact(
                override_rule_summary_path,
                label="override_rule_summary.csv",
                required_columns=OVERRIDE_RULE_SUMMARY_COLUMNS,
                validation_fn=run_override_rule_summary_validations,
                expected_question_id=expected_question_id,
            )
        )

    if final_labels_path.exists():
        errors.extend(
            validate_csv_artifact(
                final_labels_path,
                label="final_labels.csv",
                required_columns=FINAL_LABEL_COLUMNS,
                validation_fn=run_final_labels_validations,
                expected_question_id=expected_question_id,
            )
        )

    if review_log_path.exists():
        errors.extend(
            validate_csv_artifact(
                review_log_path,
                label="review_log.csv",
                required_columns=REVIEW_COLUMNS,
                validation_fn=run_review_log_validations,
                expected_question_id=expected_question_id,
            )
        )

    if review_summary_path.exists():
        errors.extend(
            validate_csv_artifact(
                review_summary_path,
                label="review_summary.csv",
                required_columns=REVIEW_SUMMARY_COLUMNS,
                validation_fn=run_review_summary_validations,
                expected_question_id=expected_question_id,
            )
        )

    if category_review_priorities_path.exists():
        errors.extend(
            validate_csv_artifact(
                category_review_priorities_path,
                label="category_review_priorities.csv",
                required_columns=CATEGORY_REVIEW_PRIORITY_COLUMNS,
                validation_fn=run_category_review_priority_validations,
                expected_question_id=expected_question_id,
            )
        )

    if review_samples_path.exists():
        errors.extend(
            validate_csv_artifact(
                review_samples_path,
                label="review_samples.csv",
                required_columns=REVIEW_SAMPLE_COLUMNS,
                validation_fn=run_review_sample_validations,
                expected_question_id=expected_question_id,
            )
        )

    if review_corrections_path.exists():
        errors.extend(
            validate_csv_artifact(
                review_corrections_path,
                label="review_corrections.csv",
                required_columns=REVIEW_CORRECTION_COLUMNS,
                validation_fn=run_review_correction_validations,
                expected_question_id=expected_question_id,
            )
        )

    for log_path in [
        embedding_dir / "embedding.log",
        clustering_dir / "clustering.log",
        classification_dir / "classification.log",
        review_dir / "review.log",
    ]:
        if not log_path.exists():
            continue
        log_errors = run_jsonl_log_validations(log_path.read_text(encoding="utf-8").splitlines())
        errors.extend(f"{log_path.name}: {message}" for message in log_errors)

    duplicate_path = project_dir / "02_screening" / "duplicate_responses.csv"
    if duplicate_path.exists():
        errors.extend(
            validate_csv_artifact(
                duplicate_path,
                label="duplicate_responses.csv",
                required_columns=DUPLICATE_COLUMNS,
                validation_fn=run_duplicate_validations,
            )
        )

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
