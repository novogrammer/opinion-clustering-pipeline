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
from review_override_candidates import OUTPUT_COLUMNS as OVERRIDE_CANDIDATE_COLUMNS
from review_common import REVIEW_COLUMNS, REVIEW_CORRECTION_COLUMNS, REVIEW_SAMPLE_COLUMNS, REVIEW_SUMMARY_COLUMNS
from override_rule_hits import OUTPUT_COLUMNS as OVERRIDE_RULE_HIT_COLUMNS
from override_rule_summary import OUTPUT_COLUMNS as OVERRIDE_RULE_SUMMARY_COLUMNS
from validate_category_master import run_validations as run_category_master_validations
from validate_category_conflicts import run_validations as run_category_conflict_validations
from validate_category_review_priorities import run_validations as run_category_review_priority_validations
from validate_override_rules import run_validations as run_override_rule_validations
from validate_override_candidates import run_validations as run_override_candidate_validations
from validate_override_rule_hits import run_validations as run_override_rule_hit_validations
from validate_override_rule_summary import run_validations as run_override_rule_summary_validations
from validate_cluster_summary import run_validations as run_cluster_summary_validations
from validate_clustering_metadata import run_validations as run_clustering_metadata_validations
from validate_clusters import run_validations as run_clusters_validations
from validate_embeddings_array import run_validations as run_embeddings_array_validations
from validate_embedding_failures import EMBEDDING_FAILURE_COLUMNS, run_validations as run_embedding_failure_validations
from validate_embedding_metadata import run_validations as run_embedding_metadata_validations
from validate_embedding_requests import EMBEDDING_REQUEST_COLUMNS, run_validations as run_embedding_request_validations
from validate_final_labels import run_validations as run_final_labels_validations
from validate_jsonl_log import run_validations as run_jsonl_log_validations
from validate_review_log import run_validations as run_review_log_validations
from validate_review_corrections import run_validations as run_review_correction_validations
from validate_review_samples import run_validations as run_review_sample_validations
from validate_review_summary import run_validations as run_review_summary_validations
from validate_screened_responses import SCREENED_COLUMNS


def validate_csv(path: Path, required_columns: list[str], validation_fn) -> list[str]:
    df = read_csv(path)
    validate_required_columns(df, required_columns)
    return validation_fn(df)


def validate_json(path: Path, validation_fn) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validation_fn(payload)


def validate_question_id_column(df, expected_question_id: str, label: str) -> list[str]:
    actual_question_ids = {str(value) for value in df["question_id"].tolist()}
    if actual_question_ids == {expected_question_id}:
        return []
    return [f"{label}: question_id values do not match directory name {expected_question_id}"]


def validate_response_id_subset(df, allowed_response_ids: set[str], label: str) -> list[str]:
    response_ids = {str(value) for value in df["response_id"].tolist()}
    invalid_ids = sorted(response_ids - allowed_response_ids)
    if not invalid_ids:
        return []
    return [f"{label}: response_id values not found in screened target set"]


def validate_category_id_subset(df, column: str, allowed_category_ids: set[str], label: str) -> list[str]:
    category_ids = {str(value) for value in df[column].tolist()}
    invalid_ids = sorted(category_ids - allowed_category_ids)
    if not invalid_ids:
        return []
    return [f"{label}: {column} contains category_id values not found in category_master.csv"]


def stage_has_any(paths: list[Path]) -> bool:
    return any(path.exists() for path in paths)


def collect_question_errors(question_dir: Path) -> list[str]:
    expected_question_id = question_dir.name
    project_dir = question_dir.parent.parent
    embeddings_primary_checks = [
        (
            question_dir / "03_embeddings" / "embedding_requests.csv",
            "embedding_requests.csv",
            lambda path: validate_csv(path, EMBEDDING_REQUEST_COLUMNS, run_embedding_request_validations),
        ),
        (
            question_dir / "03_embeddings" / "embedding_metadata.json",
            "embedding_metadata.json",
            lambda path: validate_json(path, run_embedding_metadata_validations),
        ),
    ]
    clustering_primary_checks = [
        (
            question_dir / "04_clustering" / "clusters.csv",
            "clusters.csv",
            lambda path: validate_csv(path, CLUSTER_COLUMNS, run_clusters_validations),
        ),
        (
            question_dir / "04_clustering" / "cluster_summary.csv",
            "cluster_summary.csv",
            lambda path: validate_csv(path, SUMMARY_COLUMNS, run_cluster_summary_validations),
        ),
        (
            question_dir / "04_clustering" / "clustering_metadata.json",
            "clustering_metadata.json",
            lambda path: validate_json(path, run_clustering_metadata_validations),
        ),
    ]
    classification_primary_checks = [
        (
            question_dir / "05_classification" / "category_master.csv",
            "category_master.csv",
            lambda path: validate_csv(path, CATEGORY_MASTER_COLUMNS, run_category_master_validations),
        ),
    ]
    final_label_check = (
        (
            question_dir / "05_classification" / "final_labels.csv",
            "final_labels.csv",
            lambda path: validate_csv(path, FINAL_LABEL_COLUMNS, run_final_labels_validations),
        )
    )
    review_primary_checks = [
        (
            question_dir / "06_review" / "review_log.csv",
            "review_log.csv",
            lambda path: validate_csv(path, REVIEW_COLUMNS, run_review_log_validations),
        ),
    ]

    errors: list[str] = []
    optional_checks = [
        (
            question_dir / "05_classification" / "category_conflicts.csv",
            "category_conflicts.csv",
            lambda path: validate_csv(path, CATEGORY_CONFLICT_COLUMNS, run_category_conflict_validations),
        ),
        (
            question_dir / "05_classification" / "manual_override_rules.csv",
            "manual_override_rules.csv",
            lambda path: validate_csv(path, OVERRIDE_RULE_COLUMNS, run_override_rule_validations),
        ),
        (
            question_dir / "05_classification" / "manual_override_candidates.csv",
            "manual_override_candidates.csv",
            lambda path: validate_csv(path, OVERRIDE_CANDIDATE_COLUMNS, run_override_candidate_validations),
        ),
        (
            question_dir / "05_classification" / "override_rule_hits.csv",
            "override_rule_hits.csv",
            lambda path: validate_csv(path, OVERRIDE_RULE_HIT_COLUMNS, run_override_rule_hit_validations),
        ),
        (
            question_dir / "05_classification" / "override_rule_summary.csv",
            "override_rule_summary.csv",
            lambda path: validate_csv(path, OVERRIDE_RULE_SUMMARY_COLUMNS, run_override_rule_summary_validations),
        ),
        (
            question_dir / "06_review" / "review_summary.csv",
            "review_summary.csv",
            lambda path: validate_csv(path, REVIEW_SUMMARY_COLUMNS, run_review_summary_validations),
        ),
        (
            question_dir / "06_review" / "category_review_priorities.csv",
            "category_review_priorities.csv",
            lambda path: validate_csv(path, CATEGORY_REVIEW_PRIORITY_COLUMNS, run_category_review_priority_validations),
        ),
        (
            question_dir / "06_review" / "review_samples.csv",
            "review_samples.csv",
            lambda path: validate_csv(path, REVIEW_SAMPLE_COLUMNS, run_review_sample_validations),
        ),
        (
            question_dir / "06_review" / "review_corrections.csv",
            "review_corrections.csv",
            lambda path: validate_csv(path, REVIEW_CORRECTION_COLUMNS, run_review_correction_validations),
        ),
    ]
    optional_log_checks = [
        (question_dir / "03_embeddings" / "embedding.log", "embedding.log"),
        (question_dir / "04_clustering" / "clustering.log", "clustering.log"),
        (question_dir / "05_classification" / "classification.log", "classification.log"),
        (question_dir / "06_review" / "review.log", "review.log"),
    ]

    review_started = stage_has_any([path for path, *_ in review_primary_checks] + [path for path, label, _ in optional_checks if "06_review" in str(path)] + [path for path, _ in optional_log_checks if "06_review" in str(path)])
    classification_started = review_started or stage_has_any([path for path, *_ in classification_primary_checks] + [final_label_check[0]] + [path for path, label, _ in optional_checks if "05_classification" in str(path)] + [path for path, _ in optional_log_checks if "05_classification" in str(path)])
    clustering_started = classification_started or stage_has_any([path for path, *_ in clustering_primary_checks] + [path for path, _ in optional_log_checks if "04_clustering" in str(path)])
    embeddings_started = clustering_started or stage_has_any([path for path, *_ in embeddings_primary_checks] + [question_dir / "03_embeddings" / "embeddings.npy", question_dir / "03_embeddings" / "embedding_failures.csv"] + [path for path, _ in optional_log_checks if "03_embeddings" in str(path)])

    required_checks: list[tuple[Path, str, object]] = []
    if embeddings_started:
        required_checks.extend(embeddings_primary_checks)
    if clustering_started:
        required_checks.extend(clustering_primary_checks)
    if classification_started:
        required_checks.extend(classification_primary_checks)
    if review_started or final_label_check[0].exists():
        required_checks.append(final_label_check)
    if review_started:
        required_checks.extend(review_primary_checks)

    for path, label, check_fn in required_checks:
        if not path.exists():
            errors.append(f"Missing file: {label} ({path})")
            continue
        try:
            check_errors = check_fn(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}: {exc}")
            continue
        errors.extend(f"{label}: {message}" for message in check_errors)

    for path, label, check_fn in optional_checks:
        if not path.exists():
            continue
        try:
            check_errors = check_fn(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}: {exc}")
            continue
        errors.extend(f"{label}: {message}" for message in check_errors)

    for path, label in optional_log_checks:
        if not path.exists():
            continue
        try:
            check_errors = run_jsonl_log_validations(path.read_text(encoding="utf-8").splitlines())
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}: {exc}")
            continue
        errors.extend(f"{label}: {message}" for message in check_errors)

    screened_path = project_dir / "02_screening" / "screened_responses.csv"
    duplicate_path = project_dir / "02_screening" / "duplicate_responses.csv"
    screened_target_ids: set[str] | None = None
    metadata_payload: dict[str, object] | None = None
    if screened_path.exists():
        try:
            screened_df = read_csv(screened_path)
            validate_required_columns(screened_df, SCREENED_COLUMNS)
            question_screened = screened_df[screened_df["question_id"].astype(str) == expected_question_id]
            screened_target = question_screened[question_screened["is_target"].astype(str).str.lower() == "true"]
            screened_target_ids = {str(value) for value in screened_target["response_id"].tolist()}
        except Exception as exc:  # noqa: BLE001
            errors.append(f"screened_consistency: {exc}")

    duplicate_response_ids: set[str] | None = None
    if duplicate_path.exists():
        try:
            duplicate_df = read_csv(duplicate_path)
            validate_required_columns(duplicate_df, DUPLICATE_COLUMNS)
            question_duplicate_df = duplicate_df[duplicate_df["question_id"].astype(str) == expected_question_id].copy()
            duplicate_response_ids = set(question_duplicate_df["response_id"].astype(str).tolist())
        except Exception as exc:  # noqa: BLE001
            errors.append(f"duplicate_consistency: {exc}")

    requests_path = question_dir / "03_embeddings" / "embedding_requests.csv"
    metadata_path = question_dir / "03_embeddings" / "embedding_metadata.json"
    failures_path = question_dir / "03_embeddings" / "embedding_failures.csv"
    embeddings_path = question_dir / "03_embeddings" / "embeddings.npy"
    if metadata_path.exists():
        try:
            metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"embedding_metadata.json: {exc}")
            metadata_payload = None

    embedding_status = str((metadata_payload or {}).get("status", "")).strip().lower()
    embedding_failed_count = int((metadata_payload or {}).get("failed_count", 0) or 0)

    if embedding_status == "completed" and not embeddings_path.exists():
        errors.append(f"Missing file: embeddings.npy ({embeddings_path})")
    if embedding_status == "failed" and not failures_path.exists():
        errors.append(f"Missing file: embedding_failures.csv ({failures_path})")
    if embedding_status == "prepared":
        if embeddings_path.exists():
            errors.append("embedding_status_consistency: embeddings.npy should not exist when status=prepared")
        if failures_path.exists():
            errors.append("embedding_status_consistency: embedding_failures.csv should not exist when status=prepared")

    if failures_path.exists():
        try:
            failure_errors = validate_csv(failures_path, EMBEDDING_FAILURE_COLUMNS, run_embedding_failure_validations)
            errors.extend(f"embedding_failures.csv: {message}" for message in failure_errors)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"embedding_failures.csv: {exc}")

    if requests_path.exists() and metadata_path.exists():
        try:
            requests_df = read_csv(requests_path)
            errors.extend(validate_question_id_column(requests_df, expected_question_id, "embedding_requests.csv"))
            if screened_target_ids is not None:
                errors.extend(validate_response_id_subset(requests_df, screened_target_ids, "embedding_requests.csv"))
            if metadata_payload is not None:
                metadata_validation_errors = run_embedding_metadata_validations(
                    metadata_payload,
                    screened_path=screened_path if screened_path.exists() else None,
                    requests_path=requests_path,
                )
                errors.extend(f"embedding_metadata.json: {message}" for message in metadata_validation_errors)
                if metadata_payload.get("row_count") != int(len(requests_df)):
                    errors.append("embedding_consistency: embedding_metadata.json row_count does not match embedding_requests.csv")
                if embedding_status in {"prepared", "completed"} and embedding_failed_count != 0:
                    errors.append("embedding_consistency: failed_count must be 0 when status is prepared or completed")
                if embedding_status == "failed":
                    if not failures_path.exists():
                        errors.append("embedding_consistency: status=failed requires embedding_failures.csv")
                    elif embedding_failed_count != int(len(read_csv(failures_path))):
                        errors.append("embedding_consistency: failed_count does not match embedding_failures.csv row count")
            if embeddings_path.exists():
                embeddings = np.load(embeddings_path)
                cross_errors = run_embeddings_array_validations(
                    embeddings,
                    request_row_count=int(len(requests_df)),
                    metadata_row_count=(metadata_payload or {}).get("row_count"),
                )
                errors.extend(f"embedding_consistency: {message}" for message in cross_errors)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"embedding_consistency: {exc}")

    if failures_path.exists():
        try:
            failures_df = read_csv(failures_path)
            errors.extend(validate_question_id_column(failures_df, expected_question_id, "embedding_failures.csv"))
            if screened_target_ids is not None:
                errors.extend(validate_response_id_subset(failures_df, screened_target_ids, "embedding_failures.csv"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"embedding_failure_consistency: {exc}")

    clusters_path = question_dir / "04_clustering" / "clusters.csv"
    summary_path = question_dir / "04_clustering" / "cluster_summary.csv"
    if clusters_path.exists() and summary_path.exists():
        try:
            clusters_df = read_csv(clusters_path)
            summary_df = read_csv(summary_path)
            errors.extend(validate_question_id_column(clusters_df, expected_question_id, "clusters.csv"))
            errors.extend(validate_question_id_column(summary_df, expected_question_id, "cluster_summary.csv"))
            if requests_path.exists():
                requests_df = read_csv(requests_path)
                request_ids = {str(value) for value in requests_df["response_id"].tolist()}
                errors.extend(validate_response_id_subset(clusters_df, request_ids, "clusters.csv"))
            cluster_topics = {str(value) for value in clusters_df["topic_id"].tolist()}
            summary_topics = {str(value) for value in summary_df["topic_id"].tolist()}
            if cluster_topics != summary_topics:
                errors.append(
                    "cluster_consistency: topic_id sets differ between clusters.csv and cluster_summary.csv"
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"cluster_consistency: {exc}")

    clustering_metadata_path = question_dir / "04_clustering" / "clustering_metadata.json"
    if clustering_metadata_path.exists():
        try:
            clustering_metadata_payload = json.loads(clustering_metadata_path.read_text(encoding="utf-8"))
            clustering_metadata_errors = run_clustering_metadata_validations(
                clustering_metadata_payload,
                requests_path=requests_path if requests_path.exists() else None,
                embeddings_path=embeddings_path if embeddings_path.exists() else None,
            )
            errors.extend(f"clustering_metadata.json: {message}" for message in clustering_metadata_errors)
            if requests_path.exists() and clustering_metadata_payload.get("row_count") != int(len(read_csv(requests_path))):
                errors.append("cluster_metadata_consistency: clustering_metadata.json row_count does not match embedding_requests.csv")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"cluster_metadata_consistency: {exc}")

    category_master_path = question_dir / "05_classification" / "category_master.csv"
    final_labels_path = question_dir / "05_classification" / "final_labels.csv"
    override_rules_path = question_dir / "05_classification" / "manual_override_rules.csv"
    review_log_path = question_dir / "06_review" / "review_log.csv"
    if category_master_path.exists() and final_labels_path.exists():
        try:
            category_master_df = read_csv(category_master_path)
            final_labels_df = read_csv(final_labels_path)
            errors.extend(validate_question_id_column(final_labels_df, expected_question_id, "final_labels.csv"))
            if screened_target_ids is not None:
                errors.extend(validate_response_id_subset(final_labels_df, screened_target_ids, "final_labels.csv"))
            category_ids = set(category_master_df["category_id"].astype(str).tolist())
            predicted_ids = set(final_labels_df["predicted_category_id"].astype(str).tolist())
            invalid_predicted_ids = sorted(predicted_ids - category_ids - {"OTHER"})
            if invalid_predicted_ids:
                errors.append(
                    "classification_consistency: final_labels.csv contains unknown predicted_category_id values"
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"classification_consistency: {exc}")

    if category_master_path.exists() and override_rules_path.exists():
        try:
            category_master_df = read_csv(category_master_path)
            override_rules_df = read_csv(override_rules_path)
            allowed_category_ids = set(category_master_df["category_id"].astype(str).tolist()) | {"OTHER"}
            errors.extend(
                validate_category_id_subset(
                    override_rules_df,
                    "override_category_id",
                    allowed_category_ids,
                    "manual_override_rules.csv",
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"override_rule_consistency: {exc}")

    override_candidates_path = question_dir / "05_classification" / "manual_override_candidates.csv"
    if category_master_path.exists() and override_candidates_path.exists():
        try:
            category_master_df = read_csv(category_master_path)
            override_candidates_df = read_csv(override_candidates_path)
            allowed_category_ids = set(category_master_df["category_id"].astype(str).tolist()) | {"OTHER"}
            errors.extend(
                validate_category_id_subset(
                    override_candidates_df,
                    "override_category_id",
                    allowed_category_ids,
                    "manual_override_candidates.csv",
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"override_candidate_consistency: {exc}")

    override_rule_hits_path = question_dir / "05_classification" / "override_rule_hits.csv"
    if override_rules_path.exists() and override_rule_hits_path.exists():
        try:
            override_rules_df = read_csv(override_rules_path)
            override_rule_hits_df = read_csv(override_rule_hits_path)
            rule_ids = set(override_rules_df["rule_id"].astype(str).tolist())
            hit_rule_ids = set(override_rule_hits_df["rule_id"].astype(str).tolist())
            invalid_rule_ids = sorted(hit_rule_ids - rule_ids)
            if invalid_rule_ids:
                errors.append("override_rule_hit_consistency: override_rule_hits.csv contains unknown rule_id values")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"override_rule_hit_consistency: {exc}")

    override_rule_summary_path = question_dir / "05_classification" / "override_rule_summary.csv"
    if override_rule_hits_path.exists() and override_rule_summary_path.exists():
        try:
            override_rule_hits_df = read_csv(override_rule_hits_path)
            override_rule_summary_df = read_csv(override_rule_summary_path)
            hit_rule_ids = set(override_rule_hits_df["rule_id"].astype(str).tolist())
            summary_rule_ids = set(override_rule_summary_df["rule_id"].astype(str).tolist())
            if hit_rule_ids != summary_rule_ids:
                errors.append(
                    "override_rule_summary_consistency: rule_id sets differ between override_rule_hits.csv and override_rule_summary.csv"
                )
            errors.extend(validate_question_id_column(override_rule_summary_df, expected_question_id, "override_rule_summary.csv"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"override_rule_summary_consistency: {exc}")

    if final_labels_path.exists() and review_log_path.exists():
        try:
            final_labels_df = read_csv(final_labels_path)
            review_log_df = read_csv(review_log_path)
            errors.extend(validate_question_id_column(review_log_df, expected_question_id, "review_log.csv"))
            final_ids = set(final_labels_df["response_id"].astype(str).tolist())
            review_ids = set(review_log_df["response_id"].astype(str).tolist())
            if final_ids != review_ids:
                errors.append("review_consistency: response_id sets differ between final_labels.csv and review_log.csv")
            if duplicate_response_ids is not None:
                review_duplicate_ids = set(
                    review_log_df[
                        review_log_df["review_trigger"].astype(str).map(
                            lambda value: "duplicate_response" in [token.strip() for token in value.split("|") if token.strip()]
                        )
                    ]["response_id"].astype(str).tolist()
                )
                missing_duplicate_ids = sorted(duplicate_response_ids - review_duplicate_ids)
                unexpected_duplicate_ids = sorted(review_duplicate_ids - duplicate_response_ids)
                if missing_duplicate_ids:
                    errors.append(
                        "review_duplicate_consistency: some duplicate_responses.csv rows are missing duplicate_response trigger in review_log.csv"
                    )
                if unexpected_duplicate_ids:
                    errors.append(
                        "review_duplicate_consistency: review_log.csv has duplicate_response trigger for responses not listed in duplicate_responses.csv"
                    )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"review_consistency: {exc}")

    review_samples_path = question_dir / "06_review" / "review_samples.csv"
    if review_log_path.exists() and review_samples_path.exists():
        try:
            review_log_df = read_csv(review_log_path)
            review_samples_df = read_csv(review_samples_path)
            errors.extend(validate_question_id_column(review_samples_df, expected_question_id, "review_samples.csv"))
            allowed_response_ids = set(review_log_df["response_id"].astype(str).tolist())
            errors.extend(
                validate_response_id_subset(
                    review_samples_df,
                    allowed_response_ids,
                    "review_samples.csv",
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"review_sample_consistency: {exc}")

    review_corrections_path = question_dir / "06_review" / "review_corrections.csv"
    if review_log_path.exists() and review_corrections_path.exists():
        try:
            review_log_df = read_csv(review_log_path)
            review_corrections_df = read_csv(review_corrections_path)
            errors.extend(validate_question_id_column(review_corrections_df, expected_question_id, "review_corrections.csv"))
            reviewed_ids = set(
                review_log_df[
                    review_log_df["review_status"].astype(str).str.lower() == "reviewed"
                ]["response_id"].astype(str).tolist()
            )
            errors.extend(
                validate_response_id_subset(
                    review_corrections_df,
                    reviewed_ids,
                    "review_corrections.csv",
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"review_correction_consistency: {exc}")

    if review_corrections_path.exists() and override_candidates_path.exists():
        try:
            review_corrections_df = read_csv(review_corrections_path)
            override_candidates_df = read_csv(override_candidates_path)
            correction_response_ids = set(review_corrections_df["response_id"].astype(str).tolist())
            errors.extend(
                validate_response_id_subset(
                    override_candidates_df.rename(columns={"source_response_id": "response_id"}),
                    correction_response_ids,
                    "manual_override_candidates.csv",
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"override_candidate_source_consistency: {exc}")

    if screened_path.exists() and override_rule_hits_path.exists():
        try:
            override_rule_hits_df = read_csv(override_rule_hits_path)
            if screened_target_ids is not None:
                errors.extend(
                    validate_response_id_subset(
                        override_rule_hits_df,
                        screened_target_ids,
                        "override_rule_hits.csv",
                    )
                )
            errors.extend(validate_question_id_column(override_rule_hits_df, expected_question_id, "override_rule_hits.csv"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"override_rule_hit_screened_consistency: {exc}")

    review_summary_path = question_dir / "06_review" / "review_summary.csv"
    if final_labels_path.exists() and review_summary_path.exists():
        try:
            final_labels_df = read_csv(final_labels_path)
            review_summary_df = read_csv(review_summary_path)
            errors.extend(validate_question_id_column(review_summary_df, expected_question_id, "review_summary.csv"))
            final_category_ids = {str(value) for value in final_labels_df["predicted_category_id"].tolist()}
            summary_category_ids = {str(value) for value in review_summary_df["predicted_category_id"].tolist()}
            if final_category_ids != summary_category_ids:
                errors.append(
                    "review_summary_consistency: predicted_category_id sets differ between final_labels.csv and review_summary.csv"
                )
            final_category_counts = (
                final_labels_df.groupby("predicted_category_id", sort=False)
                .size()
                .to_dict()
            )
            for _, row in review_summary_df.iterrows():
                category_id = str(row["predicted_category_id"])
                expected_total = int(final_category_counts.get(category_id, 0))
                if int(row["total_count"]) != expected_total:
                    errors.append(
                        f"review_summary_consistency: total_count does not match final_labels.csv for predicted_category_id={category_id}"
                    )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"review_summary_consistency: {exc}")

    if review_log_path.exists() and review_summary_path.exists():
        try:
            review_log_df = read_csv(review_log_path)
            review_summary_df = read_csv(review_summary_path)
            for _, row in review_summary_df.iterrows():
                category_id = str(row["predicted_category_id"])
                category_group = review_log_df[
                    review_log_df["predicted_category_id"].astype(str) == category_id
                ].copy()
                if int(row["reviewed_count"]) != int(
                    (category_group["review_status"].astype(str).str.lower() == "reviewed").sum()
                ):
                    errors.append(
                        f"review_summary_review_log_consistency: reviewed_count mismatch for predicted_category_id={category_id}"
                    )
                if int(row["pending_count"]) != int(
                    (category_group["review_status"].astype(str).str.lower() == "pending").sum()
                ):
                    errors.append(
                        f"review_summary_review_log_consistency: pending_count mismatch for predicted_category_id={category_id}"
                    )
                if int(row["skipped_count"]) != int(
                    (category_group["review_status"].astype(str).str.lower() == "skipped").sum()
                ):
                    errors.append(
                        f"review_summary_review_log_consistency: skipped_count mismatch for predicted_category_id={category_id}"
                    )
                corrected_count = int(
                    (
                        (category_group["review_status"].astype(str).str.lower() == "reviewed")
                        & (
                            category_group["reviewed_category_id"].astype(str).str.strip()
                            != category_group["predicted_category_id"].astype(str).str.strip()
                        )
                    ).sum()
                )
                if int(row["corrected_count"]) != corrected_count:
                    errors.append(
                        f"review_summary_review_log_consistency: corrected_count mismatch for predicted_category_id={category_id}"
                    )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"review_summary_review_log_consistency: {exc}")

    category_conflicts_path = question_dir / "05_classification" / "category_conflicts.csv"
    if category_master_path.exists() and category_conflicts_path.exists():
        try:
            category_master_df = read_csv(category_master_path)
            category_conflicts_df = read_csv(category_conflicts_path)
            allowed_category_ids = {str(value) for value in category_master_df["category_id"].tolist()}
            errors.extend(
                validate_category_id_subset(
                    category_conflicts_df,
                    "left_category_id",
                    allowed_category_ids,
                    "category_conflicts.csv",
                )
            )
            errors.extend(
                validate_category_id_subset(
                    category_conflicts_df,
                    "right_category_id",
                    allowed_category_ids,
                    "category_conflicts.csv",
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"category_conflicts_consistency: {exc}")

    category_review_priorities_path = question_dir / "06_review" / "category_review_priorities.csv"
    if review_summary_path.exists() and category_review_priorities_path.exists():
        try:
            review_summary_df = read_csv(review_summary_path)
            category_review_priorities_df = read_csv(category_review_priorities_path)
            review_summary_ids = {str(value) for value in review_summary_df["predicted_category_id"].tolist()}
            priority_ids = {str(value) for value in category_review_priorities_df["category_id"].tolist()}
            if review_summary_ids != priority_ids:
                errors.append(
                    "review_priority_consistency: category_id sets differ between review_summary.csv and category_review_priorities.csv"
                )
            errors.extend(
                validate_question_id_column(
                    category_review_priorities_df,
                    expected_question_id,
                    "category_review_priorities.csv",
                )
            )
            review_summary_lookup = (
                review_summary_df[
                    [
                        "predicted_category_id",
                        "needs_definition_review",
                        "definition_review_reason",
                        "correction_rate",
                        "corrected_count",
                        "high_priority_count",
                    ]
                ]
                .astype(str)
                .set_index("predicted_category_id")
                .to_dict(orient="index")
            )
            for _, row in category_review_priorities_df.iterrows():
                category_id = str(row["category_id"])
                summary_row = review_summary_lookup.get(category_id)
                if summary_row is None:
                    continue
                if str(row["needs_definition_review"]).lower() != str(summary_row["needs_definition_review"]).lower():
                    errors.append(
                        f"review_priority_consistency: needs_definition_review mismatch for category_id={category_id}"
                    )
                if str(row["definition_review_reason"]).strip() != str(summary_row["definition_review_reason"]).strip():
                    errors.append(
                        f"review_priority_consistency: definition_review_reason mismatch for category_id={category_id}"
                    )
                if float(row["correction_rate"]) != float(summary_row["correction_rate"]):
                    errors.append(
                        f"review_priority_consistency: correction_rate mismatch for category_id={category_id}"
                    )
                if int(row["corrected_count"]) != int(summary_row["corrected_count"]):
                    errors.append(
                        f"review_priority_consistency: corrected_count mismatch for category_id={category_id}"
                    )
                if int(row["high_priority_count"]) != int(summary_row["high_priority_count"]):
                    errors.append(
                        f"review_priority_consistency: high_priority_count mismatch for category_id={category_id}"
                    )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"review_priority_consistency: {exc}")

    if category_conflicts_path.exists() and category_review_priorities_path.exists():
        try:
            category_conflicts_df = read_csv(category_conflicts_path)
            category_review_priorities_df = read_csv(category_review_priorities_path)
            conflict_ids = {
                str(value)
                for column in ["left_category_id", "right_category_id"]
                for value in category_conflicts_df[column].tolist()
            }
            priority_conflicted_ids = {
                str(row["category_id"])
                for _, row in category_review_priorities_df.iterrows()
                if int(row["conflict_pair_count"]) > 0
            }
            missing_conflicted_ids = sorted(conflict_ids - priority_conflicted_ids)
            if missing_conflicted_ids:
                errors.append(
                    "review_priority_conflict_consistency: some conflicted categories are missing conflict_pair_count > 0"
                )
            expected_conflict_counts: dict[str, int] = {}
            expected_high_conflict_counts: dict[str, int] = {}
            for _, row in category_conflicts_df.iterrows():
                left_id = str(row["left_category_id"])
                right_id = str(row["right_category_id"])
                conflict_level = str(row["conflict_level"]).strip().lower()
                for category_id in [left_id, right_id]:
                    expected_conflict_counts[category_id] = expected_conflict_counts.get(category_id, 0) + 1
                    if conflict_level == "high":
                        expected_high_conflict_counts[category_id] = expected_high_conflict_counts.get(category_id, 0) + 1
            for _, row in category_review_priorities_df.iterrows():
                category_id = str(row["category_id"])
                if int(row["conflict_pair_count"]) != int(expected_conflict_counts.get(category_id, 0)):
                    errors.append(
                        f"review_priority_conflict_consistency: conflict_pair_count mismatch for category_id={category_id}"
                    )
                if int(row["high_conflict_pair_count"]) != int(expected_high_conflict_counts.get(category_id, 0)):
                    errors.append(
                        f"review_priority_conflict_consistency: high_conflict_pair_count mismatch for category_id={category_id}"
                    )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"review_priority_conflict_consistency: {exc}")

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
