from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns, write_csv, write_json
from screening import SCREENED_COLUMNS, run_validations as run_screened_validations


CATEGORY_MASTER_COLUMNS = [
    "category_id",
    "category_name",
    "category_definition",
]
TOPIC_CATEGORY_MAPPING_COLUMNS = [
    "topic_id",
    "category_id",
]
FINAL_LABEL_COLUMNS = [
    "response_id",
    "question_id",
    "answer_text",
    "topic_id",
    "predicted_category_id",
    "predicted_category_name",
]
CLASSIFICATION_METADATA_KEYS = [
    "created_at",
    "question_id",
    "row_count",
    "category_count",
    "mapping_count",
    "input_screened_path",
    "input_clusters_path",
    "input_category_master_path",
    "input_topic_category_mapping_path",
]
FALLBACK_CATEGORY_ID = "OTHER"
FALLBACK_CATEGORY_NAME = "その他"
OUTLIER_TOPIC_ID = "-1"


def normalize_text(value: object) -> str:
    return str(value).strip()


def validate_required_text(df: pd.DataFrame, columns: list[str]) -> list[str]:
    errors: list[str] = []
    for column in columns:
        blank_mask = df[column].map(lambda value: normalize_text(value) == "")
        count = int(blank_mask.sum())
        if count > 0:
            errors.append(f"Blank values found in {column}: {count}")
    return errors


def validate_no_duplicate_values(df: pd.DataFrame, column: str, label: str) -> list[str]:
    duplicate_mask = df[column].astype(str).duplicated(keep=False)
    duplicates = df.loc[duplicate_mask, column].astype(str).tolist()
    if not duplicates:
        return []
    joined = ", ".join(dict.fromkeys(duplicates))
    return [f"Duplicate {label} values found: {joined}"]


def run_category_master_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if len(df) == 0:
        return ["category_master.csv must contain at least one category"]
    errors.extend(validate_no_duplicate_values(df, "category_id", "category_id"))
    errors.extend(validate_required_text(df, CATEGORY_MASTER_COLUMNS))
    return errors


def run_topic_category_mapping_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if len(df) == 0:
        return ["topic_category_mapping.csv must contain at least one mapping"]
    errors.extend(validate_no_duplicate_values(df, "topic_id", "topic_id"))
    errors.extend(validate_required_text(df, TOPIC_CATEGORY_MAPPING_COLUMNS))
    invalid_outlier_rows = [
        str(index + 1)
        for index, value in enumerate(df["topic_id"].astype(str))
        if normalize_text(value) == OUTLIER_TOPIC_ID
    ]
    if invalid_outlier_rows:
        errors.append(f"topic_id=-1 must not appear in topic_category_mapping.csv rows: {', '.join(invalid_outlier_rows)}")
    return errors


def run_final_label_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_no_duplicate_values(df, "response_id", "response_id"))
    errors.extend(
        validate_required_text(
            df,
            [
                "response_id",
                "question_id",
                "answer_text",
                "topic_id",
                "predicted_category_id",
                "predicted_category_name",
            ],
        )
    )
    return errors


def build_metadata_payload(
    *,
    question_id: str,
    row_count: int,
    category_count: int,
    mapping_count: int,
    screened_path: Path,
    clusters_path: Path,
    category_master_path: Path,
    topic_category_mapping_path: Path,
) -> dict[str, object]:
    return {
        "created_at": utc_now_iso(),
        "question_id": question_id,
        "row_count": row_count,
        "category_count": category_count,
        "mapping_count": mapping_count,
        "input_screened_path": str(screened_path),
        "input_clusters_path": str(clusters_path),
        "input_category_master_path": str(category_master_path),
        "input_topic_category_mapping_path": str(topic_category_mapping_path),
    }


def run_classification_metadata_validations(
    payload: dict[str, object],
    *,
    screened_path: Path,
    clusters_path: Path,
    category_master_path: Path,
    topic_category_mapping_path: Path,
    final_labels_path: Path,
) -> list[str]:
    errors: list[str] = []
    for key in CLASSIFICATION_METADATA_KEYS:
        if key not in payload:
            errors.append(f"Missing key: {key}")
    if payload.get("question_id", "") == "":
        errors.append("question_id must not be blank")
    for key in ("row_count", "category_count", "mapping_count"):
        value = payload.get(key)
        if not isinstance(value, int) or value < 0:
            errors.append(f"{key} must be a non-negative integer")
    expected_paths = {
        "input_screened_path": str(screened_path),
        "input_clusters_path": str(clusters_path),
        "input_category_master_path": str(category_master_path),
        "input_topic_category_mapping_path": str(topic_category_mapping_path),
    }
    for key, expected in expected_paths.items():
        if payload.get(key) != expected:
            errors.append(f"{key} does not match the expected path")
    if not final_labels_path.exists():
        errors.append("final_labels.csv must exist when writing classification metadata")
    elif int(len(read_csv(final_labels_path))) != payload.get("row_count"):
        errors.append("row_count does not match final_labels.csv row count")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reapply curated topic mappings for one question.")
    parser.add_argument("--input", required=True, type=Path, help="Path to screened_responses.csv")
    parser.add_argument("--question-id", required=True, help="Target question_id")
    parser.add_argument("--clusters", required=True, type=Path, help="Path to 04_clustering/clusters.csv")
    parser.add_argument("--category-master", required=True, type=Path, help="Path to 05_curation/category_master.csv")
    parser.add_argument(
        "--topic-category-mapping",
        required=True,
        type=Path,
        help="Path to 05_curation/topic_category_mapping.csv",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Path to 06_classification directory")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    return parser.parse_args()


def build_target_rows(df: pd.DataFrame, question_id: str) -> pd.DataFrame:
    return df[(df["question_id"] == question_id) & (df["is_target"].astype(str).str.lower() == "true")].copy()


def validate_cluster_alignment(target_rows: pd.DataFrame, clusters_df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if len(target_rows) != len(clusters_df):
        errors.append("clusters.csv row count does not match screened target row count")
        return errors
    target_pairs = set(zip(target_rows["response_id"].astype(str), target_rows["question_id"].astype(str)))
    cluster_pairs = set(zip(clusters_df["response_id"].astype(str), clusters_df["question_id"].astype(str)))
    if target_pairs != cluster_pairs:
        errors.append("clusters.csv responses do not match screened target responses")
    return errors


def validate_mapping_category_ids(mapping_df: pd.DataFrame, category_master_df: pd.DataFrame) -> list[str]:
    known_ids = set(category_master_df["category_id"].astype(str))
    mapped_ids = set(mapping_df["category_id"].astype(str))
    missing = sorted(mapped_ids - known_ids)
    if not missing:
        return []
    return [f"topic_category_mapping.csv contains category_id values not found in category_master.csv: {', '.join(missing)}"]


def validate_missing_non_outlier_topics(clusters_df: pd.DataFrame, mapping_df: pd.DataFrame) -> list[str]:
    cluster_topics = {normalize_text(value) for value in clusters_df["topic_id"].astype(str)}
    non_outlier_topics = sorted(topic for topic in cluster_topics if topic != OUTLIER_TOPIC_ID)
    mapped_topics = {normalize_text(value) for value in mapping_df["topic_id"].astype(str)}
    missing = [topic for topic in non_outlier_topics if topic not in mapped_topics]
    if not missing:
        return []
    return [f"topic_category_mapping.csv is missing topic_id values: {', '.join(missing)}"]


def build_final_labels_df(
    *,
    responses_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    category_master_df: pd.DataFrame,
    question_id: str,
) -> pd.DataFrame:
    target_rows = build_target_rows(responses_df, question_id)
    merged = target_rows.merge(
        clusters_df[["response_id", "question_id", "topic_id"]],
        on=["response_id", "question_id"],
        how="left",
    )
    merged["topic_id"] = merged["topic_id"].map(normalize_text)

    mapping_lookup = mapping_df.copy()
    mapping_lookup["topic_id"] = mapping_lookup["topic_id"].map(normalize_text)
    mapping_lookup["category_id"] = mapping_lookup["category_id"].map(normalize_text)

    category_lookup = category_master_df.copy()
    category_lookup["category_id"] = category_lookup["category_id"].map(normalize_text)
    category_lookup["category_name"] = category_lookup["category_name"].map(normalize_text)

    merged = merged.merge(mapping_lookup, on="topic_id", how="left")
    merged = merged.merge(category_lookup[["category_id", "category_name"]], on="category_id", how="left")

    outlier_mask = merged["topic_id"] == OUTLIER_TOPIC_ID
    merged.loc[outlier_mask, "category_id"] = FALLBACK_CATEGORY_ID
    merged.loc[outlier_mask, "category_name"] = FALLBACK_CATEGORY_NAME

    merged["predicted_category_id"] = merged["category_id"].map(normalize_text)
    merged["predicted_category_name"] = merged["category_name"].map(normalize_text)
    return merged[
        [
            "response_id",
            "question_id",
            "answer_text",
            "topic_id",
            "predicted_category_id",
            "predicted_category_name",
        ]
    ].copy()


def main() -> None:
    args = parse_args()

    responses_df = read_csv(args.input)
    validate_required_columns(responses_df, SCREENED_COLUMNS)
    screened_errors = run_screened_validations(responses_df)
    if screened_errors:
        raise SystemExit("\n".join(screened_errors))

    clusters_df = read_csv(args.clusters)
    validate_required_columns(clusters_df, ["response_id", "question_id", "topic_id"])
    cluster_alignment_errors = validate_cluster_alignment(build_target_rows(responses_df, args.question_id), clusters_df)
    if cluster_alignment_errors:
        raise SystemExit("\n".join(cluster_alignment_errors))

    category_master_df = read_csv(args.category_master)
    validate_required_columns(category_master_df, CATEGORY_MASTER_COLUMNS)
    category_master_errors = run_category_master_validations(category_master_df)
    if category_master_errors:
        raise SystemExit("\n".join(category_master_errors))

    mapping_df = read_csv(args.topic_category_mapping)
    validate_required_columns(mapping_df, TOPIC_CATEGORY_MAPPING_COLUMNS)
    mapping_errors = run_topic_category_mapping_validations(mapping_df)
    mapping_errors.extend(validate_mapping_category_ids(mapping_df, category_master_df))
    mapping_errors.extend(validate_missing_non_outlier_topics(clusters_df, mapping_df))
    if mapping_errors:
        raise SystemExit("\n".join(mapping_errors))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_labels_path = args.output_dir / "final_labels.csv"
    metadata_path = args.output_dir / "classification_metadata.json"

    final_labels_df = build_final_labels_df(
        responses_df=responses_df,
        clusters_df=clusters_df,
        mapping_df=mapping_df,
        category_master_df=category_master_df,
        question_id=args.question_id,
    )
    final_label_errors = run_final_label_validations(final_labels_df)
    if final_label_errors:
        raise SystemExit("\n".join(final_label_errors))

    write_csv(final_labels_df, final_labels_path)

    metadata_payload = build_metadata_payload(
        question_id=args.question_id,
        row_count=int(len(final_labels_df)),
        category_count=int(len(category_master_df)),
        mapping_count=int(len(mapping_df)),
        screened_path=args.input,
        clusters_path=args.clusters,
        category_master_path=args.category_master,
        topic_category_mapping_path=args.topic_category_mapping,
    )
    metadata_errors = run_classification_metadata_validations(
        metadata_payload,
        screened_path=args.input,
        clusters_path=args.clusters,
        category_master_path=args.category_master,
        topic_category_mapping_path=args.topic_category_mapping,
        final_labels_path=final_labels_path,
    )
    if metadata_errors:
        raise SystemExit("\n".join(metadata_errors))
    write_json(metadata_payload, metadata_path)

    if args.log is not None:
        append_jsonl(
            {
                "event": "classification",
                "input": str(args.input),
                "clusters": str(args.clusters),
                "category_master": str(args.category_master),
                "topic_category_mapping": str(args.topic_category_mapping),
                "output_dir": str(args.output_dir),
                "question_id": args.question_id,
                "row_count": int(len(final_labels_df)),
                "category_count": int(len(category_master_df)),
                "mapping_count": int(len(mapping_df)),
                "created_at": utc_now_iso(),
            },
            args.log,
        )


if __name__ == "__main__":
    main()
