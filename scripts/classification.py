from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns, write_csv, write_json
from embeddings import file_sha1, run_embedding_metadata_validations
from screening import SCREENED_COLUMNS, run_validations as run_screened_validations


CATEGORY_MASTER_COLUMNS = [
    "category_id",
    "category_name",
    "category_definition",
    "representative_examples",
]
FINAL_LABEL_COLUMNS = [
    "response_id",
    "question_id",
    "answer_text",
    "predicted_category_id",
    "predicted_category_name",
    "confidence",
    "reason",
    "needs_human_review",
]
CLASSIFICATION_METADATA_KEYS = [
    "created_at",
    "question_id",
    "row_count",
    "category_count",
    "embedding_model",
    "confidence_threshold",
    "input_screened_path",
    "input_screened_sha1",
    "input_embeddings_path",
    "input_embeddings_sha1",
    "input_category_master_path",
    "input_category_master_sha1",
    "category_embeddings_path",
]
FALLBACK_CATEGORY_ID = "OTHER"
FALLBACK_CATEGORY_NAME = "その他"


def normalize_text(value: object) -> str:
    return str(value).strip()


def validate_no_duplicate_category_ids(df: pd.DataFrame) -> list[str]:
    duplicate_mask = df["category_id"].duplicated(keep=False)
    duplicates = df.loc[duplicate_mask, "category_id"].tolist()
    if not duplicates:
        return []
    joined = ", ".join(dict.fromkeys(duplicates))
    return [f"Duplicate category_id values found: {joined}"]


def validate_required_text(df: pd.DataFrame, columns: list[str]) -> list[str]:
    errors: list[str] = []
    for column in columns:
        blank_mask = df[column].map(lambda value: normalize_text(value) == "")
        count = int(blank_mask.sum())
        if count > 0:
            errors.append(f"Blank values found in {column}: {count}")
    return errors


def run_category_master_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if len(df) == 0:
        return ["category_master.csv must contain at least one category"]
    errors.extend(validate_no_duplicate_category_ids(df))
    errors.extend(validate_required_text(df, CATEGORY_MASTER_COLUMNS))
    return errors


def validate_no_duplicate_response_ids(df: pd.DataFrame) -> list[str]:
    duplicate_mask = df["response_id"].duplicated(keep=False)
    duplicates = df.loc[duplicate_mask, "response_id"].tolist()
    if not duplicates:
        return []
    joined = ", ".join(dict.fromkeys(duplicates))
    return [f"Duplicate response_id values found: {joined}"]


def validate_confidence_range(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for idx, value in enumerate(df["confidence"], start=1):
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            errors.append(f"Invalid confidence at row {idx}: {value}")
            continue
        if confidence < 0 or confidence > 1:
            errors.append(f"Out-of-range confidence at row {idx}: {value}")
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


def run_final_label_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_no_duplicate_response_ids(df))
    errors.extend(validate_confidence_range(df))
    errors.extend(validate_boolean_column(df, "needs_human_review"))
    errors.extend(
        validate_required_text(
            df,
            ["response_id", "question_id", "answer_text", "predicted_category_id", "predicted_category_name", "reason"],
        )
    )
    return errors


def build_category_embedding_input(row: pd.Series) -> str:
    return "\n".join(
        [
            f"カテゴリ名: {normalize_text(row['category_name'])}",
            f"定義: {normalize_text(row['category_definition'])}",
            f"代表例: {normalize_text(row['representative_examples'])}",
        ]
    )


def load_embedding_metadata(metadata_path: Path, embeddings_path: Path, screened_path: Path) -> dict[str, object]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"embedding_metadata.json does not exist: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    errors = run_embedding_metadata_validations(
        payload,
        screened_path=screened_path,
        embeddings_path=embeddings_path,
        failures_path=embeddings_path.parent / "embedding_failures.csv",
    )
    if errors:
        raise ValueError("\n".join(errors))
    if payload.get("status") != "completed":
        raise ValueError(f"embedding_metadata.json status must be completed: {payload.get('status')}")
    return payload


def validate_embeddings_row_count(embeddings: np.ndarray, expected_row_count: int) -> list[str]:
    row_count = int(len(embeddings))
    if row_count != expected_row_count:
        return [f"embeddings.npy row count does not match screened target row count: {row_count} != {expected_row_count}"]
    return []


def cosine_similarity_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    left_safe = np.where(left_norm == 0, 1.0, left_norm)
    right_safe = np.where(right_norm == 0, 1.0, right_norm)
    normalized_left = left / left_safe
    normalized_right = right / right_safe
    return normalized_left @ normalized_right.T


def to_confidence(similarity: float) -> float:
    clipped = max(-1.0, min(1.0, float(similarity)))
    return round((clipped + 1.0) / 2.0, 3)


def build_metadata_payload(
    *,
    question_id: str,
    row_count: int,
    category_count: int,
    embedding_model: str,
    confidence_threshold: float,
    screened_path: Path,
    embeddings_path: Path,
    category_master_path: Path,
    category_embeddings_path: Path,
) -> dict[str, object]:
    return {
        "created_at": utc_now_iso(),
        "question_id": question_id,
        "row_count": row_count,
        "category_count": category_count,
        "embedding_model": embedding_model,
        "confidence_threshold": confidence_threshold,
        "input_screened_path": str(screened_path),
        "input_screened_sha1": file_sha1(screened_path),
        "input_embeddings_path": str(embeddings_path),
        "input_embeddings_sha1": file_sha1(embeddings_path),
        "input_category_master_path": str(category_master_path),
        "input_category_master_sha1": file_sha1(category_master_path),
        "category_embeddings_path": str(category_embeddings_path),
    }


def run_classification_metadata_validations(
    payload: dict[str, object],
    *,
    screened_path: Path,
    embeddings_path: Path,
    category_master_path: Path,
    category_embeddings_path: Path,
    final_labels_path: Path,
) -> list[str]:
    errors: list[str] = []
    for key in CLASSIFICATION_METADATA_KEYS:
        if key not in payload:
            errors.append(f"Missing key: {key}")
    if payload.get("question_id", "") == "":
        errors.append("question_id must not be blank")
    if not isinstance(payload.get("row_count"), int) or payload["row_count"] < 0:
        errors.append("row_count must be a non-negative integer")
    if not isinstance(payload.get("category_count"), int) or payload["category_count"] <= 0:
        errors.append("category_count must be a positive integer")
    try:
        threshold = float(payload.get("confidence_threshold"))
    except (TypeError, ValueError):
        errors.append("confidence_threshold must be numeric")
    else:
        if threshold < 0 or threshold > 1:
            errors.append("confidence_threshold must be between 0 and 1")
    if payload.get("embedding_model", "") == "":
        errors.append("embedding_model must not be blank")
    expected_paths = {
        "input_screened_path": str(screened_path),
        "input_embeddings_path": str(embeddings_path),
        "input_category_master_path": str(category_master_path),
        "category_embeddings_path": str(category_embeddings_path),
    }
    for key, expected in expected_paths.items():
        if payload.get(key) != expected:
            errors.append(f"{key} does not match the expected path")
    sha1_expectations = {
        "input_screened_sha1": file_sha1(screened_path),
        "input_embeddings_sha1": file_sha1(embeddings_path),
        "input_category_master_sha1": file_sha1(category_master_path),
    }
    for key, expected in sha1_expectations.items():
        if payload.get(key) != expected:
            errors.append(f"{key} does not match the expected file")
    if not category_embeddings_path.exists():
        errors.append("category_embeddings.npy must exist when writing classification metadata")
    if not final_labels_path.exists():
        errors.append("final_labels.csv must exist when writing classification metadata")
    elif int(len(read_csv(final_labels_path))) != payload.get("row_count"):
        errors.append("row_count does not match final_labels.csv row count")
    return errors


def request_category_embeddings(client: OpenAI, category_master_df: pd.DataFrame, model: str) -> np.ndarray:
    inputs = category_master_df.apply(build_category_embedding_input, axis=1).tolist()
    if not inputs:
        return np.empty((0, 0), dtype=np.float32)
    response = client.embeddings.create(model=model, input=inputs)
    if len(response.data) != len(inputs):
        raise ValueError(f"Category embedding response count mismatch: {len(response.data)} != {len(inputs)}")
    return np.asarray([item.embedding for item in response.data], dtype=np.float32)


def build_final_labels_df(
    responses_df: pd.DataFrame,
    response_embeddings: np.ndarray,
    category_master_df: pd.DataFrame,
    category_embeddings: np.ndarray,
    question_id: str,
    confidence_threshold: float,
) -> pd.DataFrame:
    filtered = responses_df[
        (responses_df["question_id"] == question_id)
        & (responses_df["is_target"].astype(str).str.lower() == "true")
    ].copy()
    if len(filtered) == 0:
        return pd.DataFrame(columns=FINAL_LABEL_COLUMNS)

    similarity = cosine_similarity_matrix(response_embeddings, category_embeddings)
    best_indices = similarity.argmax(axis=1)
    best_scores = similarity[np.arange(len(filtered)), best_indices]

    predicted_ids: list[str] = []
    predicted_names: list[str] = []
    confidences: list[float] = []
    reasons: list[str] = []
    needs_review: list[bool] = []

    for idx, best_index in enumerate(best_indices):
        category_row = category_master_df.iloc[int(best_index)]
        confidence = to_confidence(float(best_scores[idx]))
        below_threshold = confidence < confidence_threshold
        if below_threshold:
            predicted_ids.append(FALLBACK_CATEGORY_ID)
            predicted_names.append(FALLBACK_CATEGORY_NAME)
            needs_review.append(True)
        else:
            predicted_ids.append(str(category_row["category_id"]))
            predicted_names.append(str(category_row["category_name"]))
            needs_review.append(False)
        confidences.append(confidence)
        reasons.append(
            "nearest_category="
            f"{category_row['category_id']}:{category_row['category_name']};"
            f"confidence={confidence:.3f};"
            f"threshold={confidence_threshold:.3f}"
        )

    filtered["predicted_category_id"] = predicted_ids
    filtered["predicted_category_name"] = predicted_names
    filtered["confidence"] = confidences
    filtered["reason"] = reasons
    filtered["needs_human_review"] = needs_review
    return filtered[FINAL_LABEL_COLUMNS]


def validate_predicted_category_ids(final_labels_df: pd.DataFrame, category_master_df: pd.DataFrame) -> list[str]:
    allowed_category_ids = set(category_master_df["category_id"].astype(str).tolist()) | {FALLBACK_CATEGORY_ID}
    predicted_ids = set(final_labels_df["predicted_category_id"].astype(str).tolist())
    invalid_ids = sorted(predicted_ids - allowed_category_ids)
    if not invalid_ids:
        return []
    return [f"predicted_category_id contains values not found in category_master.csv: {', '.join(invalid_ids)}"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vector-nearest final classification for one question.")
    parser.add_argument("--input", required=True, type=Path, help="Path to screened_responses.csv")
    parser.add_argument("--question-id", required=True, help="Target question_id")
    parser.add_argument("--embeddings", required=True, type=Path, help="Path to 03_embeddings/embeddings.npy")
    parser.add_argument("--category-master", required=True, type=Path, help="Path to category master CSV")
    parser.add_argument("--output-dir", required=True, type=Path, help="Path to 05_classification directory")
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    responses_df = read_csv(args.input)
    validate_required_columns(responses_df, SCREENED_COLUMNS)
    screened_errors = run_screened_validations(responses_df)
    if screened_errors:
        raise SystemExit("\n".join(screened_errors))

    category_master_df = read_csv(args.category_master)
    validate_required_columns(category_master_df, CATEGORY_MASTER_COLUMNS)
    category_master_errors = run_category_master_validations(category_master_df)
    if category_master_errors:
        raise SystemExit("\n".join(category_master_errors))

    target_rows = responses_df[
        (responses_df["question_id"] == args.question_id)
        & (responses_df["is_target"].astype(str).str.lower() == "true")
    ].copy()
    response_embeddings = np.load(args.embeddings)
    embedding_count_errors = validate_embeddings_row_count(response_embeddings, int(len(target_rows)))
    if embedding_count_errors:
        raise SystemExit("\n".join(embedding_count_errors))

    embedding_metadata_path = args.embeddings.parent / "embedding_metadata.json"
    embedding_metadata = load_embedding_metadata(
        metadata_path=embedding_metadata_path,
        embeddings_path=args.embeddings,
        screened_path=args.input,
    )
    model = str(embedding_metadata["model"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_labels_path = args.output_dir / "final_labels.csv"
    category_embeddings_path = args.output_dir / "category_embeddings.npy"
    metadata_path = args.output_dir / "classification_metadata.json"

    client = OpenAI()
    category_embeddings = request_category_embeddings(client=client, category_master_df=category_master_df, model=model)
    if len(category_embeddings) != len(category_master_df):
        raise SystemExit("category_embeddings.npy row count does not match category_master.csv row count")
    final_labels_df = build_final_labels_df(
        responses_df=responses_df,
        response_embeddings=response_embeddings,
        category_master_df=category_master_df,
        category_embeddings=category_embeddings,
        question_id=args.question_id,
        confidence_threshold=args.confidence_threshold,
    )
    final_label_errors = run_final_label_validations(final_labels_df)
    final_label_errors.extend(validate_predicted_category_ids(final_labels_df, category_master_df))
    if final_label_errors:
        raise SystemExit("\n".join(final_label_errors))

    np.save(category_embeddings_path, category_embeddings)
    write_csv(final_labels_df, final_labels_path)

    metadata_payload = build_metadata_payload(
        question_id=args.question_id,
        row_count=int(len(final_labels_df)),
        category_count=int(len(category_master_df)),
        embedding_model=model,
        confidence_threshold=args.confidence_threshold,
        screened_path=args.input,
        embeddings_path=args.embeddings,
        category_master_path=args.category_master,
        category_embeddings_path=category_embeddings_path,
    )
    metadata_errors = run_classification_metadata_validations(
        metadata_payload,
        screened_path=args.input,
        embeddings_path=args.embeddings,
        category_master_path=args.category_master,
        category_embeddings_path=category_embeddings_path,
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
                "embeddings": str(args.embeddings),
                "category_master": str(args.category_master),
                "output_dir": str(args.output_dir),
                "question_id": args.question_id,
                "row_count": int(len(final_labels_df)),
                "category_count": int(len(category_master_df)),
                "embedding_model": model,
                "confidence_threshold": args.confidence_threshold,
                "created_at": utc_now_iso(),
            },
            args.log,
        )


if __name__ == "__main__":
    main()
