from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from common import REQUIRED_RESPONSE_COLUMNS, append_jsonl, read_csv, utc_now_iso, validate_required_columns, write_csv, write_json
from screening import SCREENED_COLUMNS, run_validations as run_screened_validations


INPUT_TEMPLATE_VERSION = "v1"
PREPROCESSING_VERSION = "screened_responses_v1"
FAILURE_COLUMNS = ["response_id", "question_id", "embedding_input_text", "error_type", "error_message"]
EMBEDDING_INPUT_TEMPLATE = "設問ID: {question_id}\n質問: {question_text}\n回答: {answer_text}"
EMBEDDING_METADATA_KEYS = [
    "model",
    "question_id",
    "row_count",
    "input_screened_path",
    "input_screened_sha1",
    "input_template_version",
    "input_template",
    "preprocessing_version",
    "batch_size",
    "failed_count",
    "status",
    "created_at",
]


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha1_text(value: object) -> bool:
    text = str(value)
    return len(text) == 40 and all(char in "0123456789abcdef" for char in text.lower())


def validate_failure_required_text(df: pd.DataFrame, columns: list[str]) -> list[str]:
    errors: list[str] = []
    for column in columns:
        blank_mask = df[column].map(lambda value: str(value).strip() == "")
        count = int(blank_mask.sum())
        if count > 0:
            errors.append(f"Blank values found in {column}: {count}")
    return errors


def validate_failure_no_duplicate_response_ids(df: pd.DataFrame) -> list[str]:
    duplicate_mask = df["response_id"].duplicated(keep=False)
    duplicates = df.loc[duplicate_mask, "response_id"].tolist()
    if not duplicates:
        return []
    joined = ", ".join(dict.fromkeys(duplicates))
    return [f"Duplicate response_id values found: {joined}"]


def run_failure_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if len(df) == 0:
        return ["embedding_failures.csv must not be empty when present"]
    errors.extend(validate_failure_required_text(df, FAILURE_COLUMNS))
    errors.extend(validate_failure_no_duplicate_response_ids(df))
    return errors


def run_embedding_metadata_validations(
    payload: dict[str, object],
    *,
    screened_path: Path | None = None,
    embeddings_path: Path | None = None,
    failures_path: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    for key in EMBEDDING_METADATA_KEYS:
        if key not in payload:
            errors.append(f"Missing key: {key}")
    if "row_count" in payload and not isinstance(payload["row_count"], int):
        errors.append("row_count must be an integer")
    elif "row_count" in payload and payload["row_count"] < 0:
        errors.append("row_count must be greater than or equal to 0")
    if "batch_size" in payload and not isinstance(payload["batch_size"], int):
        errors.append("batch_size must be an integer")
    elif "batch_size" in payload and payload["batch_size"] <= 0:
        errors.append("batch_size must be greater than 0")
    if "failed_count" in payload and not isinstance(payload["failed_count"], int):
        errors.append("failed_count must be an integer")
    elif "failed_count" in payload and payload["failed_count"] < 0:
        errors.append("failed_count must be greater than or equal to 0")
    if "status" in payload and payload["status"] not in {"prepared", "completed", "failed"}:
        errors.append("status must be one of: prepared, completed, failed")
    if "input_screened_path" in payload and str(payload["input_screened_path"]).strip() == "":
        errors.append("input_screened_path must not be blank")
    if "input_screened_sha1" in payload and not is_sha1_text(payload["input_screened_sha1"]):
        errors.append("input_screened_sha1 must be a 40-character SHA-1 hex string")
    if screened_path is not None:
        if payload.get("input_screened_path") != str(screened_path):
            errors.append("input_screened_path does not match the provided screened path")
        elif payload.get("input_screened_sha1") != file_sha1(screened_path):
            errors.append("input_screened_sha1 does not match the provided screened file")

    status = payload.get("status")
    failed_count = payload.get("failed_count")

    if status in {"prepared", "completed"} and failed_count not in {0, None}:
        errors.append("failed_count must be 0 when status is prepared or completed")
    if status == "failed" and failed_count == 0:
        errors.append("failed_count must be greater than 0 when status is failed")

    if embeddings_path is not None:
        embeddings_exists = embeddings_path.exists()
        if status == "prepared" and embeddings_exists:
            errors.append("embeddings.npy must not exist when status is prepared")
        if status == "failed" and embeddings_exists:
            errors.append("embeddings.npy must not exist when status is failed")
        if status == "completed" and not embeddings_exists:
            errors.append("embeddings.npy must exist when status is completed")

    if failures_path is not None:
        failures_exists = failures_path.exists()
        if status in {"prepared", "completed"} and failures_exists:
            errors.append("embedding_failures.csv must not exist when status is prepared or completed")
        if status == "failed":
            if not failures_exists:
                errors.append("embedding_failures.csv must exist when status is failed")
            else:
                failure_count = int(len(read_csv(failures_path)))
                if failed_count != failure_count:
                    errors.append("failed_count does not match embedding_failures.csv row count")
    return errors


def run_embeddings_array_validations(
    embeddings: np.ndarray,
    *,
    metadata_row_count: int | None = None,
    metadata_status: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if embeddings.ndim not in (1, 2):
        errors.append(f"embeddings.npy must be 1D or 2D, got ndim={embeddings.ndim}")

    row_count = int(len(embeddings))
    if metadata_row_count is not None and row_count != metadata_row_count:
        errors.append(
            f"embeddings.npy row count does not match embedding_metadata.json row_count: {row_count} != {metadata_row_count}"
        )
    if metadata_status is not None and metadata_status != "completed":
        errors.append(f"embeddings.npy requires embedding_metadata.json status=completed, got {metadata_status}")
    if embeddings.ndim == 2 and row_count > 0 and embeddings.shape[1] <= 0:
        errors.append("embeddings.npy has zero embedding dimensions")
    return errors


def build_embedding_input(question_id: str, question_text: str, answer_text: str) -> str:
    return EMBEDDING_INPUT_TEMPLATE.format(
        question_id=question_id,
        question_text=question_text,
        answer_text=answer_text,
    )


def build_target_rows(df: pd.DataFrame, question_id: str) -> pd.DataFrame:
    filtered = df[(df["question_id"] == question_id) & (df["is_target"].astype(str).str.lower() == "true")].copy()
    filtered["embedding_input_text"] = filtered.apply(
        lambda row: build_embedding_input(
            question_id=str(row["question_id"]),
            question_text=str(row["question_text"]),
            answer_text=str(row["answer_text"]),
        ),
        axis=1,
    )
    return filtered[["response_id", "question_id", "embedding_input_text"]]


def build_failure_rows(batch_df: pd.DataFrame, exc: Exception) -> pd.DataFrame:
    failures = batch_df.copy()
    failures["error_type"] = type(exc).__name__
    failures["error_message"] = str(exc)
    return failures[FAILURE_COLUMNS]


def request_embeddings(
    client: OpenAI,
    targets_df: pd.DataFrame,
    model: str,
    batch_size: int,
    max_retries: int,
    retry_base_seconds: float,
    log_path: Path | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    vectors: list[list[float]] = []
    failure_frames: list[pd.DataFrame] = []

    for start in range(0, len(targets_df), batch_size):
        batch_df = targets_df.iloc[start : start + batch_size].copy()
        batch = batch_df["embedding_input_text"].tolist()
        attempt = 0
        while True:
            try:
                response = client.embeddings.create(model=model, input=batch)
                if len(response.data) != len(batch):
                    raise ValueError(f"Embedding response count mismatch: {len(response.data)} != {len(batch)}")
                vectors.extend(item.embedding for item in response.data)
                break
            except Exception as exc:  # noqa: BLE001
                if log_path is not None:
                    append_jsonl(
                        {
                            "event": "embeddings_retry",
                            "model": model,
                            "batch_start": start,
                            "batch_size": len(batch),
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "created_at": utc_now_iso(),
                        },
                        log_path,
                    )
                if attempt >= max_retries:
                    failure_frames.append(build_failure_rows(batch_df, exc))
                    break
                time.sleep(retry_base_seconds * (2**attempt))
                attempt += 1

    failures_df = pd.concat(failure_frames, ignore_index=True) if failure_frames else pd.DataFrame(columns=FAILURE_COLUMNS)
    return np.asarray(vectors, dtype=np.float32), failures_df


def build_metadata_payload(
    *,
    screened_input_path: Path,
    model: str,
    question_id: str,
    row_count: int,
    batch_size: int,
    failed_count: int,
    status: str,
) -> dict[str, object]:
    return {
        "model": model,
        "question_id": question_id,
        "row_count": row_count,
        "input_screened_path": str(screened_input_path),
        "input_screened_sha1": file_sha1(screened_input_path),
        "input_template_version": INPUT_TEMPLATE_VERSION,
        "input_template": EMBEDDING_INPUT_TEMPLATE,
        "preprocessing_version": PREPROCESSING_VERSION,
        "batch_size": batch_size,
        "failed_count": failed_count,
        "status": status,
        "created_at": utc_now_iso(),
    }


def metadata_matches(existing_path: Path, *, model: str, question_id: str, batch_size: int) -> bool:
    if not existing_path.exists():
        return False
    payload = json.loads(existing_path.read_text(encoding="utf-8"))
    return (
        payload.get("model") == model
        and payload.get("question_id") == question_id
        and payload.get("input_template_version") == INPUT_TEMPLATE_VERSION
        and payload.get("input_template") == EMBEDDING_INPUT_TEMPLATE
        and payload.get("preprocessing_version") == PREPROCESSING_VERSION
        and payload.get("batch_size") == batch_size
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embeddings for one question.")
    parser.add_argument("--input", required=True, type=Path, help="Path to screened_responses.csv")
    parser.add_argument("--question-id", required=True, help="Target question_id")
    parser.add_argument("--output-dir", required=True, type=Path, help="Path to 03_embeddings directory")
    parser.add_argument("--model", default="text-embedding-3-small", help="Embedding model name")
    parser.add_argument("--batch-size", type=int, default=100, help="Embedding request batch size")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retry count per batch after the first attempt")
    parser.add_argument("--retry-base-seconds", type=float, default=1.0, help="Base sleep seconds for exponential backoff")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    parser.add_argument("--force", action="store_true", help="Regenerate artifacts even if matching outputs already exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    df = read_csv(args.input)
    validate_required_columns(df, SCREENED_COLUMNS)
    screened_errors = run_screened_validations(df)
    if screened_errors:
        raise SystemExit("\n".join(screened_errors))

    targets_df = build_target_rows(df, args.question_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = args.output_dir / "embeddings.npy"
    metadata_path = args.output_dir / "embedding_metadata.json"
    failures_path = args.output_dir / "embedding_failures.csv"

    if (
        not args.force
        and metadata_matches(
            metadata_path,
            model=args.model,
            question_id=args.question_id,
            batch_size=args.batch_size,
        )
        and embeddings_path.exists()
        and not failures_path.exists()
    ):
        metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata_payload.get("status") == "completed"
            and metadata_payload.get("row_count") == int(len(targets_df))
            and metadata_payload.get("input_screened_sha1") == file_sha1(args.input)
        ):
            if args.log is not None:
                append_jsonl(
                    {
                        "event": "embeddings_reused",
                        "input": str(args.input),
                        "output_dir": str(args.output_dir),
                        "question_id": args.question_id,
                        "row_count": int(len(targets_df)),
                        "model": args.model,
                        "batch_size": args.batch_size,
                        "created_at": utc_now_iso(),
                    },
                    args.log,
                )
            return

    if failures_path.exists():
        failures_path.unlink()
    if embeddings_path.exists():
        embeddings_path.unlink()

    client = OpenAI()
    if len(targets_df) == 0:
        embeddings = np.empty((0, 0), dtype=np.float32)
        failures_df = pd.DataFrame(columns=FAILURE_COLUMNS)
    else:
        embeddings, failures_df = request_embeddings(
            client=client,
            targets_df=targets_df,
            model=args.model,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            retry_base_seconds=args.retry_base_seconds,
            log_path=args.log,
        )

    if len(failures_df) > 0:
        failure_errors = run_failure_validations(failures_df)
        if failure_errors:
            raise SystemExit("\n".join(failure_errors))
        write_csv(failures_df, failures_path)
        metadata_payload = build_metadata_payload(
            screened_input_path=args.input,
            model=args.model,
            question_id=args.question_id,
            row_count=int(len(targets_df)),
            batch_size=args.batch_size,
            failed_count=int(len(failures_df)),
            status="failed",
        )
        metadata_errors = run_embedding_metadata_validations(
            metadata_payload,
            screened_path=args.input,
            embeddings_path=embeddings_path,
            failures_path=failures_path,
        )
        if metadata_errors:
            raise SystemExit("\n".join(metadata_errors))
        write_json(metadata_payload, metadata_path)
        if args.log is not None:
            append_jsonl(
                {
                    "event": "embeddings_failed",
                    "input": str(args.input),
                    "output_dir": str(args.output_dir),
                    "question_id": args.question_id,
                    "row_count": int(len(targets_df)),
                    "failure_count": int(len(failures_df)),
                    "model": args.model,
                    "batch_size": args.batch_size,
                    "created_at": utc_now_iso(),
                },
                args.log,
            )
        return

    if len(targets_df) > 0:
        embedding_errors = run_embeddings_array_validations(
            embeddings,
            metadata_row_count=int(len(targets_df)),
            metadata_status="completed",
        )
        if embedding_errors:
            raise SystemExit("\n".join(embedding_errors))

    np.save(embeddings_path, embeddings)
    metadata_payload = build_metadata_payload(
        screened_input_path=args.input,
        model=args.model,
        question_id=args.question_id,
        row_count=int(len(targets_df)),
        batch_size=args.batch_size,
        failed_count=0,
        status="completed",
    )
    metadata_errors = run_embedding_metadata_validations(
        metadata_payload,
        screened_path=args.input,
        embeddings_path=embeddings_path,
        failures_path=failures_path,
    )
    if metadata_errors:
        raise SystemExit("\n".join(metadata_errors))
    write_json(metadata_payload, metadata_path)
    if args.log is not None:
        append_jsonl(
            {
                "event": "embeddings",
                "input": str(args.input),
                "output_dir": str(args.output_dir),
                "question_id": args.question_id,
                "row_count": int(len(targets_df)),
                "model": args.model,
                "batch_size": args.batch_size,
                "created_at": utc_now_iso(),
            },
            args.log,
        )


if __name__ == "__main__":
    main()
