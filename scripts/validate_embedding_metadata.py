from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from common import append_jsonl, read_csv, utc_now_iso


REQUIRED_KEYS = [
    "model",
    "question_id",
    "row_count",
    "input_screened_path",
    "input_screened_sha1",
    "input_requests_path",
    "input_requests_sha1",
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


def run_validations(
    payload: dict,
    *,
    screened_path: Path | None = None,
    requests_path: Path | None = None,
    embeddings_path: Path | None = None,
    failures_path: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_KEYS:
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
    if "input_requests_path" in payload and str(payload["input_requests_path"]).strip() == "":
        errors.append("input_requests_path must not be blank")
    if "input_screened_sha1" in payload and not is_sha1_text(payload["input_screened_sha1"]):
        errors.append("input_screened_sha1 must be a 40-character SHA-1 hex string")
    if "input_requests_sha1" in payload and not is_sha1_text(payload["input_requests_sha1"]):
        errors.append("input_requests_sha1 must be a 40-character SHA-1 hex string")
    if screened_path is not None:
        if payload.get("input_screened_path") != str(screened_path):
            errors.append("input_screened_path does not match the provided screened path")
        elif payload.get("input_screened_sha1") != file_sha1(screened_path):
            errors.append("input_screened_sha1 does not match the provided screened file")
    if requests_path is not None:
        if payload.get("input_requests_path") != str(requests_path):
            errors.append("input_requests_path does not match the provided requests path")
        elif payload.get("input_requests_sha1") != file_sha1(requests_path):
            errors.append("input_requests_sha1 does not match the provided requests file")
        elif payload.get("row_count") != int(len(read_csv(requests_path))):
            errors.append("row_count does not match the provided requests file")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 03_embeddings/embedding_metadata.json")
    parser.add_argument("--input", required=True, type=Path, help="Path to embedding_metadata.json")
    parser.add_argument("--screened", type=Path, default=None, help="Optional path to screened_responses.csv")
    parser.add_argument("--requests", type=Path, default=None, help="Optional path to embedding_requests.csv")
    parser.add_argument("--embeddings", type=Path, default=None, help="Optional path to embeddings.npy")
    parser.add_argument("--failures", type=Path, default=None, help="Optional path to embedding_failures.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append validation results as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    errors = run_validations(
        payload,
        screened_path=args.screened,
        requests_path=args.requests,
        embeddings_path=args.embeddings,
        failures_path=args.failures,
    )
    log_payload = {
        "event": "validate_embedding_metadata",
        "input": str(args.input),
        "screened": None if args.screened is None else str(args.screened),
        "requests": None if args.requests is None else str(args.requests),
        "embeddings": None if args.embeddings is None else str(args.embeddings),
        "failures": None if args.failures is None else str(args.failures),
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
