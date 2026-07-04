from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

from common import append_jsonl, read_csv, utc_now_iso


REQUIRED_KEYS = [
    "created_at",
    "row_count",
    "input_requests_path",
    "input_requests_sha1",
    "input_embeddings_path",
    "input_embeddings_sha1",
    "parameters",
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
    requests_path: Path | None = None,
    embeddings_path: Path | None = None,
    clusters_path: Path | None = None,
    summary_path: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in payload:
            errors.append(f"Missing key: {key}")
    if "row_count" in payload and not isinstance(payload["row_count"], int):
        errors.append("row_count must be an integer")
    elif "row_count" in payload and payload["row_count"] < 0:
        errors.append("row_count must be greater than or equal to 0")
    if "parameters" in payload and not isinstance(payload["parameters"], dict):
        errors.append("parameters must be an object")
    if "input_requests_path" in payload and str(payload["input_requests_path"]).strip() == "":
        errors.append("input_requests_path must not be blank")
    if "input_embeddings_path" in payload and str(payload["input_embeddings_path"]).strip() == "":
        errors.append("input_embeddings_path must not be blank")
    if "input_requests_sha1" in payload and not is_sha1_text(payload["input_requests_sha1"]):
        errors.append("input_requests_sha1 must be a 40-character SHA-1 hex string")
    if "input_embeddings_sha1" in payload and not is_sha1_text(payload["input_embeddings_sha1"]):
        errors.append("input_embeddings_sha1 must be a 40-character SHA-1 hex string")

    if requests_path is not None:
        if payload.get("input_requests_path") != str(requests_path):
            errors.append("input_requests_path does not match the provided requests path")
        elif payload.get("input_requests_sha1") != file_sha1(requests_path):
            errors.append("input_requests_sha1 does not match the provided requests file")
        elif payload.get("row_count") != int(len(read_csv(requests_path))):
            errors.append("row_count does not match the provided requests file")

    if embeddings_path is not None:
        if payload.get("input_embeddings_path") != str(embeddings_path):
            errors.append("input_embeddings_path does not match the provided embeddings path")
        elif payload.get("input_embeddings_sha1") != file_sha1(embeddings_path):
            errors.append("input_embeddings_sha1 does not match the provided embeddings file")

    if clusters_path is not None:
        if not clusters_path.exists():
            errors.append("clusters.csv must exist when validating clustering metadata")
        elif payload.get("row_count") != int(len(read_csv(clusters_path))):
            errors.append("row_count does not match clusters.csv row count")

    if summary_path is not None and not summary_path.exists():
        errors.append("cluster_summary.csv must exist when validating clustering metadata")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 04_clustering/clustering_metadata.json")
    parser.add_argument("--input", required=True, type=Path, help="Path to clustering_metadata.json")
    parser.add_argument("--requests", type=Path, default=None, help="Optional path to embedding_requests.csv")
    parser.add_argument("--embeddings", type=Path, default=None, help="Optional path to embeddings.npy")
    parser.add_argument("--clusters", type=Path, default=None, help="Optional path to clusters.csv")
    parser.add_argument("--summary", type=Path, default=None, help="Optional path to cluster_summary.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append validation results as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    errors = run_validations(
        payload,
        requests_path=args.requests,
        embeddings_path=args.embeddings,
        clusters_path=args.clusters,
        summary_path=args.summary,
    )
    log_payload = {
        "event": "validate_clustering_metadata",
        "input": str(args.input),
        "requests": None if args.requests is None else str(args.requests),
        "embeddings": None if args.embeddings is None else str(args.embeddings),
        "clusters": None if args.clusters is None else str(args.clusters),
        "summary": None if args.summary is None else str(args.summary),
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
