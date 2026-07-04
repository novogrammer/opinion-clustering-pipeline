from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

from common import append_jsonl, read_csv, utc_now_iso


REQUIRED_KEYS = [
    "created_at",
    "row_count",
    "input_screened_path",
    "input_screened_sha1",
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
    screened_path: Path | None = None,
    embeddings_path: Path | None = None,
    clusters_path: Path | None = None,
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
    if "input_screened_path" in payload and str(payload["input_screened_path"]).strip() == "":
        errors.append("input_screened_path must not be blank")
    if "input_embeddings_path" in payload and str(payload["input_embeddings_path"]).strip() == "":
        errors.append("input_embeddings_path must not be blank")
    if "input_screened_sha1" in payload and not is_sha1_text(payload["input_screened_sha1"]):
        errors.append("input_screened_sha1 must be a 40-character SHA-1 hex string")
    if "input_embeddings_sha1" in payload and not is_sha1_text(payload["input_embeddings_sha1"]):
        errors.append("input_embeddings_sha1 must be a 40-character SHA-1 hex string")

    if screened_path is not None:
        if payload.get("input_screened_path") != str(screened_path):
            errors.append("input_screened_path does not match the provided screened path")
        elif payload.get("input_screened_sha1") != file_sha1(screened_path):
            errors.append("input_screened_sha1 does not match the provided screened file")

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
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 04_clustering/clustering_metadata.json")
    parser.add_argument("--input", required=True, type=Path, help="Path to clustering_metadata.json")
    parser.add_argument("--screened", type=Path, default=None, help="Optional path to screened_responses.csv")
    parser.add_argument("--embeddings", type=Path, default=None, help="Optional path to embeddings.npy")
    parser.add_argument("--clusters", type=Path, default=None, help="Optional path to clusters.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append validation results as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    errors = run_validations(
        payload,
        screened_path=args.screened,
        embeddings_path=args.embeddings,
        clusters_path=args.clusters,
    )
    log_payload = {
        "event": "validate_clustering_metadata",
        "input": str(args.input),
        "screened": None if args.screened is None else str(args.screened),
        "embeddings": None if args.embeddings is None else str(args.embeddings),
        "clusters": None if args.clusters is None else str(args.clusters),
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
