from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import append_jsonl, read_csv, utc_now_iso


def run_validations(
    embeddings: np.ndarray,
    request_row_count: int | None = None,
    metadata_row_count: int | None = None,
    metadata_status: str | None = None,
) -> list[str]:
    errors: list[str] = []

    if embeddings.ndim not in (1, 2):
        errors.append(f"embeddings.npy must be 1D or 2D, got ndim={embeddings.ndim}")

    row_count = int(len(embeddings))
    if request_row_count is not None and row_count != request_row_count:
        errors.append(
            f"embeddings.npy row count does not match embedding_requests.csv: {row_count} != {request_row_count}"
        )
    if metadata_row_count is not None and row_count != metadata_row_count:
        errors.append(
            f"embeddings.npy row count does not match embedding_metadata.json row_count: {row_count} != {metadata_row_count}"
        )
    if metadata_status is not None and metadata_status != "completed":
        errors.append(f"embeddings.npy requires embedding_metadata.json status=completed, got {metadata_status}")

    if embeddings.ndim == 2 and row_count > 0 and embeddings.shape[1] <= 0:
        errors.append("embeddings.npy has zero embedding dimensions")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 03_embeddings/embeddings.npy")
    parser.add_argument("--input", required=True, type=Path, help="Path to embeddings.npy")
    parser.add_argument("--requests", type=Path, default=None, help="Optional path to embedding_requests.csv")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional path to embedding_metadata.json")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append validation results as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings = np.load(args.input)
    request_row_count = None
    metadata_row_count = None
    metadata_status = None

    if args.requests is not None:
        request_row_count = int(len(read_csv(args.requests)))
    if args.metadata is not None:
        payload = json.loads(args.metadata.read_text(encoding="utf-8"))
        metadata_row_count = payload.get("row_count")
        metadata_status = payload.get("status")

    errors = run_validations(
        embeddings,
        request_row_count=request_row_count,
        metadata_row_count=metadata_row_count,
        metadata_status=metadata_status,
    )
    log_payload = {
        "event": "validate_embeddings_array",
        "input": str(args.input),
        "request_row_count": request_row_count,
        "metadata_row_count": metadata_row_count,
        "metadata_status": metadata_status,
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
