from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clustering_common import CLUSTER_COLUMNS
from common import REQUIRED_RESPONSE_COLUMNS, append_jsonl, read_csv, utc_now_iso, validate_required_columns, write_csv, write_json
from validate_clustering_metadata import run_validations as run_clustering_metadata_validations
from validate_clusters import run_validations as run_clusters_validations
from validate_embeddings_array import run_validations as run_embeddings_array_validations
from validate_screened_responses import run_validations as run_screened_validations


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_metadata_payload(
    *,
    screened_path: Path,
    embeddings_path: Path,
    row_count: int,
    parameters: dict[str, object],
) -> dict[str, object]:
    return {
        "created_at": utc_now_iso(),
        "row_count": row_count,
        "input_screened_path": str(screened_path),
        "input_screened_sha1": file_sha1(screened_path),
        "input_embeddings_path": str(embeddings_path),
        "input_embeddings_sha1": file_sha1(embeddings_path),
        "parameters": parameters,
    }


def clustering_outputs_reusable(
    *,
    metadata_path: Path,
    clusters_path: Path,
    screened_path: Path,
    embeddings_path: Path,
    expected_row_count: int,
    expected_parameters: dict[str, object],
) -> bool:
    if not (metadata_path.exists() and clusters_path.exists()):
        return False
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return (
        payload.get("row_count") == expected_row_count
        and payload.get("input_screened_path") == str(screened_path)
        and payload.get("input_screened_sha1") == file_sha1(screened_path)
        and payload.get("input_embeddings_path") == str(embeddings_path)
        and payload.get("input_embeddings_sha1") == file_sha1(embeddings_path)
        and payload.get("parameters") == expected_parameters
    )


def build_target_rows(df: pd.DataFrame, question_id: str) -> pd.DataFrame:
    return df[(df["question_id"] == question_id) & (df["is_target"].astype(str).str.lower() == "true")].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BERTopic clustering for one question.")
    parser.add_argument("--input", required=True, type=Path, help="Path to screened_responses.csv")
    parser.add_argument("--question-id", required=True, help="Target question_id")
    parser.add_argument("--embeddings", required=True, type=Path, help="Path to embeddings.npy")
    parser.add_argument("--output-dir", required=True, type=Path, help="Path to 04_clustering directory")
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-n-components", type=int, default=5)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=10)
    parser.add_argument("--hdbscan-min-samples", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Regenerate artifacts even if matching outputs already exist")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    return parser.parse_args()


def build_topic_model(args: argparse.Namespace) -> BERTopic:
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    umap_model = UMAP(
        n_neighbors=args.umap_n_neighbors,
        n_components=args.umap_n_components,
        metric="cosine",
        random_state=args.random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=args.hdbscan_min_cluster_size,
        min_samples=args.hdbscan_min_samples,
        metric="euclidean",
        prediction_data=True,
    )
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        verbose=False,
    )


def build_clusters_df(target_rows: pd.DataFrame, topics: list[int], probabilities: np.ndarray | None) -> pd.DataFrame:
    if probabilities is None:
        probability_values = [None] * len(topics)
    elif probabilities.ndim == 1:
        probability_values = probabilities.tolist()
    else:
        probability_values = probabilities.max(axis=1).tolist()

    clusters_df = pd.DataFrame(
        {
            "response_id": target_rows["response_id"].astype(str).tolist(),
            "question_id": target_rows["question_id"].astype(str).tolist(),
            "topic_id": topics,
            "topic_probability": probability_values,
        }
    )
    clusters_df["is_outlier"] = clusters_df["topic_id"] == -1
    return clusters_df[CLUSTER_COLUMNS]


def build_single_topic_df(target_rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "response_id": target_rows["response_id"].astype(str).tolist(),
            "question_id": target_rows["question_id"].astype(str).tolist(),
            "topic_id": [0] * len(target_rows),
            "topic_probability": [1.0] * len(target_rows),
            "is_outlier": [False] * len(target_rows),
        },
        columns=CLUSTER_COLUMNS,
    )


def main() -> None:
    args = parse_args()

    screened_df = read_csv(args.input)
    validate_required_columns(screened_df, REQUIRED_RESPONSE_COLUMNS + ["is_target", "screening_reason"])
    screened_errors = run_screened_validations(screened_df)
    if screened_errors:
        raise SystemExit("\n".join(screened_errors))

    target_rows = build_target_rows(screened_df, args.question_id)
    embeddings = np.load(args.embeddings)
    embedding_errors = run_embeddings_array_validations(embeddings)
    if embedding_errors:
        raise SystemExit("\n".join(embedding_errors))
    if len(target_rows) != len(embeddings):
        raise ValueError("screened target row count does not match embeddings.npy row count")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clusters_path = args.output_dir / "clusters.csv"
    metadata_path = args.output_dir / "clustering_metadata.json"
    requested_parameters = {
        "umap_n_neighbors": args.umap_n_neighbors,
        "umap_n_components": args.umap_n_components,
        "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
        "hdbscan_min_samples": args.hdbscan_min_samples,
        "random_state": args.random_state,
    }

    if (
        not args.force
        and clustering_outputs_reusable(
            metadata_path=metadata_path,
            clusters_path=clusters_path,
            screened_path=args.input,
            embeddings_path=args.embeddings,
            expected_row_count=int(len(target_rows)),
            expected_parameters=requested_parameters,
        )
    ):
        if args.log is not None:
            append_jsonl(
                {
                    "event": "clustering_reused",
                    "input": str(args.input),
                    "embeddings": str(args.embeddings),
                    "output_dir": str(args.output_dir),
                    "row_count": int(len(target_rows)),
                    "parameters": requested_parameters,
                    "created_at": utc_now_iso(),
                },
                args.log,
            )
        return

    if len(target_rows) == 0:
        clusters_df = pd.DataFrame(columns=CLUSTER_COLUMNS)
        model_params = requested_parameters
    elif len(target_rows) == 1:
        clusters_df = build_single_topic_df(target_rows)
        model_params = {
            "mode": "single_document_fallback",
            "random_state": args.random_state,
        }
    else:
        args.umap_n_neighbors = min(args.umap_n_neighbors, max(2, len(target_rows) - 1))
        args.umap_n_components = min(args.umap_n_components, embeddings.shape[1], len(target_rows) - 1)
        args.hdbscan_min_cluster_size = min(args.hdbscan_min_cluster_size, len(target_rows))
        args.hdbscan_min_samples = min(args.hdbscan_min_samples, max(1, len(target_rows) - 1))
        topic_model = build_topic_model(args)
        docs = target_rows["answer_text"].astype(str).tolist()
        topics, probabilities = topic_model.fit_transform(docs, embeddings)
        clusters_df = build_clusters_df(target_rows, topics, probabilities)
        model_params = {
            "umap_n_neighbors": args.umap_n_neighbors,
            "umap_n_components": args.umap_n_components,
            "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
            "hdbscan_min_samples": args.hdbscan_min_samples,
            "random_state": args.random_state,
        }

    cluster_errors = run_clusters_validations(clusters_df)
    if cluster_errors:
        raise SystemExit("\n".join(cluster_errors))
    write_csv(clusters_df, clusters_path)
    metadata_payload = build_metadata_payload(
        screened_path=args.input,
        embeddings_path=args.embeddings,
        row_count=int(len(target_rows)),
        parameters=model_params,
    )
    metadata_errors = run_clustering_metadata_validations(
        metadata_payload,
        screened_path=args.input,
        embeddings_path=args.embeddings,
        clusters_path=clusters_path,
    )
    if metadata_errors:
        raise SystemExit("\n".join(metadata_errors))
    write_json(metadata_payload, metadata_path)
    if args.log is not None:
        append_jsonl(
            {
                "event": "clustering",
                "input": str(args.input),
                "embeddings": str(args.embeddings),
                "output_dir": str(args.output_dir),
                "row_count": int(len(target_rows)),
                "parameters": model_params,
                "created_at": utc_now_iso(),
            },
            args.log,
        )


if __name__ == "__main__":
    main()
