from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clustering_common import CLUSTER_COLUMNS, SUMMARY_COLUMNS
from common import append_jsonl, read_csv, utc_now_iso, write_csv, write_json
from validate_cluster_summary import run_validations as run_cluster_summary_validations
from validate_clustering_metadata import run_validations as run_clustering_metadata_validations
from validate_clusters import run_validations as run_clusters_validations
from validate_embedding_requests import run_validations as run_embedding_request_validations
from validate_embeddings_array import run_validations as run_embeddings_array_validations


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_metadata_payload(
    *,
    requests_path: Path,
    embeddings_path: Path,
    row_count: int,
    parameters: dict[str, object],
) -> dict[str, object]:
    return {
        "created_at": utc_now_iso(),
        "row_count": row_count,
        "input_requests_path": str(requests_path),
        "input_requests_sha1": file_sha1(requests_path),
        "input_embeddings_path": str(embeddings_path),
        "input_embeddings_sha1": file_sha1(embeddings_path),
        "parameters": parameters,
    }


def clustering_outputs_reusable(
    *,
    metadata_path: Path,
    clusters_path: Path,
    summary_path: Path,
    requests_path: Path,
    embeddings_path: Path,
    expected_row_count: int,
    expected_parameters: dict[str, object],
) -> bool:
    if not (metadata_path.exists() and clusters_path.exists() and summary_path.exists()):
        return False
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return (
        payload.get("row_count") == expected_row_count
        and payload.get("input_requests_path") == str(requests_path)
        and payload.get("input_requests_sha1") == file_sha1(requests_path)
        and payload.get("input_embeddings_path") == str(embeddings_path)
        and payload.get("input_embeddings_sha1") == file_sha1(embeddings_path)
        and payload.get("parameters") == expected_parameters
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BERTopic clustering for one question.")
    parser.add_argument("--input", required=True, type=Path, help="Path to embedding_requests.csv")
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


def build_topic_label(topic_model: BERTopic, topic_id: int) -> str:
    if topic_id == -1:
        return "outlier"
    topic_terms = topic_model.get_topic(topic_id) or []
    if not topic_terms:
        return f"topic_{topic_id}"
    return ", ".join(term for term, _score in topic_terms[:5])


def build_clusters_df(requests_df: pd.DataFrame, topics: list[int], probabilities: np.ndarray | None) -> pd.DataFrame:
    if probabilities is None:
        probability_values = [None] * len(topics)
    elif probabilities.ndim == 1:
        probability_values = probabilities.tolist()
    else:
        probability_values = probabilities.max(axis=1).tolist()

    clusters_df = pd.DataFrame(
        {
            "response_id": requests_df["response_id"],
            "question_id": requests_df["question_id"],
            "topic_id": topics,
            "topic_probability": probability_values,
        }
    )
    clusters_df["is_outlier"] = clusters_df["topic_id"] == -1
    return clusters_df[CLUSTER_COLUMNS]


def select_representative_answers(topic_rows: pd.DataFrame, limit: int = 5) -> list[str]:
    if len(topic_rows) == 0:
        return []

    working = topic_rows.copy()
    if "topic_probability" not in working.columns:
        working["topic_probability"] = None
    working = working[["embedding_input_text", "topic_probability"]].copy()
    working["embedding_input_text"] = working["embedding_input_text"].astype(str)
    working = working.drop_duplicates(subset=["embedding_input_text"], keep="first").reset_index(drop=True)
    if len(working) <= limit:
        return working["embedding_input_text"].tolist()

    if working["topic_probability"].notna().sum() == 0:
        step = max(1, len(working) // limit)
        sampled = working.iloc[::step].head(limit)
        return sampled["embedding_input_text"].tolist()

    selected: list[str] = []

    def append_unique(values: list[str]) -> None:
        for value in values:
            if value not in selected:
                selected.append(value)
            if len(selected) >= limit:
                return

    high_examples = (
        working.sort_values(by=["topic_probability", "embedding_input_text"], ascending=[False, True], kind="stable")
        ["embedding_input_text"]
        .head(2)
        .tolist()
    )
    low_examples = (
        working.sort_values(by=["topic_probability", "embedding_input_text"], ascending=[True, True], kind="stable")
        ["embedding_input_text"]
        .head(2)
        .tolist()
    )
    middle_examples = (
        working.sort_values(by=["topic_probability", "embedding_input_text"], ascending=[False, True], kind="stable")
        .iloc[[len(working) // 2]]
        ["embedding_input_text"]
        .tolist()
    )

    append_unique(high_examples)
    append_unique(middle_examples)
    append_unique(low_examples)

    if len(selected) < limit:
        append_unique(working["embedding_input_text"].tolist())

    return selected[:limit]


def build_summary_df(requests_df: pd.DataFrame, clusters_df: pd.DataFrame, topic_model: BERTopic) -> pd.DataFrame:
    merged = requests_df.merge(clusters_df, on=["response_id", "question_id"], how="inner")
    summaries: list[dict[str, object]] = []

    for topic_id, topic_rows in merged.groupby("topic_id", sort=True):
        representative_answers = select_representative_answers(topic_rows)
        topic_label = build_topic_label(topic_model, int(topic_id))
        confidence = topic_rows["topic_probability"].dropna().mean()
        summaries.append(
            {
                "question_id": str(topic_rows["question_id"].iloc[0]),
                "topic_id": int(topic_id),
                "cluster_size": int(len(topic_rows)),
                "representative_answers": " || ".join(representative_answers),
                "candidate_label": topic_label,
                "candidate_definition": "",
                "include_criteria": "",
                "exclude_criteria": "",
                "split_suggestion": "",
                "confidence": None if pd.isna(confidence) else float(confidence),
            }
        )

    return pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)


def build_single_topic_outputs(requests_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clusters_df = pd.DataFrame(
        {
            "response_id": requests_df["response_id"],
            "question_id": requests_df["question_id"],
            "topic_id": [0] * len(requests_df),
            "topic_probability": [1.0] * len(requests_df),
            "is_outlier": [False] * len(requests_df),
        },
        columns=CLUSTER_COLUMNS,
    )
    summary_df = pd.DataFrame(
        [
            {
                "question_id": str(requests_df["question_id"].iloc[0]),
                "topic_id": 0,
                "cluster_size": int(len(requests_df)),
                "representative_answers": " || ".join(select_representative_answers(requests_df)),
                "candidate_label": "single_cluster",
                "candidate_definition": "",
                "include_criteria": "",
                "exclude_criteria": "",
                "split_suggestion": "",
                "confidence": 1.0,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )
    return clusters_df, summary_df


def main() -> None:
    args = parse_args()
    requests_df = read_csv(args.input)
    embeddings = np.load(args.embeddings)
    request_errors = run_embedding_request_validations(requests_df)
    if request_errors:
        raise SystemExit("\n".join(request_errors))
    embedding_array_errors = run_embeddings_array_validations(
        embeddings,
        request_row_count=int(len(requests_df)),
    )
    if embedding_array_errors:
        raise SystemExit("\n".join(embedding_array_errors))

    if len(requests_df) != len(embeddings):
        raise ValueError("embedding_requests.csv row count does not match embeddings.npy row count")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clusters_path = args.output_dir / "clusters.csv"
    summary_path = args.output_dir / "cluster_summary.csv"
    metadata_path = args.output_dir / "clustering_metadata.json"

    docs = requests_df["embedding_input_text"].tolist()
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
            summary_path=summary_path,
            requests_path=args.input,
            embeddings_path=args.embeddings,
            expected_row_count=int(len(requests_df)),
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
                    "row_count": int(len(requests_df)),
                    "parameters": requested_parameters,
                    "created_at": utc_now_iso(),
                },
                args.log,
            )
        return

    if len(docs) == 0:
        clusters_df = pd.DataFrame(columns=CLUSTER_COLUMNS)
        summary_df = pd.DataFrame(columns=SUMMARY_COLUMNS)
        model_params = {
            "umap_n_neighbors": args.umap_n_neighbors,
            "umap_n_components": args.umap_n_components,
            "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
            "hdbscan_min_samples": args.hdbscan_min_samples,
            "random_state": args.random_state,
        }
    elif len(docs) == 1:
        clusters_df, summary_df = build_single_topic_outputs(requests_df)
        model_params = {
            "mode": "single_document_fallback",
            "random_state": args.random_state,
        }
    else:
        args.umap_n_neighbors = min(args.umap_n_neighbors, max(2, len(docs) - 1))
        args.umap_n_components = min(args.umap_n_components, embeddings.shape[1], len(docs) - 1)
        args.hdbscan_min_cluster_size = min(args.hdbscan_min_cluster_size, len(docs))
        args.hdbscan_min_samples = min(args.hdbscan_min_samples, max(1, len(docs) - 1))
        topic_model = build_topic_model(args)
        topics, probabilities = topic_model.fit_transform(docs, embeddings)
        clusters_df = build_clusters_df(requests_df, topics, probabilities)
        summary_df = build_summary_df(requests_df, clusters_df, topic_model)
        model_params = {
            "umap_n_neighbors": args.umap_n_neighbors,
            "umap_n_components": args.umap_n_components,
            "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
            "hdbscan_min_samples": args.hdbscan_min_samples,
            "random_state": args.random_state,
        }

    write_csv(clusters_df, clusters_path)
    write_csv(summary_df, summary_path)
    cluster_errors = run_clusters_validations(clusters_df)
    if cluster_errors:
        raise SystemExit("\n".join(cluster_errors))
    summary_errors = run_cluster_summary_validations(summary_df, clusters_df=clusters_df)
    if summary_errors:
        raise SystemExit("\n".join(summary_errors))
    if args.log is not None:
        append_jsonl(
            {
                "event": "clustering",
                "input": str(args.input),
                "embeddings": str(args.embeddings),
                "output_dir": str(args.output_dir),
                "row_count": int(len(requests_df)),
                "parameters": model_params,
                "created_at": utc_now_iso(),
            },
            args.log,
        )
    metadata_payload = build_metadata_payload(
        requests_path=args.input,
        embeddings_path=args.embeddings,
        row_count=int(len(requests_df)),
        parameters=model_params,
    )
    metadata_errors = run_clustering_metadata_validations(
        metadata_payload,
        requests_path=args.input,
        embeddings_path=args.embeddings,
        clusters_path=clusters_path,
        summary_path=summary_path,
    )
    if metadata_errors:
        raise SystemExit("\n".join(metadata_errors))
    write_json(metadata_payload, metadata_path)


if __name__ == "__main__":
    main()
