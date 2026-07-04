from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns, write_csv, write_json
from embeddings import file_sha1, run_embeddings_array_validations
from screening import SCREENED_COLUMNS, run_validations as run_screened_validations


CLUSTER_COLUMNS = ["response_id", "question_id", "topic_id", "topic_probability", "is_outlier"]
REPRESENTATIVE_COLUMNS = ["topic_id", "response_id", "question_id", "answer_text", "topic_probability", "representative_rank"]
DRAFT_COLUMNS = [
    "topic_id",
    "draft_category_name",
    "draft_category_definition",
    "draft_representative_examples",
    "draft_confidence",
    "split_suggestion",
]
CLUSTERING_METADATA_KEYS = [
    "created_at",
    "row_count",
    "input_screened_path",
    "input_screened_sha1",
    "input_embeddings_path",
    "input_embeddings_sha1",
    "parameters",
]


def is_sha1_text(value: object) -> bool:
    text = str(value)
    return len(text) == 40 and all(char in "0123456789abcdef" for char in text.lower())


def validate_no_duplicate_response_ids(df: pd.DataFrame) -> list[str]:
    duplicate_mask = df["response_id"].duplicated(keep=False)
    duplicates = df.loc[duplicate_mask, "response_id"].tolist()
    if not duplicates:
        return []
    joined = ", ".join(dict.fromkeys(duplicates))
    return [f"Duplicate response_id values found: {joined}"]


def validate_probability_values(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for idx, value in enumerate(df["topic_probability"], start=1):
        if str(value).strip() == "":
            continue
        try:
            probability = float(value)
        except (TypeError, ValueError):
            errors.append(f"Invalid topic_probability at row {idx}: {value}")
            continue
        if probability < 0 or probability > 1:
            errors.append(f"Out-of-range topic_probability at row {idx}: {value}")
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


def validate_topic_outlier_consistency(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for idx, row in df.iterrows():
        topic_id = str(row["topic_id"])
        is_outlier = str(row["is_outlier"]).lower() == "true"
        if is_outlier and topic_id != "-1":
            errors.append(f"Row {idx + 1}: is_outlier=true requires topic_id=-1")
        if not is_outlier and topic_id == "-1":
            errors.append(f"Row {idx + 1}: topic_id=-1 requires is_outlier=true")
    return errors


def run_cluster_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_no_duplicate_response_ids(df))
    errors.extend(validate_probability_values(df))
    errors.extend(validate_boolean_column(df, "is_outlier"))
    errors.extend(validate_topic_outlier_consistency(df))
    return errors


def run_clustering_metadata_validations(
    payload: dict[str, object],
    *,
    screened_path: Path | None = None,
    embeddings_path: Path | None = None,
    clusters_path: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    for key in CLUSTERING_METADATA_KEYS:
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
    representatives_path: Path,
    drafts_path: Path,
    screened_path: Path,
    embeddings_path: Path,
    expected_row_count: int,
    expected_parameters: dict[str, object],
) -> bool:
    if not (metadata_path.exists() and clusters_path.exists() and representatives_path.exists() and drafts_path.exists()):
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
    parser.add_argument("--draft-model", required=True, help="Chat model name used for cluster label drafts")
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-n-components", type=int, default=5)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=10)
    parser.add_argument("--hdbscan-min-samples", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Regenerate artifacts even if matching outputs already exist")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    return parser.parse_args()


def build_topic_model(args: argparse.Namespace):
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
            "is_outlier": [topic_id == -1 for topic_id in topics],
        }
    )
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


def build_representatives_df(clusters_df: pd.DataFrame, target_rows: pd.DataFrame, per_topic: int = 3) -> pd.DataFrame:
    if len(clusters_df) == 0:
        return pd.DataFrame(columns=REPRESENTATIVE_COLUMNS)
    merged = clusters_df.merge(
        target_rows[["response_id", "question_id", "answer_text"]],
        on=["response_id", "question_id"],
        how="left",
    )
    representative_frames: list[pd.DataFrame] = []
    for topic_id, group in merged.groupby("topic_id", sort=True):
        ordered = group.sort_values(
            by=["topic_probability", "response_id"],
            ascending=[False, True],
            na_position="last",
        ).head(per_topic).copy()
        ordered["representative_rank"] = range(1, len(ordered) + 1)
        representative_frames.append(ordered[REPRESENTATIVE_COLUMNS])
    if not representative_frames:
        return pd.DataFrame(columns=REPRESENTATIVE_COLUMNS)
    return pd.concat(representative_frames, ignore_index=True)


def build_draft_prompt(question_id: str, topic_id: str, representatives: pd.DataFrame) -> str:
    lines = [
        f"設問ID: {question_id}",
        f"クラスタID: {topic_id}",
        "代表回答:",
    ]
    for _, row in representatives.iterrows():
        lines.append(f"- {row['answer_text']}")
    lines.extend(
        [
            "",
            "このクラスタの草案を JSON で返してください。",
            'keys: draft_category_name, draft_category_definition, draft_representative_examples, draft_confidence, split_suggestion',
            "draft_representative_examples は 1 つの文字列で返してください。",
        ]
    )
    return "\n".join(lines)


def parse_draft_response(content: str, topic_id: str) -> dict[str, str]:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    payload = json.loads(stripped or "{}")
    return {
        "topic_id": str(topic_id),
        "draft_category_name": str(payload.get("draft_category_name", "")).strip(),
        "draft_category_definition": str(payload.get("draft_category_definition", "")).strip(),
        "draft_representative_examples": str(payload.get("draft_representative_examples", "")).strip(),
        "draft_confidence": str(payload.get("draft_confidence", "")).strip(),
        "split_suggestion": str(payload.get("split_suggestion", "")).strip(),
    }


def request_cluster_draft(client: OpenAI, model: str, question_id: str, topic_id: str, representatives: pd.DataFrame) -> dict[str, str]:
    prompt = build_draft_prompt(question_id=question_id, topic_id=str(topic_id), representatives=representatives)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You create short Japanese clustering drafts and always return valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or "{}"
    return parse_draft_response(content, str(topic_id))


def build_drafts_df(representatives_df: pd.DataFrame, question_id: str, draft_model: str) -> pd.DataFrame:
    if len(representatives_df) == 0:
        return pd.DataFrame(columns=DRAFT_COLUMNS)
    client = OpenAI()
    rows: list[dict[str, str]] = []
    for topic_id, group in representatives_df.groupby("topic_id", sort=True):
        rows.append(
            request_cluster_draft(
                client=client,
                model=draft_model,
                question_id=question_id,
                topic_id=str(topic_id),
                representatives=group,
            )
        )
    return pd.DataFrame(rows, columns=DRAFT_COLUMNS)


def main() -> None:
    args = parse_args()

    screened_df = read_csv(args.input)
    validate_required_columns(screened_df, SCREENED_COLUMNS)
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
    representatives_path = args.output_dir / "cluster_representatives.csv"
    drafts_path = args.output_dir / "cluster_label_drafts.csv"
    metadata_path = args.output_dir / "clustering_metadata.json"
    requested_parameters = {
        "draft_model": args.draft_model,
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
            representatives_path=representatives_path,
            drafts_path=drafts_path,
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
        model_params = {"draft_model": args.draft_model, "mode": "single_document_fallback", "random_state": args.random_state}
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
            "draft_model": args.draft_model,
            "umap_n_neighbors": args.umap_n_neighbors,
            "umap_n_components": args.umap_n_components,
            "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
            "hdbscan_min_samples": args.hdbscan_min_samples,
            "random_state": args.random_state,
        }

    cluster_errors = run_cluster_validations(clusters_df)
    if cluster_errors:
        raise SystemExit("\n".join(cluster_errors))
    representatives_df = build_representatives_df(clusters_df, target_rows)
    drafts_df = build_drafts_df(representatives_df, args.question_id, args.draft_model)
    write_csv(clusters_df, clusters_path)
    write_csv(representatives_df, representatives_path)
    write_csv(drafts_df, drafts_path)
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
