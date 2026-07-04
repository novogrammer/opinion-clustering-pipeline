from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from openai import OpenAI

from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns, write_csv, write_json
from screening import SCREENED_COLUMNS, run_validations as run_screened_validations
from clustering import CLUSTER_COLUMNS, run_cluster_validations


REPRESENTATIVE_COLUMNS = ["topic_id", "response_id", "question_id", "answer_text", "topic_probability", "representative_rank"]
CATEGORY_MASTER_COLUMNS = [
    "category_id",
    "category_name",
    "category_definition",
    "representative_examples",
]
CURATION_METADATA_KEYS = [
    "created_at",
    "question_id",
    "draft_model",
    "row_count",
    "cluster_count",
    "input_screened_path",
    "input_clusters_path",
]


def validate_required_text(df: pd.DataFrame, columns: list[str]) -> list[str]:
    errors: list[str] = []
    for column in columns:
        blank_mask = df[column].map(lambda value: str(value).strip() == "")
        count = int(blank_mask.sum())
        if count > 0:
            errors.append(f"Blank values found in {column}: {count}")
    return errors


def validate_no_duplicate_response_ids(df: pd.DataFrame) -> list[str]:
    duplicate_mask = df["response_id"].duplicated(keep=False)
    duplicates = df.loc[duplicate_mask, "response_id"].tolist()
    if not duplicates:
        return []
    joined = ", ".join(dict.fromkeys(duplicates))
    return [f"Duplicate response_id values found: {joined}"]


def run_representative_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if len(df) == 0:
        return []
    errors.extend(validate_no_duplicate_response_ids(df))
    errors.extend(validate_required_text(df, ["topic_id", "response_id", "question_id", "answer_text", "representative_rank"]))
    return errors


def validate_no_duplicate_category_ids(df: pd.DataFrame) -> list[str]:
    duplicate_mask = df["category_id"].duplicated(keep=False)
    duplicates = df.loc[duplicate_mask, "category_id"].tolist()
    if not duplicates:
        return []
    joined = ", ".join(dict.fromkeys(duplicates))
    return [f"Duplicate category_id values found: {joined}"]


def run_category_master_validations(df: pd.DataFrame, *, label: str = "category_master.csv") -> list[str]:
    errors: list[str] = []
    if len(df) == 0:
        return [f"{label} must contain at least one category"]
    errors.extend(validate_no_duplicate_category_ids(df))
    errors.extend(validate_required_text(df, CATEGORY_MASTER_COLUMNS))
    return errors


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
            "このクラスタのカテゴリ草案を JSON で返してください。",
            'keys: category_id, category_name, category_definition, representative_examples',
            "category_id は CAT001 のような英数字IDにしてください。",
            "representative_examples は 1 つの文字列で返してください。",
        ]
    )
    return "\n".join(lines)


def parse_draft_response(content: str) -> dict[str, str]:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    payload = json.loads(stripped or "{}")
    return {
        "category_id": str(payload.get("category_id", "")).strip(),
        "category_name": str(payload.get("category_name", "")).strip(),
        "category_definition": str(payload.get("category_definition", "")).strip(),
        "representative_examples": str(payload.get("representative_examples", "")).strip(),
    }


def request_category_draft(client: OpenAI, model: str, question_id: str, topic_id: str, representatives: pd.DataFrame) -> dict[str, str]:
    prompt = build_draft_prompt(question_id=question_id, topic_id=topic_id, representatives=representatives)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You create short Japanese category drafts and always return valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return parse_draft_response(response.choices[0].message.content or "{}")


def build_category_master_draft_df(representatives_df: pd.DataFrame, question_id: str, draft_model: str) -> pd.DataFrame:
    if len(representatives_df) == 0:
        return pd.DataFrame(columns=CATEGORY_MASTER_COLUMNS)
    client = OpenAI()
    rows: list[dict[str, str]] = []
    for topic_id, group in representatives_df.groupby("topic_id", sort=True):
        draft = request_category_draft(
            client=client,
            model=draft_model,
            question_id=question_id,
            topic_id=str(topic_id),
            representatives=group,
        )
        if draft["category_id"] == "":
            numeric_topic = str(topic_id).isdigit()
            if numeric_topic and int(topic_id) >= 0:
                draft["category_id"] = f"CAT{int(topic_id) + 1:03d}"
            else:
                draft["category_id"] = f"CAT{len(rows) + 1:03d}"
        rows.append(draft)
    return pd.DataFrame(rows, columns=CATEGORY_MASTER_COLUMNS)


def build_metadata_payload(
    *,
    question_id: str,
    draft_model: str,
    row_count: int,
    cluster_count: int,
    screened_path: Path,
    clusters_path: Path,
) -> dict[str, object]:
    return {
        "created_at": utc_now_iso(),
        "question_id": question_id,
        "draft_model": draft_model,
        "row_count": row_count,
        "cluster_count": cluster_count,
        "input_screened_path": str(screened_path),
        "input_clusters_path": str(clusters_path),
    }


def run_curation_metadata_validations(payload: dict[str, object], draft_path: Path) -> list[str]:
    errors: list[str] = []
    for key in CURATION_METADATA_KEYS:
        if key not in payload:
            errors.append(f"Missing key: {key}")
    if payload.get("question_id", "") == "":
        errors.append("question_id must not be blank")
    if payload.get("draft_model", "") == "":
        errors.append("draft_model must not be blank")
    if not isinstance(payload.get("row_count"), int) or payload["row_count"] < 0:
        errors.append("row_count must be a non-negative integer")
    if not isinstance(payload.get("cluster_count"), int) or payload["cluster_count"] < 0:
        errors.append("cluster_count must be a non-negative integer")
    if not draft_path.exists():
        errors.append("category_master_draft.csv must exist when writing curation metadata")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare curation artifacts for one question.")
    parser.add_argument("--input", required=True, type=Path, help="Path to screened_responses.csv")
    parser.add_argument("--clusters", required=True, type=Path, help="Path to 04_clustering/clusters.csv")
    parser.add_argument("--question-id", required=True, help="Target question_id")
    parser.add_argument("--draft-model", required=True, help="Chat model name used for category drafts")
    parser.add_argument("--output-dir", required=True, type=Path, help="Path to 05_curation directory")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    responses_df = read_csv(args.input)
    validate_required_columns(responses_df, SCREENED_COLUMNS)
    screened_errors = run_screened_validations(responses_df)
    if screened_errors:
        raise SystemExit("\n".join(screened_errors))

    clusters_df = read_csv(args.clusters)
    validate_required_columns(clusters_df, CLUSTER_COLUMNS)
    cluster_errors = run_cluster_validations(clusters_df)
    if cluster_errors:
        raise SystemExit("\n".join(cluster_errors))

    target_rows = responses_df[
        (responses_df["question_id"] == args.question_id)
        & (responses_df["is_target"].astype(str).str.lower() == "true")
    ].copy()
    if len(target_rows) != len(clusters_df):
        raise SystemExit("clusters.csv row count does not match screened target row count")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    representatives_path = args.output_dir / "cluster_representatives.csv"
    draft_path = args.output_dir / "category_master_draft.csv"
    metadata_path = args.output_dir / "curation_metadata.json"

    representatives_df = build_representatives_df(clusters_df, target_rows)
    representative_errors = run_representative_validations(representatives_df)
    if representative_errors:
        raise SystemExit("\n".join(representative_errors))

    category_master_draft_df = build_category_master_draft_df(
        representatives_df=representatives_df,
        question_id=args.question_id,
        draft_model=args.draft_model,
    )
    category_errors = run_category_master_validations(category_master_draft_df, label="category_master_draft.csv")
    if category_errors:
        raise SystemExit("\n".join(category_errors))

    write_csv(representatives_df, representatives_path)
    write_csv(category_master_draft_df, draft_path)

    metadata_payload = build_metadata_payload(
        question_id=args.question_id,
        draft_model=args.draft_model,
        row_count=int(len(representatives_df)),
        cluster_count=int(clusters_df["topic_id"].nunique()),
        screened_path=args.input,
        clusters_path=args.clusters,
    )
    metadata_errors = run_curation_metadata_validations(metadata_payload, draft_path)
    if metadata_errors:
        raise SystemExit("\n".join(metadata_errors))
    write_json(metadata_payload, metadata_path)
    if args.log is not None:
        append_jsonl(
            {
                "event": "curation",
                "input": str(args.input),
                "clusters": str(args.clusters),
                "question_id": args.question_id,
                "draft_model": args.draft_model,
                "output_dir": str(args.output_dir),
                "row_count": int(len(representatives_df)),
                "cluster_count": int(clusters_df["topic_id"].nunique()),
                "created_at": utc_now_iso(),
            },
            args.log,
        )


if __name__ == "__main__":
    main()
