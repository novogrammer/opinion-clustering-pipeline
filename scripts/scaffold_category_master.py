from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from classification_common import CATEGORY_MASTER_COLUMNS
from clustering import SUMMARY_COLUMNS
from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns, write_csv


def extract_example_positive(value: str) -> str:
    if not value.strip():
        return ""
    return value.split(" || ")[0].strip()


def is_outlier_row(row: pd.Series) -> bool:
    try:
        topic_id = int(str(row["topic_id"]).strip())
    except ValueError:
        return False
    return topic_id == -1


def build_category_definition(row: pd.Series, category_name: str) -> str:
    candidate_definition = str(row["candidate_definition"]).strip()
    if candidate_definition:
        return candidate_definition
    topic_id = str(row["topic_id"]).strip()
    cluster_size = str(row["cluster_size"]).strip()
    return f"topic_id={topic_id} の代表回答から作成した暫定カテゴリ。cluster_size={cluster_size}。{category_name} に関する回答。"


def build_category_rows(
    summary_df: pd.DataFrame,
    category_prefix: str,
    include_outlier: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    sequence = 1

    for _, row in summary_df.iterrows():
        if not include_outlier and is_outlier_row(row):
            continue

        candidate_label = str(row["candidate_label"]).strip()
        topic_id = str(row["topic_id"]).strip()
        category_name = candidate_label or f"topic_{topic_id}"

        rows.append(
            {
                "category_id": f"{category_prefix}{sequence:03d}",
                "category_name": category_name,
                "category_definition": build_category_definition(row, category_name),
                "include_criteria": str(row["include_criteria"]).strip(),
                "exclude_criteria": str(row["exclude_criteria"]).strip(),
                "example_positive": extract_example_positive(str(row["representative_answers"])),
                "example_negative": "",
            }
        )
        sequence += 1

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scaffold 05_classification/category_master.csv from 04_clustering/cluster_summary.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to cluster_summary.csv")
    parser.add_argument("--output", required=True, type=Path, help="Path to category_master.csv")
    parser.add_argument("--category-prefix", default="CAT", help="Prefix for generated category_id values")
    parser.add_argument("--include-outlier", action="store_true", help="Include topic_id=-1 rows")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_df = read_csv(args.input)
    validate_required_columns(summary_df, SUMMARY_COLUMNS)

    category_rows = build_category_rows(
        summary_df=summary_df,
        category_prefix=args.category_prefix,
        include_outlier=args.include_outlier,
    )
    category_master_df = pd.DataFrame(category_rows, columns=CATEGORY_MASTER_COLUMNS)
    write_csv(category_master_df, args.output)

    if args.log is not None:
        append_jsonl(
            {
                "event": "scaffold_category_master",
                "input": str(args.input),
                "output": str(args.output),
                "row_count": int(len(category_master_df)),
                "include_outlier": args.include_outlier,
                "category_prefix": args.category_prefix,
                "created_at": utc_now_iso(),
            },
            args.log,
        )


if __name__ == "__main__":
    main()
