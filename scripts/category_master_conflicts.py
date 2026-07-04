from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

from classification_common import CATEGORY_MASTER_COLUMNS
from classification_keywords import collect_exclude_keywords, collect_category_keywords, normalize_text
from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns, write_csv


OUTPUT_COLUMNS = [
    "left_category_id",
    "left_category_name",
    "right_category_id",
    "right_category_name",
    "shared_keywords",
    "left_keywords_only",
    "right_keywords_only",
    "name_overlap",
    "conflict_level",
    "notes",
]


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        folded = value.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        output.append(value)
    return output


def tokenize_name(value: str) -> set[str]:
    normalized = normalize_text(value)
    tokens = [token.strip() for token in normalized.replace("/", " ").replace("・", " ").split() if token.strip()]
    return {token.casefold() for token in tokens}


def determine_conflict_level(shared_keywords: list[str], name_overlap: bool, left_excludes: list[str], right_excludes: list[str]) -> str:
    if len(shared_keywords) >= 2:
        return "high"
    if len(shared_keywords) == 1 and name_overlap:
        return "high"
    if len(shared_keywords) == 1:
        return "medium"
    if name_overlap:
        return "medium"

    shared_excludes = {
        keyword.casefold() for keyword in left_excludes
    } & {
        keyword.casefold() for keyword in right_excludes
    }
    if shared_excludes:
        return "low"
    return "none"


def build_notes(shared_keywords: list[str], name_overlap: bool, left_excludes: list[str], right_excludes: list[str]) -> str:
    notes: list[str] = []
    if shared_keywords:
        notes.append("include_criteria overlap")
    if name_overlap:
        notes.append("category_name overlap")
    shared_excludes = unique_preserve_order(
        [keyword for keyword in left_excludes if keyword.casefold() in {value.casefold() for value in right_excludes}]
    )
    if shared_excludes:
        notes.append(f"shared exclude={','.join(shared_excludes)}")
    return "; ".join(notes)


def build_conflicts_df(category_master_df: pd.DataFrame) -> pd.DataFrame:
    categories: list[dict[str, object]] = []
    for _, row in category_master_df.iterrows():
        categories.append(
            {
                "category_id": str(row["category_id"]).strip(),
                "category_name": str(row["category_name"]).strip(),
                "keywords": unique_preserve_order(collect_category_keywords(row)),
                "exclude_keywords": unique_preserve_order(collect_exclude_keywords(row)),
                "name_tokens": tokenize_name(str(row["category_name"])),
            }
        )

    rows: list[dict[str, str]] = []
    for left, right in combinations(categories, 2):
        left_keywords_casefold = {keyword.casefold(): keyword for keyword in left["keywords"]}
        right_keywords_casefold = {keyword.casefold(): keyword for keyword in right["keywords"]}
        shared_folds = left_keywords_casefold.keys() & right_keywords_casefold.keys()
        shared_keywords = [left_keywords_casefold[key] for key in shared_folds]
        left_only = [keyword for keyword in left["keywords"] if keyword.casefold() not in shared_folds]
        right_only = [keyword for keyword in right["keywords"] if keyword.casefold() not in shared_folds]
        name_overlap = len(left["name_tokens"] & right["name_tokens"]) > 0
        conflict_level = determine_conflict_level(
            shared_keywords,
            name_overlap,
            left["exclude_keywords"],
            right["exclude_keywords"],
        )
        if conflict_level == "none":
            continue

        rows.append(
            {
                "left_category_id": str(left["category_id"]),
                "left_category_name": str(left["category_name"]),
                "right_category_id": str(right["category_id"]),
                "right_category_name": str(right["category_name"]),
                "shared_keywords": "|".join(shared_keywords),
                "left_keywords_only": "|".join(left_only),
                "right_keywords_only": "|".join(right_only),
                "name_overlap": str(name_overlap).lower(),
                "conflict_level": conflict_level,
                "notes": build_notes(shared_keywords, name_overlap, left["exclude_keywords"], right["exclude_keywords"]),
            }
        )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect potential category definition conflicts in category_master.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to category_master.csv")
    parser.add_argument("--output", required=True, type=Path, help="Path to category_conflicts.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    category_master_df = read_csv(args.input)
    validate_required_columns(category_master_df, CATEGORY_MASTER_COLUMNS)
    conflicts_df = build_conflicts_df(category_master_df)
    write_csv(conflicts_df, args.output)
    if args.log is not None:
        append_jsonl(
            {
                "event": "category_master_conflicts",
                "input": str(args.input),
                "output": str(args.output),
                "row_count": int(len(conflicts_df)),
                "created_at": utc_now_iso(),
            },
            args.log,
        )


if __name__ == "__main__":
    main()
