from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from category_master_conflicts import OUTPUT_COLUMNS as CATEGORY_CONFLICT_COLUMNS
from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns, write_csv
from review_common import CATEGORY_REVIEW_PRIORITY_COLUMNS as OUTPUT_COLUMNS
from review_common import REVIEW_SUMMARY_COLUMNS
from validate_category_conflicts import run_validations as run_category_conflict_validations
from validate_category_review_priorities import run_validations as run_review_priority_validations
from validate_review_summary import run_validations as run_review_summary_validations


def build_conflict_index(conflicts_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    index: dict[str, dict[str, object]] = {}

    def ensure_entry(category_id: str) -> dict[str, object]:
        if category_id not in index:
            index[category_id] = {
                "pair_count": 0,
                "high_pair_count": 0,
                "counterparts": [],
            }
        return index[category_id]

    for _, row in conflicts_df.iterrows():
        left_id = str(row["left_category_id"]).strip()
        left_name = str(row["left_category_name"]).strip()
        right_id = str(row["right_category_id"]).strip()
        right_name = str(row["right_category_name"]).strip()
        conflict_level = str(row["conflict_level"]).strip().lower()

        left_entry = ensure_entry(left_id)
        right_entry = ensure_entry(right_id)

        left_entry["pair_count"] += 1
        right_entry["pair_count"] += 1
        if conflict_level == "high":
            left_entry["high_pair_count"] += 1
            right_entry["high_pair_count"] += 1

        left_entry["counterparts"].append(f"{right_id}:{right_name}:{conflict_level}")
        right_entry["counterparts"].append(f"{left_id}:{left_name}:{conflict_level}")

    return index


def compute_priority_score(
    *,
    needs_definition_review: bool,
    corrected_count: int,
    correction_rate: float,
    high_priority_count: int,
    conflict_pair_count: int,
    high_conflict_pair_count: int,
) -> float:
    score = 0.0
    if needs_definition_review:
        score += 3.0
    score += corrected_count * 1.5
    score += correction_rate * 5.0
    score += high_priority_count * 0.5
    score += conflict_pair_count * 0.5
    score += high_conflict_pair_count * 1.0
    return round(score, 3)


def build_priority_df(review_summary_df: pd.DataFrame, conflicts_df: pd.DataFrame) -> pd.DataFrame:
    conflict_index = build_conflict_index(conflicts_df)
    rows: list[dict[str, object]] = []

    for _, row in review_summary_df.iterrows():
        category_id = str(row["predicted_category_id"]).strip()
        category_name = str(row["predicted_category_name"]).strip()
        conflict_info = conflict_index.get(
            category_id,
            {"pair_count": 0, "high_pair_count": 0, "counterparts": []},
        )

        needs_definition_review = str(row["needs_definition_review"]).strip().lower() == "true"
        corrected_count = int(row["corrected_count"])
        correction_rate = float(row["correction_rate"])
        high_priority_count = int(row["high_priority_count"])
        conflict_pair_count = int(conflict_info["pair_count"])
        high_conflict_pair_count = int(conflict_info["high_pair_count"])
        priority_score = compute_priority_score(
            needs_definition_review=needs_definition_review,
            corrected_count=corrected_count,
            correction_rate=correction_rate,
            high_priority_count=high_priority_count,
            conflict_pair_count=conflict_pair_count,
            high_conflict_pair_count=high_conflict_pair_count,
        )

        rows.append(
            {
                "question_id": str(row["question_id"]).strip(),
                "category_id": category_id,
                "category_name": category_name,
                "needs_definition_review": str(needs_definition_review).lower(),
                "definition_review_reason": str(row["definition_review_reason"]).strip(),
                "correction_rate": correction_rate,
                "corrected_count": corrected_count,
                "high_priority_count": high_priority_count,
                "conflict_pair_count": conflict_pair_count,
                "high_conflict_pair_count": high_conflict_pair_count,
                "conflict_categories": "|".join(conflict_info["counterparts"]),
                "priority_score": priority_score,
                "priority_rank": 0,
            }
        )

    priority_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if len(priority_df) == 0:
        return priority_df

    priority_df = priority_df.sort_values(
        by=["priority_score", "corrected_count", "high_conflict_pair_count", "category_id"],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    priority_df["priority_rank"] = range(1, len(priority_df) + 1)
    return priority_df[OUTPUT_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine review_summary.csv and category_conflicts.csv into category_review_priorities.csv")
    parser.add_argument("--review-summary", required=True, type=Path, help="Path to review_summary.csv")
    parser.add_argument("--category-conflicts", required=True, type=Path, help="Path to category_conflicts.csv")
    parser.add_argument("--output", required=True, type=Path, help="Path to category_review_priorities.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review_summary_df = read_csv(args.review_summary)
    validate_required_columns(review_summary_df, REVIEW_SUMMARY_COLUMNS)
    review_summary_errors = run_review_summary_validations(review_summary_df)
    if review_summary_errors:
        raise SystemExit("\n".join(review_summary_errors))
    conflicts_df = read_csv(args.category_conflicts)
    validate_required_columns(conflicts_df, CATEGORY_CONFLICT_COLUMNS)
    category_conflict_errors = run_category_conflict_validations(conflicts_df)
    if category_conflict_errors:
        raise SystemExit("\n".join(category_conflict_errors))
    priority_df = build_priority_df(review_summary_df, conflicts_df)
    priority_errors = run_review_priority_validations(priority_df)
    if priority_errors:
        raise SystemExit("\n".join(priority_errors))
    write_csv(priority_df, args.output)
    if args.log is not None:
        append_jsonl(
            {
                "event": "category_review_priorities",
                "review_summary": str(args.review_summary),
                "category_conflicts": str(args.category_conflicts),
                "output": str(args.output),
                "row_count": int(len(priority_df)),
                "created_at": utc_now_iso(),
            },
            args.log,
        )


if __name__ == "__main__":
    main()
