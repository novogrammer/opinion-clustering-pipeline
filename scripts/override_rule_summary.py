from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns, write_csv
from override_common import OVERRIDE_RULE_HIT_COLUMNS, OVERRIDE_RULE_SUMMARY_COLUMNS as OUTPUT_COLUMNS
from validate_override_rule_hits import run_validations as run_override_rule_hit_validations
from validate_override_rule_summary import run_validations as run_override_rule_summary_validations


def build_summary_df(hits_df: pd.DataFrame) -> pd.DataFrame:
    if len(hits_df) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    rows: list[dict[str, object]] = []
    for (
        rule_id,
        question_id,
        match_type,
        pattern,
        override_category_id,
        override_category_name,
        needs_human_review,
        priority,
        note,
    ), group in hits_df.groupby(
        [
            "rule_id",
            "question_id",
            "match_type",
            "pattern",
            "override_category_id",
            "override_category_name",
            "needs_human_review",
            "priority",
            "note",
        ],
        sort=True,
    ):
        response_ids = group["response_id"].astype(str).tolist()
        rows.append(
            {
                "rule_id": str(rule_id),
                "question_id": str(question_id),
                "match_type": str(match_type),
                "pattern": str(pattern),
                "override_category_id": str(override_category_id),
                "override_category_name": str(override_category_name),
                "needs_human_review": str(needs_human_review).lower(),
                "priority": str(priority),
                "hit_count": int(len(group)),
                "unique_answer_count": int(group["answer_text"].astype(str).nunique()),
                "sample_response_ids": "|".join(response_ids[:10]),
                "note": str(note),
            }
        )

    summary_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    summary_df["_priority_int"] = pd.to_numeric(summary_df["priority"], errors="coerce").fillna(9999)
    summary_df = summary_df.sort_values(
        by=["hit_count", "_priority_int", "rule_id"],
        ascending=[False, True, True],
        kind="stable",
    ).drop(columns=["_priority_int"]).reset_index(drop=True)
    return summary_df[OUTPUT_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate override_rule_hits.csv by rule_id")
    parser.add_argument("--input", required=True, type=Path, help="Path to override_rule_hits.csv")
    parser.add_argument("--output", required=True, type=Path, help="Path to override_rule_summary.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hits_df = read_csv(args.input)
    validate_required_columns(hits_df, OVERRIDE_RULE_HIT_COLUMNS)
    hit_errors = run_override_rule_hit_validations(hits_df)
    if hit_errors:
        raise SystemExit("\n".join(hit_errors))
    summary_df = build_summary_df(hits_df)
    summary_errors = run_override_rule_summary_validations(summary_df)
    if summary_errors:
        raise SystemExit("\n".join(summary_errors))
    write_csv(summary_df, args.output)
    if args.log is not None:
        append_jsonl(
            {
                "event": "override_rule_summary",
                "input": str(args.input),
                "output": str(args.output),
                "row_count": int(len(summary_df)),
                "created_at": utc_now_iso(),
            },
            args.log,
        )


if __name__ == "__main__":
    main()
