from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns, write_csv
from review_common import REVIEW_COLUMNS, REVIEW_SUMMARY_COLUMNS as SUMMARY_COLUMNS
from validate_review_log import run_validations as run_review_log_validations
from validate_review_summary import run_validations as run_review_summary_validations


def normalize_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def build_top_review_triggers(series: pd.Series) -> str:
    counter: Counter[str] = Counter()
    for value in series.astype(str):
        for token in value.split("|"):
            token = token.strip()
            if token:
                counter[token] += 1
    if not counter:
        return ""
    return "|".join(f"{token}:{count}" for token, count in counter.most_common(5))


def parse_trigger_counts(value: str) -> dict[str, int]:
    trigger_counts: dict[str, int] = {}
    for part in str(value).split("|"):
        token = part.strip()
        if not token or ":" not in token:
            continue
        name, count_text = token.rsplit(":", 1)
        try:
            trigger_counts[name] = int(count_text)
        except ValueError:
            continue
    return trigger_counts


def determine_definition_review(
    *,
    total_count: int,
    reviewed_count: int,
    corrected_count: int,
    high_priority_count: int,
    correction_rate: float,
    trigger_counts: dict[str, int],
) -> tuple[bool, str]:
    reasons: list[str] = []
    if corrected_count >= 3:
        reasons.append("many_corrections")
    if reviewed_count >= 3 and correction_rate >= 0.3:
        reasons.append("high_correction_rate")
    if total_count > 0 and (high_priority_count / total_count) >= 0.5:
        reasons.append("high_priority_concentration")
    if trigger_counts.get("low_confidence", 0) >= 3:
        reasons.append("low_confidence_cluster")
    if trigger_counts.get("ambiguous_match", 0) >= 2:
        reasons.append("ambiguous_match_cluster")
    if trigger_counts.get("fallback_category", 0) >= 2 or trigger_counts.get("other_category", 0) >= 2:
        reasons.append("other_category_cluster")
    if trigger_counts.get("duplicate_response", 0) >= 2:
        reasons.append("duplicate_response_cluster")
    return (len(reasons) > 0, "|".join(reasons) if reasons else "none")


def build_summary_df(review_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (question_id, category_id, category_name), group in review_df.groupby(
        ["question_id", "predicted_category_id", "predicted_category_name"],
        sort=True,
    ):
        reviewed_mask = group["review_status"].astype(str).str.lower() == "reviewed"
        pending_mask = group["review_status"].astype(str).str.lower() == "pending"
        skipped_mask = group["review_status"].astype(str).str.lower() == "skipped"
        corrected_mask = reviewed_mask & (
            group["reviewed_category_id"].astype(str).str.strip()
            != group["predicted_category_id"].astype(str).str.strip()
        )
        high_mask = group["review_priority"].astype(str).str.lower() == "high"
        medium_mask = group["review_priority"].astype(str).str.lower() == "medium"
        low_mask = group["review_priority"].astype(str).str.lower() == "low"

        reviewed_count = int(reviewed_mask.sum())
        corrected_count = int(corrected_mask.sum())
        total_count = int(len(group))
        high_priority_count = int(high_mask.sum())
        medium_priority_count = int(medium_mask.sum())
        low_priority_count = int(low_mask.sum())
        correction_rate = round(corrected_count / reviewed_count, 3) if reviewed_count > 0 else 0.0
        top_review_triggers = build_top_review_triggers(group["review_trigger"])
        needs_definition_review, definition_review_reason = determine_definition_review(
            total_count=total_count,
            reviewed_count=reviewed_count,
            corrected_count=corrected_count,
            high_priority_count=high_priority_count,
            correction_rate=correction_rate,
            trigger_counts=parse_trigger_counts(top_review_triggers),
        )
        rows.append(
            {
                "question_id": str(question_id),
                "predicted_category_id": str(category_id),
                "predicted_category_name": str(category_name),
                "total_count": total_count,
                "reviewed_count": reviewed_count,
                "corrected_count": corrected_count,
                "pending_count": int(pending_mask.sum()),
                "skipped_count": int(skipped_mask.sum()),
                "high_priority_count": high_priority_count,
                "medium_priority_count": medium_priority_count,
                "low_priority_count": low_priority_count,
                "correction_rate": correction_rate,
                "top_review_triggers": top_review_triggers,
                "needs_definition_review": str(needs_definition_review).lower(),
                "definition_review_reason": definition_review_reason,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate review_log.csv into review_summary.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to review_log.csv")
    parser.add_argument("--output", required=True, type=Path, help="Path to review_summary.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review_df = read_csv(args.input)
    validate_required_columns(review_df, REVIEW_COLUMNS)
    review_errors = run_review_log_validations(review_df)
    if review_errors:
        raise SystemExit("\n".join(review_errors))
    summary_df = build_summary_df(review_df)
    summary_errors = run_review_summary_validations(summary_df)
    if summary_errors:
        raise SystemExit("\n".join(summary_errors))
    write_csv(summary_df, args.output)
    if args.log is not None:
        append_jsonl(
            {
                "event": "review_summary",
                "input": str(args.input),
                "output": str(args.output),
                "row_count": int(len(summary_df)),
                "created_at": utc_now_iso(),
            },
            args.log,
        )


if __name__ == "__main__":
    main()
