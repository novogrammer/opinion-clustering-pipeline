from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns, write_csv
from review_common import REVIEW_COLUMNS, REVIEW_SAMPLE_COLUMNS as OUTPUT_COLUMNS
from validate_review_log import run_validations as run_review_log_validations
from validate_review_samples import run_validations as run_review_sample_validations


def has_trigger(value: str, trigger: str) -> bool:
    tokens = [token.strip() for token in str(value).split("|") if token.strip()]
    return trigger in tokens


def select_samples(
    review_df: pd.DataFrame,
    *,
    medium_per_category: int,
    include_low: bool,
    priority_trigger: str | None,
) -> pd.DataFrame:
    working = review_df.copy()
    working["review_priority"] = working["review_priority"].astype(str).str.lower()
    working["review_status"] = working["review_status"].astype(str).str.lower()
    working["sample_reason"] = ""
    working["sample_bucket"] = ""

    selected_frames: list[pd.DataFrame] = []
    selected_ids: set[str] = set()

    if priority_trigger:
        trigger_df = working[working["review_trigger"].map(lambda value: has_trigger(value, priority_trigger))].copy()
        trigger_df["sample_reason"] = f"trigger:{priority_trigger}"
        trigger_df["sample_bucket"] = "trigger"
        selected_frames.append(trigger_df)
        selected_ids.update(trigger_df["response_id"].astype(str).tolist())

    high_df = working[
        (working["review_priority"] == "high")
        & (~working["response_id"].astype(str).isin(selected_ids))
    ].copy()
    high_df["sample_reason"] = "high_priority"
    high_df["sample_bucket"] = "high"
    selected_frames.append(high_df)
    selected_ids.update(high_df["response_id"].astype(str).tolist())

    medium_source = working[
        (working["review_priority"] == "medium")
        & (~working["response_id"].astype(str).isin(selected_ids))
    ].copy()
    medium_parts: list[pd.DataFrame] = []
    for _, group in medium_source.groupby("predicted_category_id", sort=True):
        sampled = group.head(medium_per_category).copy()
        sampled["sample_reason"] = f"medium_priority_top_{medium_per_category}"
        sampled["sample_bucket"] = "medium"
        medium_parts.append(sampled)
    if medium_parts:
        medium_df = pd.concat(medium_parts, ignore_index=True)
        selected_frames.append(medium_df)
        selected_ids.update(medium_df["response_id"].astype(str).tolist())

    if include_low:
        low_source = working[
            (working["review_priority"] == "low")
            & (~working["response_id"].astype(str).isin(selected_ids))
        ].copy()
        low_parts: list[pd.DataFrame] = []
        for _, group in low_source.groupby("predicted_category_id", sort=True):
            sampled = group.head(1).copy()
            sampled["sample_reason"] = "low_priority_reference"
            sampled["sample_bucket"] = "low"
            low_parts.append(sampled)
        if low_parts:
            low_df = pd.concat(low_parts, ignore_index=True)
            selected_frames.append(low_df)

    if not selected_frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    result = pd.concat(selected_frames, ignore_index=True)
    result = result.drop_duplicates(subset=["response_id"], keep="first")
    bucket_order = {"trigger": 0, "high": 1, "medium": 2, "low": 3}
    result["_bucket_order"] = result["sample_bucket"].map(bucket_order).fillna(9)
    result = result.sort_values(
        by=["_bucket_order", "predicted_category_id", "response_id"],
        ascending=[True, True, True],
        kind="stable",
    ).drop(columns=["_bucket_order"])
    return result[OUTPUT_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract review samples from review_log.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to review_log.csv")
    parser.add_argument("--output", required=True, type=Path, help="Path to review_samples.csv")
    parser.add_argument("--medium-per-category", type=int, default=3)
    parser.add_argument("--include-low", action="store_true")
    parser.add_argument("--priority-trigger", default=None, help="Optional review_trigger token to always include")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review_df = read_csv(args.input)
    validate_required_columns(review_df, REVIEW_COLUMNS)
    review_errors = run_review_log_validations(review_df)
    if review_errors:
        raise SystemExit("\n".join(review_errors))
    samples_df = select_samples(
        review_df,
        medium_per_category=args.medium_per_category,
        include_low=args.include_low,
        priority_trigger=args.priority_trigger,
    )
    sample_errors = run_review_sample_validations(samples_df)
    if sample_errors:
        raise SystemExit("\n".join(sample_errors))
    write_csv(samples_df, args.output)
    if args.log is not None:
        append_jsonl(
            {
                "event": "review_samples",
                "input": str(args.input),
                "output": str(args.output),
                "row_count": int(len(samples_df)),
                "medium_per_category": args.medium_per_category,
                "include_low": args.include_low,
                "priority_trigger": args.priority_trigger,
                "created_at": utc_now_iso(),
            },
            args.log,
        )


if __name__ == "__main__":
    main()
