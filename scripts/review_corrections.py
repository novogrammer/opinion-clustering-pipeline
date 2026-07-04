from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns, write_csv
from review_common import REVIEW_COLUMNS, REVIEW_CORRECTION_COLUMNS as OUTPUT_COLUMNS
from validate_review_corrections import run_validations as run_review_correction_validations
from validate_review_log import run_validations as run_review_log_validations


def build_corrections_df(review_df: pd.DataFrame) -> pd.DataFrame:
    reviewed = review_df[review_df["review_status"].astype(str).str.lower() == "reviewed"].copy()
    corrected = reviewed[
        reviewed["reviewed_category_id"].astype(str).str.strip()
        != reviewed["predicted_category_id"].astype(str).str.strip()
    ].copy()
    if len(corrected) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    corrected["correction_type"] = corrected.apply(
        lambda row: (
            "fallback_to_category"
            if str(row["predicted_category_id"]).strip() == "OTHER"
            else "category_change"
        ),
        axis=1,
    )
    corrected = corrected.sort_values(
        by=["reviewed_at", "reviewer", "response_id"],
        ascending=[True, True, True],
        kind="stable",
    )
    return corrected[OUTPUT_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract corrected examples from review_log.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to review_log.csv")
    parser.add_argument("--output", required=True, type=Path, help="Path to review_corrections.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review_df = read_csv(args.input)
    validate_required_columns(review_df, REVIEW_COLUMNS)
    review_errors = run_review_log_validations(review_df)
    if review_errors:
        raise SystemExit("\n".join(review_errors))
    corrections_df = build_corrections_df(review_df)
    correction_errors = run_review_correction_validations(corrections_df)
    if correction_errors:
        raise SystemExit("\n".join(correction_errors))
    write_csv(corrections_df, args.output)
    if args.log is not None:
        append_jsonl(
            {
                "event": "review_corrections",
                "input": str(args.input),
                "output": str(args.output),
                "row_count": int(len(corrections_df)),
                "created_at": utc_now_iso(),
            },
            args.log,
        )


if __name__ == "__main__":
    main()
