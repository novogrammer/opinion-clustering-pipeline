from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from common import REQUIRED_RESPONSE_COLUMNS, append_jsonl, read_csv, utc_now_iso, validate_required_columns, write_csv
from duplicate_common import DUPLICATE_OUTPUT_COLUMNS, normalize_answer_text
from screening_common import SCREENED_COLUMNS
from validate_duplicate_responses import run_validations as run_duplicate_validations
from validate_screened_responses import run_validations as run_screened_validations

def build_duplicate_df(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    validate_required_columns(working, SCREENED_COLUMNS)
    target_rows = working[working["is_target"].astype(str).str.lower() == "true"].copy()
    target_rows["normalized_answer_text"] = target_rows["answer_text"].map(normalize_answer_text)
    target_rows = target_rows[target_rows["normalized_answer_text"] != ""].copy()

    grouped = target_rows.groupby(["question_id", "normalized_answer_text"], sort=True, dropna=False)
    rows: list[dict[str, object]] = []

    for (question_id, normalized_answer_text), group in grouped:
        if len(group) <= 1:
            continue
        sorted_group = group.sort_values(by=["response_id"], kind="stable").copy()
        canonical_response_id = str(sorted_group["response_id"].iloc[0])
        duplicate_group_id = hashlib.sha1(
            f"{question_id}\n{normalized_answer_text}".encode("utf-8")
        ).hexdigest()[:12]
        duplicate_count = int(len(sorted_group))
        for _, row in sorted_group.iterrows():
            response_id = str(row["response_id"])
            rows.append(
                {
                    "duplicate_group_id": duplicate_group_id,
                    "question_id": str(question_id),
                    "response_id": response_id,
                    "canonical_response_id": canonical_response_id,
                    "duplicate_count": duplicate_count,
                    "is_canonical": str(response_id == canonical_response_id).lower(),
                    "answer_text": str(row["answer_text"]),
                }
            )

    return pd.DataFrame(rows, columns=DUPLICATE_OUTPUT_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract duplicate free-answer responses for audit")
    parser.add_argument("--input", required=True, type=Path, help="Path to screened_responses.csv")
    parser.add_argument("--output", required=True, type=Path, help="Path to duplicate_responses.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_csv(args.input)
    validate_required_columns(df, REQUIRED_RESPONSE_COLUMNS + ["is_target", "screening_reason"])
    screened_errors = run_screened_validations(df)
    if screened_errors:
        raise SystemExit("\n".join(screened_errors))
    duplicate_df = build_duplicate_df(df)
    duplicate_errors = run_duplicate_validations(duplicate_df)
    if duplicate_errors:
        raise SystemExit("\n".join(duplicate_errors))
    write_csv(duplicate_df, args.output)
    if args.log is not None:
        duplicate_target_count = int(duplicate_df["response_id"].nunique()) if len(duplicate_df) > 0 else 0
        append_jsonl(
            {
                "event": "duplicate_responses",
                "input": str(args.input),
                "output": str(args.output),
                "row_count": int(len(duplicate_df)),
                "group_count": int(duplicate_df["duplicate_group_id"].nunique()) if len(duplicate_df) > 0 else 0,
                "duplicate_target_count": duplicate_target_count,
                "created_at": utc_now_iso(),
            },
            args.log,
        )


if __name__ == "__main__":
    main()
