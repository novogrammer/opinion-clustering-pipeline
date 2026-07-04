from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from classification_common import OVERRIDE_RULE_COLUMNS
from classification_keywords import normalize_text
from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns, write_csv
from override_common import OVERRIDE_CANDIDATE_COLUMNS as OUTPUT_COLUMNS
from review_corrections import OUTPUT_COLUMNS as REVIEW_CORRECTION_COLUMNS
from validate_override_candidates import run_validations as run_override_candidate_validations
from validate_review_corrections import run_validations as run_review_correction_validations


def build_rule_id(index: int) -> str:
    return f"CAND{index:03d}"


def build_candidates_df(corrections_df: pd.DataFrame) -> pd.DataFrame:
    if len(corrections_df) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    working = corrections_df.copy()
    working["normalized_answer_text"] = working["answer_text"].map(lambda value: normalize_text(str(value)))
    working = working.sort_values(
        by=["question_id", "reviewed_category_id", "normalized_answer_text", "response_id"],
        ascending=[True, True, True, True],
        kind="stable",
    )
    working = working.drop_duplicates(
        subset=["question_id", "normalized_answer_text", "reviewed_category_id"],
        keep="first",
    ).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for idx, row in working.iterrows():
        note_parts = [
            f"source_predicted={str(row['predicted_category_id']).strip()}:{str(row['predicted_category_name']).strip()}",
            f"source_reviewed={str(row['reviewed_category_id']).strip()}:{str(row['reviewed_category_name']).strip()}",
        ]
        review_comment = str(row["review_comment"]).strip()
        if review_comment:
            note_parts.append(f"comment={review_comment}")

        rows.append(
            {
                "rule_id": build_rule_id(idx + 1),
                "question_id": str(row["question_id"]).strip(),
                "match_type": "exact",
                "pattern": str(row["normalized_answer_text"]).strip(),
                "override_category_id": str(row["reviewed_category_id"]).strip(),
                "override_category_name": str(row["reviewed_category_name"]).strip(),
                "needs_human_review": "true",
                "priority": str(100 + idx),
                "note": " | ".join(note_parts),
                "approved": "false",
                "approved_note": "",
                "source_response_id": str(row["response_id"]).strip(),
                "source_predicted_category_id": str(row["predicted_category_id"]).strip(),
                "source_predicted_category_name": str(row["predicted_category_name"]).strip(),
                "source_review_comment": review_comment,
                "source_correction_type": str(row["correction_type"]).strip(),
            }
        )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manual override rule candidates from review_corrections.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to review_corrections.csv")
    parser.add_argument("--output", required=True, type=Path, help="Path to manual_override_candidates.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corrections_df = read_csv(args.input)
    validate_required_columns(corrections_df, REVIEW_CORRECTION_COLUMNS)
    correction_errors = run_review_correction_validations(corrections_df)
    if correction_errors:
        raise SystemExit("\n".join(correction_errors))
    candidates_df = build_candidates_df(corrections_df)
    candidate_errors = run_override_candidate_validations(candidates_df)
    if candidate_errors:
        raise SystemExit("\n".join(candidate_errors))
    write_csv(candidates_df, args.output)
    if args.log is not None:
        append_jsonl(
            {
                "event": "review_override_candidates",
                "input": str(args.input),
                "output": str(args.output),
                "row_count": int(len(candidates_df)),
                "created_at": utc_now_iso(),
            },
            args.log,
        )


if __name__ == "__main__":
    main()
