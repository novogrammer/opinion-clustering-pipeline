from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from classification_common import OVERRIDE_RULE_COLUMNS
from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns, write_csv
from override_common import OVERRIDE_CANDIDATE_COLUMNS
from validate_override_candidates import run_validations as run_override_candidate_validations
from validate_override_rules import run_validations as run_override_rule_validations


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def build_rules_df(candidates_df: pd.DataFrame) -> pd.DataFrame:
    approved = candidates_df[candidates_df["approved"].map(parse_bool)].copy()
    if len(approved) == 0:
        return pd.DataFrame(columns=OVERRIDE_RULE_COLUMNS)
    return approved[OVERRIDE_RULE_COLUMNS].copy()


def merge_rules(existing_df: pd.DataFrame, new_rules_df: pd.DataFrame) -> pd.DataFrame:
    if len(existing_df) == 0:
        return new_rules_df[OVERRIDE_RULE_COLUMNS].copy()
    if len(new_rules_df) == 0:
        return existing_df[OVERRIDE_RULE_COLUMNS].copy()

    existing = existing_df.copy()
    replacements = set(new_rules_df["rule_id"].astype(str).tolist())
    existing = existing[~existing["rule_id"].astype(str).isin(replacements)].copy()
    merged = pd.concat([existing[OVERRIDE_RULE_COLUMNS], new_rules_df[OVERRIDE_RULE_COLUMNS]], ignore_index=True)

    def parse_priority(value: object) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 9999

    merged["_priority_int"] = merged["priority"].map(parse_priority)
    merged = merged.sort_values(by=["_priority_int", "rule_id"], ascending=[True, True], kind="stable")
    merged = merged.drop(columns=["_priority_int"]).reset_index(drop=True)
    return merged[OVERRIDE_RULE_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote approved override candidates into manual_override_rules.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to manual_override_candidates.csv")
    parser.add_argument("--output", required=True, type=Path, help="Path to manual_override_rules.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates_df = read_csv(args.input)
    validate_required_columns(candidates_df, OVERRIDE_CANDIDATE_COLUMNS)
    candidate_errors = run_override_candidate_validations(candidates_df)
    if candidate_errors:
        raise SystemExit("\n".join(candidate_errors))
    new_rules_df = build_rules_df(candidates_df)

    if args.output.exists():
        existing_df = read_csv(args.output)
        validate_required_columns(existing_df, OVERRIDE_RULE_COLUMNS)
        existing_rule_errors = run_override_rule_validations(existing_df)
        if existing_rule_errors:
            raise SystemExit("\n".join(existing_rule_errors))
    else:
        existing_df = pd.DataFrame(columns=OVERRIDE_RULE_COLUMNS)

    merged_df = merge_rules(existing_df, new_rules_df)
    merged_rule_errors = run_override_rule_validations(merged_df)
    if merged_rule_errors:
        raise SystemExit("\n".join(merged_rule_errors))
    write_csv(merged_df, args.output)
    if args.log is not None:
        append_jsonl(
            {
                "event": "promote_override_candidates",
                "input": str(args.input),
                "output": str(args.output),
                "approved_row_count": int(len(new_rules_df)),
                "final_row_count": int(len(merged_df)),
                "created_at": utc_now_iso(),
            },
            args.log,
        )


if __name__ == "__main__":
    main()
