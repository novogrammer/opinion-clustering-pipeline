from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from classification import load_override_rules, matches_override_rule
from classification_common import OVERRIDE_RULE_COLUMNS
from common import REQUIRED_RESPONSE_COLUMNS, append_jsonl, read_csv, utc_now_iso, validate_required_columns, write_csv
from override_common import OVERRIDE_RULE_HIT_COLUMNS as OUTPUT_COLUMNS
from validate_override_rule_hits import run_validations as run_override_rule_hit_validations
from validate_override_rules import run_validations as run_override_rule_validations
from validate_screened_responses import run_validations as run_screened_validations


def build_hits_df(responses_df: pd.DataFrame, question_id: str, override_rules: list[dict[str, object]]) -> pd.DataFrame:
    filtered = responses_df[
        (responses_df["question_id"].astype(str) == question_id)
        & (responses_df["is_target"].astype(str).str.lower() == "true")
    ].copy()

    rows: list[dict[str, object]] = []
    for _, response_row in filtered.iterrows():
        answer_text = str(response_row["answer_text"])
        for rule in override_rules:
            if not matches_override_rule(answer_text, rule):
                continue
            rows.append(
                {
                    "response_id": str(response_row["response_id"]).strip(),
                    "question_id": str(response_row["question_id"]).strip(),
                    "answer_text": answer_text,
                    "rule_id": str(rule["rule_id"]).strip(),
                    "match_type": str(rule["match_type"]).strip(),
                    "pattern": str(rule["pattern"]).strip(),
                    "override_category_id": str(rule["override_category_id"]).strip(),
                    "override_category_name": str(rule["override_category_name"]).strip(),
                    "needs_human_review": str(bool(rule["needs_human_review"])).lower(),
                    "priority": str(rule["priority"]),
                    "note": str(rule["note"]).strip(),
                }
            )
            break

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List which responses are matched by manual_override_rules.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to screened_responses.csv")
    parser.add_argument("--question-id", required=True, help="Target question_id")
    parser.add_argument("--override-rules", required=True, type=Path, help="Path to manual_override_rules.csv")
    parser.add_argument("--output", required=True, type=Path, help="Path to override_rule_hits.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    responses_df = read_csv(args.input)
    validate_required_columns(responses_df, REQUIRED_RESPONSE_COLUMNS + ["is_target", "screening_reason"])
    screened_errors = run_screened_validations(responses_df)
    if screened_errors:
        raise SystemExit("\n".join(screened_errors))

    override_rules_df = read_csv(args.override_rules)
    validate_required_columns(override_rules_df, OVERRIDE_RULE_COLUMNS)
    override_rule_errors = run_override_rule_validations(override_rules_df)
    if override_rule_errors:
        raise SystemExit("\n".join(override_rule_errors))
    override_rules = load_override_rules(args.override_rules, args.question_id)
    hits_df = build_hits_df(responses_df, args.question_id, override_rules)
    hit_errors = run_override_rule_hit_validations(hits_df)
    if hit_errors:
        raise SystemExit("\n".join(hit_errors))
    write_csv(hits_df, args.output)
    if args.log is not None:
        append_jsonl(
            {
                "event": "override_rule_hits",
                "input": str(args.input),
                "override_rules": str(args.override_rules),
                "output": str(args.output),
                "question_id": args.question_id,
                "rule_count": int(len(override_rules)),
                "hit_count": int(len(hits_df)),
                "created_at": utc_now_iso(),
            },
            args.log,
        )


if __name__ == "__main__":
    main()
