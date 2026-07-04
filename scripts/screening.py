from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

from common import (
    REQUIRED_RESPONSE_COLUMNS,
    append_jsonl,
    read_csv,
    utc_now_iso,
    validate_required_columns,
    write_csv,
)
from normalize import run_validations as run_processed_validations


NON_RESPONSE_VALUES = {
    "なし",
    "特になし",
    "とくになし",
    "ない",
    "わからない",
    "なしです",
    "n/a",
    "na",
    "no answer",
}
NON_RESPONSE_CASEFOLDED = {item.casefold() for item in NON_RESPONSE_VALUES}
SYMBOL_ONLY_PATTERN = re.compile(r"^[\W_]+$", re.UNICODE)
SCREENED_COLUMNS = REQUIRED_RESPONSE_COLUMNS + [
    "is_target",
    "screening_reason",
]
ALLOWED_REASONS = {"blank", "non_response", "symbol_only", "target"}


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return normalized.strip()


def classify_answer(answer_text: str) -> tuple[bool, str]:
    normalized = normalize_text(answer_text)
    if normalized == "":
        return False, "blank"
    if normalized.casefold() in NON_RESPONSE_CASEFOLDED:
        return False, "non_response"
    if SYMBOL_ONLY_PATTERN.fullmatch(normalized):
        return False, "symbol_only"
    return True, "target"


def validate_no_duplicate_response_ids(df: pd.DataFrame) -> list[str]:
    duplicate_mask = df["response_id"].duplicated(keep=False)
    duplicates = df.loc[duplicate_mask, "response_id"].tolist()
    if not duplicates:
        return []
    joined = ", ".join(dict.fromkeys(duplicates))
    return [f"Duplicate response_id values found: {joined}"]


def validate_no_blank_core_values(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for column in ["response_id", "question_id", "question_text"]:
        blank_mask = df[column].map(lambda value: str(value).strip() == "")
        count = int(blank_mask.sum())
        if count > 0:
            errors.append(f"Blank values found in {column}: {count}")
    return errors


def validate_boolean_column(df: pd.DataFrame, column: str) -> list[str]:
    allowed = {"true", "false"}
    invalid_rows = [
        str(index + 1)
        for index, value in enumerate(df[column].astype(str).str.lower())
        if value not in allowed
    ]
    if not invalid_rows:
        return []
    return [f"Invalid boolean values in {column} at rows: {', '.join(invalid_rows)}"]


def validate_reason_values(df: pd.DataFrame) -> list[str]:
    invalid_rows = [
        f"{index + 1}:{value}"
        for index, value in enumerate(df["screening_reason"].astype(str))
        if value not in ALLOWED_REASONS
    ]
    if not invalid_rows:
        return []
    return [f"Invalid screening_reason values: {', '.join(invalid_rows)}"]


def validate_reason_boolean_consistency(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for idx, row in df.iterrows():
        is_target = str(row["is_target"]).lower() == "true"
        reason = str(row["screening_reason"])
        if is_target and reason != "target":
            errors.append(f"Row {idx + 1}: is_target=true requires screening_reason=target")
        if not is_target and reason == "target":
            errors.append(f"Row {idx + 1}: is_target=false cannot have screening_reason=target")
    return errors


def validate_reason_content_consistency(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for idx, row in df.iterrows():
        answer_text = str(row["answer_text"])
        reason = str(row["screening_reason"])
        normalized = normalize_text(answer_text)

        if reason == "blank" and normalized != "":
            errors.append(f"Row {idx + 1}: screening_reason=blank requires blank answer_text")
        elif reason == "non_response" and normalized.casefold() not in NON_RESPONSE_CASEFOLDED:
            errors.append(f"Row {idx + 1}: screening_reason=non_response requires dictionary match")
        elif reason == "symbol_only" and SYMBOL_ONLY_PATTERN.fullmatch(normalized) is None:
            errors.append(f"Row {idx + 1}: screening_reason=symbol_only requires symbol-only answer_text")
        elif reason == "target":
            if normalized == "":
                errors.append(f"Row {idx + 1}: screening_reason=target cannot have blank answer_text")
            elif normalized.casefold() in NON_RESPONSE_CASEFOLDED:
                errors.append(f"Row {idx + 1}: screening_reason=target cannot match non_response dictionary")
            elif SYMBOL_ONLY_PATTERN.fullmatch(normalized) is not None:
                errors.append(f"Row {idx + 1}: screening_reason=target cannot be symbol-only")
    return errors


def run_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_no_duplicate_response_ids(df))
    errors.extend(validate_no_blank_core_values(df))
    errors.extend(validate_boolean_column(df, "is_target"))
    errors.extend(validate_reason_values(df))
    errors.extend(validate_reason_boolean_consistency(df))
    errors.extend(validate_reason_content_consistency(df))
    return errors


def build_screened_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    results = working["answer_text"].map(classify_answer)
    working["is_target"] = results.map(lambda item: item[0])
    working["screening_reason"] = results.map(lambda item: item[1])
    return working[SCREENED_COLUMNS]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply initial screening rules to normalized responses.")
    parser.add_argument("--input", required=True, type=Path, help="Path to responses_normalized.csv")
    parser.add_argument("--output", type=Path, default=None, help="Path to screened_responses.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = read_csv(args.input)

    if args.output is None:
        raise SystemExit("--output is required")

    validate_required_columns(df, REQUIRED_RESPONSE_COLUMNS)
    processed_errors = run_processed_validations(df)
    if processed_errors:
        raise SystemExit("\n".join(processed_errors))
    screened = build_screened_dataframe(df)
    screened_errors = run_validations(screened)
    if screened_errors:
        raise SystemExit("\n".join(screened_errors))
    write_csv(screened, args.output)
    if args.log is not None:
        reason_counts = screened["screening_reason"].value_counts(dropna=False).to_dict()
        append_jsonl(
            {
                "event": "screening",
                "input": str(args.input),
                "output": str(args.output),
                "row_count": int(len(screened)),
                "target_count": int((screened["is_target"] == True).sum()),
                "reason_counts": {str(key): int(value) for key, value in reason_counts.items()},
                "created_at": utc_now_iso(),
            },
            args.log,
        )


if __name__ == "__main__":
    main()
