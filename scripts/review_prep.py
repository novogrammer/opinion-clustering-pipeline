from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from classification import FALLBACK_CATEGORY_ID, FALLBACK_CATEGORY_NAME, FINAL_LABEL_COLUMNS, run_final_label_validations
from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns, write_csv
from screening import SCREENED_COLUMNS, run_validations as run_screened_validations


REVIEW_COLUMNS = [
    "response_id",
    "question_id",
    "answer_text",
    "predicted_category_id",
    "predicted_category_name",
    "confidence",
    "reason",
    "needs_human_review",
    "review_priority",
    "review_trigger",
    "review_status",
    "reviewed_category_id",
    "reviewed_category_name",
    "review_comment",
    "reviewer",
    "reviewed_at",
]
ALLOWED_REVIEW_TRIGGER_TOKENS = {
    "fallback_category",
    "other_category",
    "needs_human_review",
    "low_confidence",
    "short_answer",
    "negation",
    "ambiguous_match",
    "multi_topic",
    "pii_detected",
    "aggressive_expression",
    "duplicate_response",
    "typical_match",
}
NEGATION_TERMS = ["ない", "ぬ", "ません", "ではない", "じゃない", "なく", "ず", "微妙", "不満", "困る"]
AMBIGUITY_TERMS = ["ambiguous_top_score", "weak_score_gap", "no_keyword_match"]
MULTI_TOPIC_MARKERS = ["。", "、", " and ", "・", "/", "また", "けど", "が", "しかし"]
EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
URL_PATTERN = re.compile(r"https?://[^\s]+")
PHONE_PATTERN = re.compile(r"(?:\+?\d[\d\-\s()]{7,}\d)")
AGGRESSIVE_TERMS = ["死ね", "最悪", "ふざけるな", "バカ", "クソ", "殺す", "詐欺"]


def validate_status_values(df: pd.DataFrame) -> list[str]:
    allowed = {"pending", "reviewed", "skipped"}
    invalid_rows = [
        str(index + 1)
        for index, value in enumerate(df["review_status"].astype(str).str.lower())
        if value not in allowed
    ]
    if not invalid_rows:
        return []
    return [f"Invalid review_status values at rows: {', '.join(invalid_rows)}"]


def validate_priority_values(df: pd.DataFrame) -> list[str]:
    allowed = {"high", "medium", "low"}
    invalid_rows = [
        str(index + 1)
        for index, value in enumerate(df["review_priority"].astype(str).str.lower())
        if value not in allowed
    ]
    if not invalid_rows:
        return []
    return [f"Invalid review_priority values at rows: {', '.join(invalid_rows)}"]


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


def validate_reviewed_rows(df: pd.DataFrame) -> list[str]:
    reviewed_mask = df["review_status"].astype(str).str.lower() == "reviewed"
    if int(reviewed_mask.sum()) == 0:
        return []
    errors: list[str] = []
    reviewed = df.loc[reviewed_mask]
    for column in ["reviewed_category_id", "reviewed_category_name", "reviewer", "reviewed_at"]:
        blank_mask = reviewed[column].map(lambda value: str(value).strip() == "")
        count = int(blank_mask.sum())
        if count > 0:
            errors.append(f"Blank values found in {column} for reviewed rows: {count}")
    return errors


def validate_skipped_rows(df: pd.DataFrame) -> list[str]:
    skipped_mask = df["review_status"].astype(str).str.lower() == "skipped"
    if int(skipped_mask.sum()) == 0:
        return []
    skipped = df.loc[skipped_mask]
    blank_trigger_mask = skipped["review_trigger"].map(lambda value: str(value).strip() == "")
    if int(blank_trigger_mask.sum()) > 0:
        return ["Blank review_trigger found for skipped rows"]
    return []


def validate_review_trigger_tokens(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for idx, value in enumerate(df["review_trigger"].astype(str), start=1):
        tokens = [token.strip() for token in value.split("|") if token.strip()]
        if not tokens:
            errors.append(f"Blank review_trigger at row {idx}")
            continue
        invalid_tokens = [token for token in tokens if token not in ALLOWED_REVIEW_TRIGGER_TOKENS]
        if invalid_tokens:
            errors.append(f"Invalid review_trigger tokens at row {idx}: {', '.join(invalid_tokens)}")
    return errors


def validate_no_duplicate_response_ids(df: pd.DataFrame) -> list[str]:
    duplicate_mask = df["response_id"].duplicated(keep=False)
    duplicates = df.loc[duplicate_mask, "response_id"].tolist()
    if not duplicates:
        return []
    joined = ", ".join(dict.fromkeys(duplicates))
    return [f"Duplicate response_id values found: {joined}"]


def validate_required_text(df: pd.DataFrame, columns: list[str]) -> list[str]:
    errors: list[str] = []
    for column in columns:
        blank_mask = df[column].map(lambda value: str(value).strip() == "")
        count = int(blank_mask.sum())
        if count > 0:
            errors.append(f"Blank values found in {column}: {count}")
    return errors


def run_review_log_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_no_duplicate_response_ids(df))
    errors.extend(validate_priority_values(df))
    errors.extend(validate_status_values(df))
    errors.extend(validate_boolean_column(df, "needs_human_review"))
    errors.extend(validate_review_trigger_tokens(df))
    errors.extend(
        validate_required_text(
            df,
            [
                "response_id",
                "question_id",
                "answer_text",
                "predicted_category_id",
                "predicted_category_name",
                "review_priority",
                "review_trigger",
                "review_status",
            ],
        )
    )
    errors.extend(validate_reviewed_rows(df))
    errors.extend(validate_skipped_rows(df))
    return errors


def normalize_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def parse_confidence(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_duplicate_response_ids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    screened_df = read_csv(path)
    validate_required_columns(screened_df, SCREENED_COLUMNS)
    screened_errors = run_screened_validations(screened_df)
    if screened_errors:
        raise ValueError("\n".join(screened_errors))
    duplicate_df = screened_df[screened_df["duplicate_group_id"].astype(str).str.strip() != ""].copy()
    if len(duplicate_df) == 0:
        return set()
    return set(duplicate_df["response_id"].astype(str).tolist())


def collect_review_triggers(row: pd.Series, confidence_threshold: float, duplicate_response_ids: set[str]) -> list[str]:
    triggers: list[str] = []
    response_id = str(row["response_id"]).strip()
    answer_text = str(row["answer_text"]).strip()
    reason = str(row["reason"]).strip()
    confidence_value = parse_confidence(row["confidence"])

    if str(row["predicted_category_id"]).strip() == FALLBACK_CATEGORY_ID:
        triggers.append("fallback_category")
    if str(row["predicted_category_name"]).strip() == FALLBACK_CATEGORY_NAME:
        triggers.append("other_category")
    if normalize_bool(row["needs_human_review"]):
        triggers.append("needs_human_review")
    if confidence_value < confidence_threshold:
        triggers.append("low_confidence")
    if len(answer_text) <= 3:
        triggers.append("short_answer")
    if any(term in answer_text for term in NEGATION_TERMS):
        triggers.append("negation")
    if any(term in reason for term in AMBIGUITY_TERMS):
        triggers.append("ambiguous_match")
    if sum(answer_text.count(marker) for marker in MULTI_TOPIC_MARKERS) >= 2 or answer_text.count("\n") >= 1:
        triggers.append("multi_topic")
    if EMAIL_PATTERN.search(answer_text) or URL_PATTERN.search(answer_text) or PHONE_PATTERN.search(answer_text):
        triggers.append("pii_detected")
    if any(term in answer_text for term in AGGRESSIVE_TERMS):
        triggers.append("aggressive_expression")
    if response_id in duplicate_response_ids:
        triggers.append("duplicate_response")
    return triggers


def choose_priority(triggers: list[str]) -> str:
    high_priority_triggers = {
        "fallback_category",
        "other_category",
        "needs_human_review",
        "low_confidence",
        "short_answer",
        "negation",
        "ambiguous_match",
        "pii_detected",
        "aggressive_expression",
        "duplicate_response",
    }
    if any(trigger in high_priority_triggers for trigger in triggers):
        return "high"
    if "multi_topic" in triggers:
        return "medium"
    return "low"


def build_review_trigger(triggers: list[str]) -> str:
    return "|".join(triggers) if triggers else "typical_match"


def choose_status(triggers: list[str]) -> str:
    return "pending" if triggers else "skipped"


def choose_priority_and_trigger(
    row: pd.Series,
    confidence_threshold: float,
    duplicate_response_ids: set[str],
) -> tuple[str, str, str]:
    triggers = collect_review_triggers(row, confidence_threshold, duplicate_response_ids)
    return choose_priority(triggers), build_review_trigger(triggers), choose_status(triggers)


def build_review_df(
    final_labels_df: pd.DataFrame,
    confidence_threshold: float,
    reviewer: str,
    duplicate_response_ids: set[str],
) -> pd.DataFrame:
    review_df = final_labels_df.copy()
    decisions = review_df.apply(
        lambda row: choose_priority_and_trigger(row, confidence_threshold, duplicate_response_ids),
        axis=1,
        result_type="expand",
    )
    decisions.columns = ["review_priority", "review_trigger", "review_status"]
    review_df[["review_priority", "review_trigger", "review_status"]] = decisions
    review_df["reviewed_category_id"] = ""
    review_df["reviewed_category_name"] = ""
    review_df["review_comment"] = ""
    review_df["reviewer"] = reviewer
    review_df["reviewed_at"] = ""
    return review_df[REVIEW_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a review log from final_labels.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to final_labels.csv")
    parser.add_argument("--output", required=True, type=Path, help="Path to review_log.csv")
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--reviewer", default="", help="Default reviewer name")
    parser.add_argument("--screened", type=Path, default=None, help="Optional path to screened_responses.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    parser.add_argument("--stamp-created-at", action="store_true", help="Fill reviewed_at with the current timestamp for exported rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_labels_df = read_csv(args.input)
    validate_required_columns(final_labels_df, FINAL_LABEL_COLUMNS)
    final_label_errors = run_final_label_validations(final_labels_df)
    if final_label_errors:
        raise SystemExit("\n".join(final_label_errors))
    duplicate_response_ids = load_duplicate_response_ids(args.screened)
    review_df = build_review_df(final_labels_df, args.confidence_threshold, args.reviewer, duplicate_response_ids)
    if args.stamp_created_at:
        review_df["reviewed_at"] = utc_now_iso()
    review_errors = run_review_log_validations(review_df)
    if review_errors:
        raise SystemExit("\n".join(review_errors))
    write_csv(review_df, args.output)
    if args.log is not None:
        append_jsonl(
            {
                "event": "review_prep",
                "input": str(args.input),
                "output": str(args.output),
                "row_count": int(len(review_df)),
                "duplicate_response_count": int(len(duplicate_response_ids)),
                "created_at": utc_now_iso(),
            },
            args.log,
        )


if __name__ == "__main__":
    main()
