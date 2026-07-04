from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from classification_common import FINAL_LABEL_COLUMNS
from common import append_jsonl, read_csv, utc_now_iso, validate_required_columns, write_csv
from duplicate_common import DUPLICATE_OUTPUT_COLUMNS as DUPLICATE_COLUMNS
from review_common import REVIEW_COLUMNS
from validate_duplicate_responses import run_validations as run_duplicate_validations
from validate_final_labels import run_validations as run_final_label_validations
from validate_review_log import run_validations as run_review_log_validations


FALLBACK_CATEGORY_ID = "OTHER"
FALLBACK_CATEGORY_NAME = "その他"
NEGATION_TERMS = ["ない", "ぬ", "ません", "ではない", "じゃない", "なく", "ず", "微妙", "不満", "困る"]
AMBIGUITY_TERMS = ["ambiguous_top_score", "weak_score_gap", "no_keyword_match"]
MULTI_TOPIC_MARKERS = ["。", "、", " and ", "・", "/", "また", "けど", "が", "しかし"]
EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
URL_PATTERN = re.compile(r"https?://[^\s]+")
PHONE_PATTERN = re.compile(r"(?:\+?\d[\d\-\s()]{7,}\d)")
AGGRESSIVE_TERMS = ["死ね", "最悪", "ふざけるな", "バカ", "クソ", "殺す", "詐欺"]


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
    duplicate_df = read_csv(path)
    validate_required_columns(duplicate_df, DUPLICATE_COLUMNS)
    duplicate_errors = run_duplicate_validations(duplicate_df)
    if duplicate_errors:
        raise ValueError("\n".join(duplicate_errors))
    return set(duplicate_df["response_id"].astype(str).tolist())


def collect_review_triggers(
    row: pd.Series,
    confidence_threshold: float,
    duplicate_response_ids: set[str],
) -> list[str]:
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
    parser.add_argument("--duplicates", type=Path, default=None, help="Optional path to duplicate_responses.csv")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    parser.add_argument(
        "--stamp-created-at",
        action="store_true",
        help="Fill reviewed_at with the current timestamp for exported rows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_labels_df = read_csv(args.input)
    validate_required_columns(final_labels_df, FINAL_LABEL_COLUMNS)
    final_label_errors = run_final_label_validations(final_labels_df)
    if final_label_errors:
        raise SystemExit("\n".join(final_label_errors))
    duplicate_response_ids = load_duplicate_response_ids(args.duplicates)
    review_df = build_review_df(
        final_labels_df,
        args.confidence_threshold,
        args.reviewer,
        duplicate_response_ids,
    )
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
