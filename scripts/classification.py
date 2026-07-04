from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

from common import REQUIRED_RESPONSE_COLUMNS, append_jsonl, read_csv, utc_now_iso, validate_required_columns, write_csv
from screening import SCREENED_COLUMNS, run_validations as run_screened_validations


CATEGORY_MASTER_COLUMNS = [
    "category_id",
    "category_name",
    "category_definition",
    "include_criteria",
    "exclude_criteria",
    "example_positive",
    "example_negative",
]
FINAL_LABEL_COLUMNS = [
    "response_id",
    "question_id",
    "answer_text",
    "predicted_category_id",
    "predicted_category_name",
    "confidence",
    "reason",
    "needs_human_review",
]
FALLBACK_CATEGORY_ID = "OTHER"
FALLBACK_CATEGORY_NAME = "その他"
NEGATION_TERMS = ["ない", "ぬ", "ません", "ではない", "じゃない", "なく", "ず", "微妙", "不満", "困る"]
MULTI_TOPIC_MARKERS = ["。", "、", " and ", "・", "/", "また", "けど", "が", "しかし"]


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return normalized.strip()


def split_keywords(value: str) -> list[str]:
    normalized = normalize_text(value)
    if not normalized:
        return []
    parts = re.split(r"[\n,、。/・;；|]+", normalized)
    return [part.strip() for part in parts if part.strip()]


def collect_category_keywords(row: pd.Series) -> list[str]:
    keywords: list[str] = []
    for column in ["category_name", "category_definition", "include_criteria", "example_positive"]:
        keywords.extend(split_keywords(str(row[column])))
    seen: set[str] = set()
    unique_keywords: list[str] = []
    for keyword in keywords:
        folded = keyword.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        unique_keywords.append(keyword)
    return unique_keywords


def collect_exclude_keywords(row: pd.Series) -> list[str]:
    keywords: list[str] = []
    for column in ["exclude_criteria", "example_negative"]:
        keywords.extend(split_keywords(str(row[column])))
    return keywords


def build_categories_df(category_master_df: pd.DataFrame) -> list[dict[str, object]]:
    categories: list[dict[str, object]] = []
    for _, row in category_master_df.iterrows():
        categories.append(
            {
                "category_id": str(row["category_id"]),
                "category_name": str(row["category_name"]),
                "keywords": collect_category_keywords(row),
                "exclude_keywords": collect_exclude_keywords(row),
            }
        )
    return categories


def validate_no_duplicate_category_ids(df: pd.DataFrame) -> list[str]:
    duplicate_mask = df["category_id"].duplicated(keep=False)
    duplicates = df.loc[duplicate_mask, "category_id"].tolist()
    if not duplicates:
        return []
    joined = ", ".join(dict.fromkeys(duplicates))
    return [f"Duplicate category_id values found: {joined}"]


def validate_no_blank_category_values(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for column in ["category_id", "category_name", "category_definition"]:
        blank_mask = df[column].map(lambda value: str(value).strip() == "")
        count = int(blank_mask.sum())
        if count > 0:
            errors.append(f"Blank values found in {column}: {count}")
    return errors


def validate_keyword_coverage(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    categories = build_categories_df(df)
    for category in categories:
        if len(category["keywords"]) == 0:
            errors.append(f"Category has no usable keywords: {category['category_id']}")
    return errors


def run_category_master_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_no_duplicate_category_ids(df))
    errors.extend(validate_no_blank_category_values(df))
    errors.extend(validate_keyword_coverage(df))
    return errors


def validate_no_duplicate_response_ids(df: pd.DataFrame) -> list[str]:
    duplicate_mask = df["response_id"].duplicated(keep=False)
    duplicates = df.loc[duplicate_mask, "response_id"].tolist()
    if not duplicates:
        return []
    joined = ", ".join(dict.fromkeys(duplicates))
    return [f"Duplicate response_id values found: {joined}"]


def validate_confidence_range(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for idx, value in enumerate(df["confidence"], start=1):
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            errors.append(f"Invalid confidence at row {idx}: {value}")
            continue
        if confidence < 0 or confidence > 1:
            errors.append(f"Out-of-range confidence at row {idx}: {value}")
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


def validate_required_text(df: pd.DataFrame, columns: list[str]) -> list[str]:
    errors: list[str] = []
    for column in columns:
        blank_mask = df[column].map(lambda value: str(value).strip() == "")
        count = int(blank_mask.sum())
        if count > 0:
            errors.append(f"Blank values found in {column}: {count}")
    return errors


def run_final_label_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_no_duplicate_response_ids(df))
    errors.extend(validate_confidence_range(df))
    errors.extend(validate_boolean_column(df, "needs_human_review"))
    errors.extend(
        validate_required_text(
            df,
            ["response_id", "question_id", "answer_text", "predicted_category_id", "predicted_category_name", "reason"],
        )
    )
    return errors


def score_category(answer_text: str, keywords: list[str], exclude_keywords: list[str]) -> tuple[int, list[str], list[str]]:
    normalized_answer = normalize_text(answer_text).casefold()
    matched = [keyword for keyword in keywords if keyword and keyword.casefold() in normalized_answer]
    excluded = [keyword for keyword in exclude_keywords if keyword and keyword.casefold() in normalized_answer]
    score = max(0, len(matched) - len(excluded))
    return score, matched, excluded


def has_negation(answer_text: str) -> bool:
    return any(term in normalize_text(answer_text) for term in NEGATION_TERMS)


def is_multi_topic_answer(answer_text: str) -> bool:
    normalized = normalize_text(answer_text)
    marker_hits = sum(normalized.count(marker) for marker in MULTI_TOPIC_MARKERS)
    return marker_hits >= 2 or "\n" in normalized


def format_candidate_summary(candidate: dict[str, object] | None) -> str:
    if candidate is None:
        return "none"
    matched = ",".join(candidate["matched"]) if candidate["matched"] else "none"
    excluded = ",".join(candidate["excluded"]) if candidate["excluded"] else "none"
    return (
        f"{candidate['category_id']}:{candidate['category_name']}"
        f"(score={candidate['score']};matched={matched};excluded={excluded})"
    )


def classify_row(
    answer_text: str,
    categories: list[dict[str, object]],
    confidence_threshold: float,
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for category in categories:
        score, matched, excluded = score_category(
            answer_text=answer_text,
            keywords=category["keywords"],
            exclude_keywords=category["exclude_keywords"],
        )
        candidates.append(
            {
                "category_id": category["category_id"],
                "category_name": category["category_name"],
                "score": score,
                "matched": matched,
                "excluded": excluded,
            }
        )

    ranked = sorted(candidates, key=lambda item: (item["score"], len(item["matched"])), reverse=True)
    top = ranked[0] if ranked else None
    second = ranked[1] if len(ranked) > 1 else None

    normalized_answer = normalize_text(answer_text)
    is_short_answer = len(normalized_answer) <= 3
    contains_negation = has_negation(answer_text)
    multi_topic = is_multi_topic_answer(answer_text)

    if top is None or top["score"] == 0:
        review_flags = ["no_keyword_match"]
        if is_short_answer:
            review_flags.append("short_answer")
        if contains_negation:
            review_flags.append("negation")
        if multi_topic:
            review_flags.append("multi_topic")
        return {
            "predicted_category_id": FALLBACK_CATEGORY_ID,
            "predicted_category_name": FALLBACK_CATEGORY_NAME,
            "confidence": 0.0,
            "reason": f"top=none second=none flags={'|'.join(review_flags)}",
            "needs_human_review": True,
        }

    score_gap = top["score"] - second["score"] if second is not None else top["score"]
    ambiguity = second is not None and top["score"] == second["score"] and top["score"] > 0
    weak_gap = second is not None and score_gap <= 1 and top["score"] > 0
    denominator = max(1, len(top["matched"]) + len(top["excluded"]) + (1 if second is not None else 0))
    confidence = round(top["score"] / denominator, 3)

    flags: list[str] = []
    if ambiguity:
        flags.append("ambiguous_top_score")
    if weak_gap:
        flags.append("weak_score_gap")
    if is_short_answer:
        flags.append("short_answer")
    if contains_negation:
        flags.append("negation")
    if multi_topic:
        flags.append("multi_topic")
    if confidence < confidence_threshold:
        flags.append("low_confidence")

    reason = (
        f"top={format_candidate_summary(top)} "
        f"second={format_candidate_summary(second)} "
        f"gap={score_gap} "
        f"flags={'|'.join(flags) if flags else 'none'}"
    )
    return {
        "predicted_category_id": top["category_id"],
        "predicted_category_name": top["category_name"],
        "confidence": confidence,
        "reason": reason,
        "needs_human_review": bool(flags),
    }


def build_final_labels_df(
    responses_df: pd.DataFrame,
    categories: list[dict[str, object]],
    question_id: str,
    confidence_threshold: float,
) -> pd.DataFrame:
    filtered = responses_df[
        (responses_df["question_id"] == question_id)
        & (responses_df["is_target"].astype(str).str.lower() == "true")
    ].copy()
    classifications = filtered["answer_text"].map(
        lambda answer_text: classify_row(
            answer_text=str(answer_text),
            categories=categories,
            confidence_threshold=confidence_threshold,
        )
    )
    for column in FINAL_LABEL_COLUMNS[3:]:
        filtered[column] = classifications.map(lambda item, key=column: item[key])
    return filtered[FINAL_LABEL_COLUMNS]


def validate_predicted_category_ids(final_labels_df: pd.DataFrame, category_master_df: pd.DataFrame) -> list[str]:
    allowed_category_ids = set(category_master_df["category_id"].astype(str).tolist()) | {FALLBACK_CATEGORY_ID}
    predicted_ids = set(final_labels_df["predicted_category_id"].astype(str).tolist())
    invalid_ids = sorted(predicted_ids - allowed_category_ids)
    if not invalid_ids:
        return []
    return [f"predicted_category_id contains values not found in category_master.csv: {', '.join(invalid_ids)}"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rule-based final classification for one question.")
    parser.add_argument("--input", required=True, type=Path, help="Path to screened_responses.csv")
    parser.add_argument("--question-id", required=True, help="Target question_id")
    parser.add_argument("--category-master", required=True, type=Path, help="Path to category master CSV")
    parser.add_argument("--output", required=True, type=Path, help="Path to final_labels.csv")
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    responses_df = read_csv(args.input)
    validate_required_columns(responses_df, SCREENED_COLUMNS)
    screened_errors = run_screened_validations(responses_df)
    if screened_errors:
        raise SystemExit("\n".join(screened_errors))

    category_master_df = read_csv(args.category_master)
    validate_required_columns(category_master_df, CATEGORY_MASTER_COLUMNS)
    category_master_errors = run_category_master_validations(category_master_df)
    if category_master_errors:
        raise SystemExit("\n".join(category_master_errors))

    categories = build_categories_df(category_master_df)

    final_labels_df = build_final_labels_df(
        responses_df=responses_df,
        categories=categories,
        question_id=args.question_id,
        confidence_threshold=args.confidence_threshold,
    )
    final_label_errors = run_final_label_validations(final_labels_df)
    final_label_errors.extend(validate_predicted_category_ids(final_labels_df, category_master_df))
    if final_label_errors:
        raise SystemExit("\n".join(final_label_errors))
    write_csv(final_labels_df, args.output)
    if args.log is not None:
        append_jsonl(
            {
                "event": "classification",
                "input": str(args.input),
                "category_master": str(args.category_master),
                "output": str(args.output),
                "question_id": args.question_id,
                "row_count": int(len(final_labels_df)),
                "created_at": utc_now_iso(),
            },
            args.log,
        )


if __name__ == "__main__":
    main()
