from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from classification_common import CATEGORY_MASTER_COLUMNS, FINAL_LABEL_COLUMNS, OVERRIDE_RULE_COLUMNS
from classification_keywords import build_categories_df, collect_category_keywords, collect_exclude_keywords, normalize_text, split_keywords
from common import REQUIRED_RESPONSE_COLUMNS, append_jsonl, read_csv, utc_now_iso, validate_required_columns, write_csv
from validate_category_master import run_validations as run_category_master_validations
from validate_final_labels import run_validations as run_final_labels_validations
from validate_override_rules import run_validations as run_override_rule_validations
from validate_screened_responses import run_validations as run_screened_validations

FALLBACK_CATEGORY_ID = "OTHER"
FALLBACK_CATEGORY_NAME = "その他"
NEGATION_TERMS = ["ない", "ぬ", "ません", "ではない", "じゃない", "なく", "ず", "微妙", "不満", "困る"]
MULTI_TOPIC_MARKERS = ["。", "、", " and ", "・", "/", "また", "けど", "が", "しかし"]

def parse_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


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


def load_override_rules(path: Path | None, question_id: str) -> list[dict[str, object]]:
    if path is None or not path.exists():
        return []
    override_df = read_csv(path)
    validate_required_columns(override_df, OVERRIDE_RULE_COLUMNS)
    filtered = override_df[override_df["question_id"].astype(str).isin([question_id, "*"])].copy()
    if len(filtered) == 0:
        return []

    def parse_priority(value: object) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 9999

    filtered["priority_int"] = filtered["priority"].map(parse_priority)
    filtered = filtered.sort_values(by=["priority_int", "rule_id"], ascending=[True, True], kind="stable")

    rules: list[dict[str, object]] = []
    for _, row in filtered.iterrows():
        rules.append(
            {
                "rule_id": str(row["rule_id"]).strip(),
                "match_type": str(row["match_type"]).strip().lower(),
                "pattern": str(row["pattern"]).strip(),
                "override_category_id": str(row["override_category_id"]).strip(),
                "override_category_name": str(row["override_category_name"]).strip(),
                "needs_human_review": parse_bool(row["needs_human_review"]),
                "priority": int(row["priority_int"]),
                "note": str(row["note"]).strip(),
            }
        )
    return rules


def matches_override_rule(answer_text: str, rule: dict[str, object]) -> bool:
    normalized = normalize_text(answer_text)
    normalized_casefold = normalized.casefold()
    pattern = str(rule["pattern"])
    match_type = str(rule["match_type"])

    if match_type == "exact":
        return normalized_casefold == normalize_text(pattern).casefold()
    if match_type == "contains":
        return normalize_text(pattern).casefold() in normalized_casefold
    if match_type == "regex":
        return re.search(pattern, normalized) is not None
    return False


def apply_override_rule(answer_text: str, override_rules: list[dict[str, object]]) -> dict[str, object] | None:
    for rule in override_rules:
        if not matches_override_rule(answer_text, rule):
            continue
        note = str(rule["note"]).strip() or "none"
        return {
            "predicted_category_id": rule["override_category_id"],
            "predicted_category_name": rule["override_category_name"],
            "confidence": 1.0,
            "reason": (
                f"override_rule={rule['rule_id']} "
                f"match_type={rule['match_type']} "
                f"pattern={rule['pattern']} "
                f"note={note}"
            ),
            "needs_human_review": bool(rule["needs_human_review"]),
        }
    return None


def classify_row(
    answer_text: str,
    categories: list[dict[str, object]],
    confidence_threshold: float,
    override_rules: list[dict[str, object]],
) -> dict[str, object]:
    override_result = apply_override_rule(answer_text, override_rules)
    if override_result is not None:
        return override_result

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
    needs_human_review = bool(flags)

    return {
        "predicted_category_id": top["category_id"],
        "predicted_category_name": top["category_name"],
        "confidence": confidence,
        "reason": reason,
        "needs_human_review": needs_human_review,
    }



def build_final_labels_df(
    responses_df: pd.DataFrame,
    categories: list[dict[str, object]],
    question_id: str,
    confidence_threshold: float,
    override_rules: list[dict[str, object]],
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
            override_rules=override_rules,
        )
    )
    for column in FINAL_LABEL_COLUMNS[3:]:
        filtered[column] = classifications.map(lambda item, key=column: item[key])
    return filtered[FINAL_LABEL_COLUMNS]


def validate_predicted_category_ids(
    final_labels_df: pd.DataFrame,
    category_master_df: pd.DataFrame,
) -> list[str]:
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
    parser.add_argument(
        "--override-rules",
        type=Path,
        default=None,
        help="Optional path to manual_override_rules.csv",
    )
    parser.add_argument("--output", required=True, type=Path, help="Path to final_labels.csv")
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append execution logs as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    responses_df = read_csv(args.input)
    validate_required_columns(responses_df, REQUIRED_RESPONSE_COLUMNS + ["is_target", "screening_reason"])
    screened_errors = run_screened_validations(responses_df)
    if screened_errors:
        raise SystemExit("\n".join(screened_errors))

    category_master_df = read_csv(args.category_master)
    validate_required_columns(category_master_df, CATEGORY_MASTER_COLUMNS)
    category_master_errors = run_category_master_validations(category_master_df)
    if category_master_errors:
        raise SystemExit("\n".join(category_master_errors))

    categories = build_categories_df(category_master_df)
    override_rules_path = args.override_rules
    if override_rules_path is None:
        default_override_rules = args.category_master.parent / "manual_override_rules.csv"
        override_rules_path = default_override_rules if default_override_rules.exists() else None
    if override_rules_path is not None:
        override_rules_df = read_csv(override_rules_path)
        validate_required_columns(override_rules_df, OVERRIDE_RULE_COLUMNS)
        override_rule_errors = run_override_rule_validations(override_rules_df)
        if override_rule_errors:
            raise SystemExit("\n".join(override_rule_errors))
    override_rules = load_override_rules(override_rules_path, args.question_id)
    final_labels_df = build_final_labels_df(
        responses_df=responses_df,
        categories=categories,
        question_id=args.question_id,
        confidence_threshold=args.confidence_threshold,
        override_rules=override_rules,
    )
    final_label_errors = run_final_labels_validations(final_labels_df)
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
                "override_rule_count": int(len(override_rules)),
                "created_at": utc_now_iso(),
            },
            args.log,
        )


if __name__ == "__main__":
    main()
