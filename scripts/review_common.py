from __future__ import annotations


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

REVIEW_SUMMARY_COLUMNS = [
    "question_id",
    "predicted_category_id",
    "predicted_category_name",
    "total_count",
    "reviewed_count",
    "corrected_count",
    "pending_count",
    "skipped_count",
    "high_priority_count",
    "medium_priority_count",
    "low_priority_count",
    "correction_rate",
    "top_review_triggers",
    "needs_definition_review",
    "definition_review_reason",
]

CATEGORY_REVIEW_PRIORITY_COLUMNS = [
    "question_id",
    "category_id",
    "category_name",
    "needs_definition_review",
    "definition_review_reason",
    "correction_rate",
    "corrected_count",
    "high_priority_count",
    "conflict_pair_count",
    "high_conflict_pair_count",
    "conflict_categories",
    "priority_score",
    "priority_rank",
]

REVIEW_SAMPLE_COLUMNS = REVIEW_COLUMNS + [
    "sample_reason",
    "sample_bucket",
]

REVIEW_CORRECTION_COLUMNS = [
    "response_id",
    "question_id",
    "answer_text",
    "predicted_category_id",
    "predicted_category_name",
    "reviewed_category_id",
    "reviewed_category_name",
    "confidence",
    "reason",
    "review_trigger",
    "review_priority",
    "review_comment",
    "reviewer",
    "reviewed_at",
    "correction_type",
]
