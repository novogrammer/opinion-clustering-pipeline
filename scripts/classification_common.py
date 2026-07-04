from __future__ import annotations


CATEGORY_MASTER_COLUMNS = [
    "category_id",
    "category_name",
    "category_definition",
    "include_criteria",
    "exclude_criteria",
    "example_positive",
    "example_negative",
]

OVERRIDE_RULE_COLUMNS = [
    "rule_id",
    "question_id",
    "match_type",
    "pattern",
    "override_category_id",
    "override_category_name",
    "needs_human_review",
    "priority",
    "note",
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
