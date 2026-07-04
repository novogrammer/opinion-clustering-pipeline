from __future__ import annotations

from classification_common import OVERRIDE_RULE_COLUMNS


OVERRIDE_CANDIDATE_COLUMNS = OVERRIDE_RULE_COLUMNS + [
    "approved",
    "approved_note",
    "source_response_id",
    "source_predicted_category_id",
    "source_predicted_category_name",
    "source_review_comment",
    "source_correction_type",
]

OVERRIDE_RULE_HIT_COLUMNS = [
    "response_id",
    "question_id",
    "answer_text",
    "rule_id",
    "match_type",
    "pattern",
    "override_category_id",
    "override_category_name",
    "needs_human_review",
    "priority",
    "note",
]

OVERRIDE_RULE_SUMMARY_COLUMNS = [
    "rule_id",
    "question_id",
    "match_type",
    "pattern",
    "override_category_id",
    "override_category_name",
    "needs_human_review",
    "priority",
    "hit_count",
    "unique_answer_count",
    "sample_response_ids",
    "note",
]
