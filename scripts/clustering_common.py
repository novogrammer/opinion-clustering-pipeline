from __future__ import annotations


CLUSTER_COLUMNS = ["response_id", "question_id", "topic_id", "topic_probability", "is_outlier"]
SUMMARY_COLUMNS = [
    "question_id",
    "topic_id",
    "cluster_size",
    "representative_answers",
    "candidate_label",
    "candidate_definition",
    "include_criteria",
    "exclude_criteria",
    "split_suggestion",
    "confidence",
]
