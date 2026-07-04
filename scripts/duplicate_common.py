from __future__ import annotations

import unicodedata


DUPLICATE_OUTPUT_COLUMNS = [
    "duplicate_group_id",
    "question_id",
    "response_id",
    "canonical_response_id",
    "duplicate_count",
    "is_canonical",
    "answer_text",
]


def normalize_answer_text(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value))
    return normalized.strip()
