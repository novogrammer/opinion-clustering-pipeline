from __future__ import annotations

import re
import unicodedata

from common import REQUIRED_RESPONSE_COLUMNS


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
SCREENED_COLUMNS = REQUIRED_RESPONSE_COLUMNS + ["is_target", "screening_reason"]
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
