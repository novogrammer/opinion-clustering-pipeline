from __future__ import annotations

import re
import unicodedata

import pandas as pd


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
    for column in [
        "category_name",
        "category_definition",
        "include_criteria",
        "example_positive",
    ]:
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
