from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

from common import append_jsonl, utc_now_iso, validate_required_columns, write_csv


OUTPUT_COLUMNS = [
    "response_id",
    "question_id",
    "question_text",
    "answer_text",
]

EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
URL_PATTERN = re.compile(r"https?://[^\s]+")
PHONE_PATTERN = re.compile(r"(?:\+?\d[\d\-\s()]{7,}\d)")


def normalize_text(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value))
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    normalized = re.sub(r"[ ]{2,}", " ", normalized)
    normalized = re.sub(r"\n{2,}", "\n", normalized)
    return normalized.strip()


def mask_text(value: str, *, mask_emails: bool = False, mask_urls: bool = False, mask_phones: bool = False) -> str:
    masked = value
    if mask_emails:
        masked = EMAIL_PATTERN.sub("[EMAIL]", masked)
    if mask_urls:
        masked = URL_PATTERN.sub("[URL]", masked)
    if mask_phones:
        masked = PHONE_PATTERN.sub("[PHONE]", masked)
    return masked


def build_normalized_dataframe(
    df: pd.DataFrame,
    source_column_map: dict[str, str],
    *,
    mask_emails: bool = False,
    mask_urls: bool = False,
    mask_phones: bool = False,
) -> pd.DataFrame:
    if not source_column_map:
        raise ValueError("source_column_map is empty")

    missing = [column for column in source_column_map if column not in df.columns]
    if missing:
        raise ValueError(f"Missing source columns: {', '.join(missing)}")

    normalized = df.rename(columns=source_column_map)[list(source_column_map.values())].copy()
    validate_required_columns(normalized, OUTPUT_COLUMNS)

    normalized = normalized[OUTPUT_COLUMNS]
    for column in OUTPUT_COLUMNS:
        normalized[column] = normalized[column].map(normalize_text)

    normalized["answer_text"] = normalized["answer_text"].map(
        lambda value: mask_text(
            value,
            mask_emails=mask_emails,
            mask_urls=mask_urls,
            mask_phones=mask_phones,
        )
    )
    return normalized


def build_mapping_markdown(
    input_path: Path,
    output_path: Path,
    row_count: int,
    source_column_map: dict[str, str],
    *,
    mask_emails: bool = False,
    mask_urls: bool = False,
    mask_phones: bool = False,
) -> str:
    mappings = [f"- `{source}` -> `{target}`" for source, target in source_column_map.items()]
    transformations = [
        "- whitespace_normalization: enabled",
        "- unicode_normalization: NFKC",
    ]
    if mask_emails:
        transformations.append("- answer_text_email_mask: enabled")
    if mask_urls:
        transformations.append("- answer_text_url_mask: enabled")
    if mask_phones:
        transformations.append("- answer_text_phone_mask: enabled")

    return (
        "# raw_to_processed mapping\n\n"
        f"- source_file: `{input_path}`\n"
        f"- output_file: `{output_path}`\n"
        f"- row_count: {row_count}\n"
        "- column_mapping:\n"
        f"{chr(10).join(mappings)}\n"
        "- transformations:\n"
        f"{chr(10).join(transformations)}\n"
    )


def write_processed_outputs(
    normalized_df: pd.DataFrame,
    *,
    input_path: Path,
    output_path: Path,
    mapping_log_path: Path,
    run_log_path: Path,
    source_column_map: dict[str, str],
    mask_emails: bool = False,
    mask_urls: bool = False,
    mask_phones: bool = False,
) -> None:
    write_csv(normalized_df, output_path)
    mapping_log_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_log_path.write_text(
        build_mapping_markdown(
            input_path=input_path,
            output_path=output_path,
            row_count=len(normalized_df),
            source_column_map=source_column_map,
            mask_emails=mask_emails,
            mask_urls=mask_urls,
            mask_phones=mask_phones,
        ),
        encoding="utf-8",
    )
    append_jsonl(
        {
            "event": "raw_to_processed",
            "input": str(input_path),
            "output": str(output_path),
            "row_count": int(len(normalized_df)),
            "mask_emails": mask_emails,
            "mask_urls": mask_urls,
            "mask_phones": mask_phones,
            "created_at": utc_now_iso(),
        },
        run_log_path,
    )
