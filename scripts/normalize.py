from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

from common import (
    REQUIRED_RESPONSE_COLUMNS,
    append_jsonl,
    read_csv,
    utc_now_iso,
    validate_required_columns,
    write_csv,
)


OUTPUT_COLUMNS = REQUIRED_RESPONSE_COLUMNS

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


def normalize_whitespace(value: str) -> str:
    return " ".join(str(value).replace("\t", " ").split())


def validate_no_duplicate_response_ids(df: pd.DataFrame) -> list[str]:
    duplicate_mask = df["response_id"].duplicated(keep=False)
    duplicates = df.loc[duplicate_mask, "response_id"].tolist()
    if not duplicates:
        return []
    joined = ", ".join(dict.fromkeys(duplicates))
    return [f"Duplicate response_id values found: {joined}"]


def validate_no_blank_core_values(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for column in ["response_id", "question_id", "question_text"]:
        blank_mask = df[column].map(lambda value: str(value).strip() == "")
        count = int(blank_mask.sum())
        if count > 0:
            errors.append(f"Blank values found in {column}: {count}")
    return errors


def validate_normalized_answer_text(df: pd.DataFrame) -> list[str]:
    non_normalized = df["answer_text"].map(lambda value: str(value) != normalize_whitespace(value))
    count = int(non_normalized.sum())
    if count == 0:
        return []
    return [f"Non-normalized answer_text values found: {count}"]


def validate_question_ids(df: pd.DataFrame) -> list[str]:
    blank_mask = df["question_id"].map(lambda value: str(value).strip() == "")
    if int(blank_mask.sum()) == 0:
        return []
    return ["question_id contains blank values"]


def run_validations(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_no_duplicate_response_ids(df))
    errors.extend(validate_no_blank_core_values(df))
    errors.extend(validate_normalized_answer_text(df))
    errors.extend(validate_question_ids(df))
    return errors


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


def resolve_column(
    df: pd.DataFrame,
    *,
    source_column: str | None,
    constant_value: str | None,
    output_name: str,
    row_count: int,
) -> pd.Series:
    if source_column and constant_value is not None:
        raise ValueError(f"Specify either --{output_name.replace('_', '-')}-col or --{output_name.replace('_', '-')}-value")
    if source_column:
        if source_column not in df.columns:
            raise ValueError(f"Missing source column for {output_name}: {source_column}")
        return df[source_column].astype(str)
    if constant_value is not None:
        return pd.Series([constant_value] * row_count, dtype=str)
    raise ValueError(f"{output_name} requires a source column or constant value")


def build_normalized_df(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, str]]:
    row_count = len(df)
    normalized = pd.DataFrame(
        {
            "response_id": resolve_column(
                df,
                source_column=args.response_id_col,
                constant_value=args.response_id_value,
                output_name="response_id",
                row_count=row_count,
            ),
            "question_id": resolve_column(
                df,
                source_column=args.question_id_col,
                constant_value=args.question_id_value,
                output_name="question_id",
                row_count=row_count,
            ),
            "question_text": resolve_column(
                df,
                source_column=args.question_text_col,
                constant_value=args.question_text_value,
                output_name="question_text",
                row_count=row_count,
            ),
            "answer_text": resolve_column(
                df,
                source_column=args.answer_text_col,
                constant_value=args.answer_text_value,
                output_name="answer_text",
                row_count=row_count,
            ),
        }
    )
    source_column_map: dict[str, str] = {}
    for output_name in OUTPUT_COLUMNS:
        source_column = getattr(args, f"{output_name}_col")
        constant_value = getattr(args, f"{output_name}_value")
        if source_column is not None:
            source_column_map[source_column] = output_name
        elif constant_value is not None:
            source_column_map[f"(constant) {constant_value}"] = output_name

    normalized = build_normalized_dataframe(
        normalized,
        {column: column for column in OUTPUT_COLUMNS},
        mask_emails=args.mask_emails,
        mask_urls=args.mask_urls,
        mask_phones=args.mask_phones,
    )
    return normalized, source_column_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize one raw CSV into 01_processed/responses_normalized.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to raw CSV")
    parser.add_argument("--output", type=Path, default=None, help="Path to responses_normalized.csv")
    parser.add_argument("--mapping-log", type=Path, default=None, help="Path to raw_to_processed_mapping.md")
    parser.add_argument("--run-log", type=Path, default=None, help="Path to raw_to_processed.log")
    parser.add_argument("--response-id-col", default=None, help="Source column mapped to response_id")
    parser.add_argument("--response-id-value", default=None, help="Constant value for response_id")
    parser.add_argument("--question-id-col", default=None, help="Source column mapped to question_id")
    parser.add_argument("--question-id-value", default=None, help="Constant value for question_id")
    parser.add_argument("--question-text-col", default=None, help="Source column mapped to question_text")
    parser.add_argument("--question-text-value", default=None, help="Constant value for question_text")
    parser.add_argument("--answer-text-col", default=None, help="Source column mapped to answer_text")
    parser.add_argument("--answer-text-value", default=None, help="Constant value for answer_text")
    parser.add_argument("--mask-emails", action="store_true")
    parser.add_argument("--mask-urls", action="store_true")
    parser.add_argument("--mask-phones", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = read_csv(args.input)

    if args.output is None or args.mapping_log is None or args.run_log is None:
        raise SystemExit("--output, --mapping-log, and --run-log are required")

    normalized_df, source_column_map = build_normalized_df(df, args)
    errors = run_validations(normalized_df)
    if errors:
        raise SystemExit("\n".join(errors))
    write_processed_outputs(
        normalized_df,
        input_path=args.input,
        output_path=args.output,
        mapping_log_path=args.mapping_log,
        run_log_path=args.run_log,
        source_column_map=source_column_map,
        mask_emails=args.mask_emails,
        mask_urls=args.mask_urls,
        mask_phones=args.mask_phones,
    )


if __name__ == "__main__":
    main()
