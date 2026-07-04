from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import read_csv
from normalize_common import OUTPUT_COLUMNS, build_normalized_dataframe, write_processed_outputs
from validate_processed import run_validations as run_processed_validations


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize one raw CSV into 01_processed/responses_normalized.csv")
    parser.add_argument("--input", required=True, type=Path, help="Path to raw CSV")
    parser.add_argument("--output", required=True, type=Path, help="Path to responses_normalized.csv")
    parser.add_argument("--mapping-log", required=True, type=Path, help="Path to raw_to_processed_mapping.md")
    parser.add_argument("--run-log", required=True, type=Path, help="Path to raw_to_processed.log")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_csv(args.input)
    normalized_df, source_column_map = build_normalized_df(df, args)
    errors = run_processed_validations(normalized_df)
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
