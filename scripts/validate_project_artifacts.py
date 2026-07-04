from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import REQUIRED_RESPONSE_COLUMNS, append_jsonl, read_csv, utc_now_iso, validate_required_columns
from duplicate_common import DUPLICATE_OUTPUT_COLUMNS, normalize_answer_text
from validate_processed import run_validations as run_processed_validations
from validate_raw_to_processed_mapping import run_validations as run_raw_to_processed_mapping_validations
from validate_question_artifacts import collect_question_errors
from validate_jsonl_log import run_validations as run_jsonl_log_validations
from validate_duplicate_responses import run_validations as run_duplicate_response_validations


DUPLICATE_COLUMNS = DUPLICATE_OUTPUT_COLUMNS
from validate_screened_responses import SCREENED_COLUMNS, run_validations as run_screened_responses_validations


def validate_question_dir(question_dir: Path) -> list[str]:
    raw_errors = collect_question_errors(question_dir)
    normalized_errors: list[str] = []
    for message in raw_errors:
        if message.startswith("Missing file: "):
            normalized_errors.append(message.replace("Missing file: ", f"{question_dir.name}/"))
        else:
            normalized_errors.append(f"{question_dir.name}/{message}")
    return normalized_errors


def stage_has_any(paths: list[Path]) -> bool:
    return any(path.exists() for path in paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate project-level artifacts under projects/{project_name}")
    parser.add_argument("--project-dir", required=True, type=Path, help="Path to projects/{project_name}")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append validation results as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir

    errors: list[str] = []

    processed_path = project_dir / "01_processed" / "responses_normalized.csv"
    screened_path = project_dir / "02_screening" / "screened_responses.csv"
    duplicate_path = project_dir / "02_screening" / "duplicate_responses.csv"
    questions_dir = project_dir / "questions"
    question_dirs = sorted(path for path in questions_dir.iterdir() if path.is_dir()) if questions_dir.exists() else []
    mapping_path = project_dir / "99_logs" / "raw_to_processed_mapping.md"
    project_log_paths = [
        project_dir / "99_logs" / "pipeline.log",
        project_dir / "99_logs" / "raw_to_processed.log",
        project_dir / "99_logs" / "screening.log",
    ]

    project_has_question_outputs = bool(question_dirs)
    project_has_screening_outputs = stage_has_any([screened_path, duplicate_path])
    project_has_normalize_outputs = (
        project_has_screening_outputs
        or project_has_question_outputs
        or processed_path.exists()
        or mapping_path.exists()
        or any(path.exists() for path in project_log_paths)
    )

    screened_df = None
    if project_has_normalize_outputs and not processed_path.exists():
        errors.append(f"Missing file: {processed_path}")
    elif processed_path.exists():
        try:
            processed_df = read_csv(processed_path)
            validate_required_columns(processed_df, REQUIRED_RESPONSE_COLUMNS)
            processed_errors = run_processed_validations(processed_df)
            errors.extend(f"01_processed/responses_normalized.csv: {message}" for message in processed_errors)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"01_processed/responses_normalized.csv: {exc}")

    if screened_path.exists():
        try:
            screened_df = read_csv(screened_path)
            validate_required_columns(screened_df, SCREENED_COLUMNS)
            screened_errors = run_screened_responses_validations(screened_df)
            errors.extend(f"02_screening/screened_responses.csv: {message}" for message in screened_errors)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"02_screening/screened_responses.csv: {exc}")

    if duplicate_path.exists():
        try:
            duplicate_df = read_csv(duplicate_path)
            validate_required_columns(duplicate_df, DUPLICATE_COLUMNS)
            duplicate_errors = run_duplicate_response_validations(duplicate_df)
            errors.extend(f"02_screening/duplicate_responses.csv: {message}" for message in duplicate_errors)
            if screened_df is not None:
                target_df = screened_df[screened_df["is_target"].astype(str).str.lower() == "true"].copy()
                target_ids = set(target_df["response_id"].astype(str).tolist())
                duplicate_ids = set(duplicate_df["response_id"].astype(str).tolist())
                invalid_ids = sorted(duplicate_ids - target_ids)
                if invalid_ids:
                    errors.append(
                        "02_screening/duplicate_responses.csv: response_id values not found in screened target rows"
                    )

                allowed_question_ids = set(target_df["question_id"].astype(str).tolist())
                duplicate_question_ids = set(duplicate_df["question_id"].astype(str).tolist())
                invalid_question_ids = sorted(duplicate_question_ids - allowed_question_ids)
                if invalid_question_ids:
                    errors.append(
                        "02_screening/duplicate_responses.csv: question_id values not found in screened target rows"
                    )

                target_lookup = (
                    target_df[["response_id", "question_id", "answer_text"]]
                    .astype(str)
                    .set_index("response_id")
                    .to_dict(orient="index")
                )
                for group_id, group in duplicate_df.groupby("duplicate_group_id", sort=False):
                    group_question_ids = set(group["question_id"].astype(str).tolist())
                    if len(group_question_ids) != 1:
                        errors.append(
                            f"02_screening/duplicate_responses.csv: {group_id} spans multiple question_id values"
                        )
                    normalized_answers = {
                        normalize_answer_text(target_lookup[str(response_id)]["answer_text"])
                        for response_id in group["response_id"].astype(str).tolist()
                        if str(response_id) in target_lookup
                    }
                    if len(normalized_answers) > 1:
                        errors.append(
                            f"02_screening/duplicate_responses.csv: {group_id} contains non-matching normalized answers"
                        )
                    canonical_ids = set(group["canonical_response_id"].astype(str).tolist())
                    response_ids = set(group["response_id"].astype(str).tolist())
                    if canonical_ids - response_ids:
                        errors.append(
                            f"02_screening/duplicate_responses.csv: {group_id} canonical_response_id is not included in its group"
                        )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"02_screening/duplicate_responses.csv: {exc}")

    if project_has_question_outputs and not questions_dir.exists():
        errors.append(f"Missing directory: {questions_dir}")
    elif questions_dir.exists():
        for question_dir in question_dirs:
            errors.extend(validate_question_dir(question_dir))

    if mapping_path.exists():
        try:
            mapping_errors = run_raw_to_processed_mapping_validations(mapping_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{mapping_path.name}: {exc}")
        else:
            errors.extend(f"{mapping_path.name}: {message}" for message in mapping_errors)
    for log_path in project_log_paths:
        if not log_path.exists():
            continue
        try:
            log_errors = run_jsonl_log_validations(log_path.read_text(encoding="utf-8").splitlines())
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{log_path.name}: {exc}")
            continue
        errors.extend(f"{log_path.name}: {message}" for message in log_errors)

    log_payload = {
        "event": "validate_project_artifacts",
        "project_dir": str(project_dir),
        "success": len(errors) == 0,
        "errors": errors,
        "created_at": utc_now_iso(),
    }
    if args.log is not None:
        append_jsonl(log_payload, args.log)

    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
