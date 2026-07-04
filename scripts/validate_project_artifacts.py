from __future__ import annotations

import argparse
from pathlib import Path

from common import REQUIRED_RESPONSE_COLUMNS, append_jsonl, read_csv, utc_now_iso, validate_required_columns
from validate_jsonl_log import run_validations as run_jsonl_log_validations
from validate_processed import run_validations as run_processed_validations
from validate_question_artifacts import collect_question_errors
from validate_raw_to_processed_mapping import run_validations as run_raw_to_processed_mapping_validations
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
    mapping_path = project_dir / "99_logs" / "raw_to_processed_mapping.md"
    questions_dir = project_dir / "questions"
    question_dirs = sorted(path for path in questions_dir.iterdir() if path.is_dir()) if questions_dir.exists() else []
    project_log_paths = [
        project_dir / "99_logs" / "pipeline.log",
        project_dir / "99_logs" / "raw_to_processed.log",
        project_dir / "99_logs" / "screening.log",
    ]

    project_started = stage_has_any([processed_path, screened_path, mapping_path, *project_log_paths]) or bool(question_dirs)
    screening_required = screened_path.exists() or bool(question_dirs)

    if project_started and not processed_path.exists():
        errors.append(f"Missing file: {processed_path}")
    if screening_required and not screened_path.exists():
        errors.append(f"Missing file: {screened_path}")

    if processed_path.exists():
        processed_df = read_csv(processed_path)
        validate_required_columns(processed_df, REQUIRED_RESPONSE_COLUMNS)
        errors.extend(f"01_processed/responses_normalized.csv: {message}" for message in run_processed_validations(processed_df))

    if screened_path.exists():
        screened_df = read_csv(screened_path)
        validate_required_columns(screened_df, SCREENED_COLUMNS)
        errors.extend(f"02_screening/screened_responses.csv: {message}" for message in run_screened_responses_validations(screened_df))

    if mapping_path.exists():
        mapping_errors = run_raw_to_processed_mapping_validations(mapping_path.read_text(encoding="utf-8"))
        errors.extend(f"{mapping_path.name}: {message}" for message in mapping_errors)

    for log_path in project_log_paths:
        if not log_path.exists():
            continue
        log_errors = run_jsonl_log_validations(log_path.read_text(encoding="utf-8").splitlines())
        errors.extend(f"{log_path.name}: {message}" for message in log_errors)

    for question_dir in question_dirs:
        errors.extend(validate_question_dir(question_dir))

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
