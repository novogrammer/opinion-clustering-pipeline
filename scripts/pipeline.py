#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

from common import append_jsonl, utc_now_iso, validate_identifier, validate_project_scaffold, validate_question_scaffold


SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = Path("projects")


def run_script(script_name: str, args: list[str]) -> None:
    command = [sys.executable, str(SCRIPTS_DIR / script_name), *args]
    subprocess.run(command, check=True)


def project_dir(project_name: str) -> Path:
    return PROJECTS_DIR / project_name


def question_dir(project_name: str, question_id: str) -> Path:
    return project_dir(project_name) / "questions" / question_id


def project_log_path(project_name: str, log_name: str) -> Path:
    return project_dir(project_name) / "99_logs" / log_name


def question_stage_log_path(project_name: str, question_id: str, stage_dir: str, log_name: str) -> Path:
    return question_dir(project_name, question_id) / stage_dir / log_name


def pipeline_log_path(project_name: str) -> Path:
    return project_log_path(project_name, "pipeline.log")


def run_pipeline_command(*, project_name: str, command_name: str, script_name: str, script_args: list[str]) -> None:
    status = "success"
    error_message = ""
    try:
        run_script(script_name, script_args)
    except subprocess.CalledProcessError as exc:
        status = "error"
        error_message = str(exc)
        raise
    finally:
        append_jsonl(
            {
                "event": "pipeline_command",
                "command": command_name,
                "script": script_name,
                "args": script_args,
                "status": status,
                "error": error_message,
                "created_at": utc_now_iso(),
            },
            pipeline_log_path(project_name),
        )


def add_project_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-name", required=True)


def add_question_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--question-id", required=True)


def validate_public_identifiers(args: argparse.Namespace) -> None:
    if hasattr(args, "project_name"):
        validate_identifier(args.project_name, "project_name")
    if hasattr(args, "question_id"):
        validate_identifier(args.question_id, "question_id")


def validate_command_context(args: argparse.Namespace) -> None:
    if args.command == "init-project":
        return
    current_project_dir = project_dir(args.project_name)
    validate_project_scaffold(current_project_dir)
    if args.command in {"init-question", "validate-processed", "validate-screening"}:
        return
    if hasattr(args, "question_id"):
        validate_question_scaffold(current_project_dir, args.question_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CLI for the opinion clustering pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparser = subparsers.add_parser("init-project", help="Create projects/{project_name}")
    add_project_argument(subparser)

    subparser = subparsers.add_parser("init-question", help="Create questions/{question_id}")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("normalize", help="Generate 01_processed/responses_normalized.csv from one raw CSV")
    add_project_argument(subparser)
    subparser.add_argument("--input", required=True)
    subparser.add_argument("--response-id-col", default=None)
    subparser.add_argument("--response-id-value", default=None)
    subparser.add_argument("--question-id-col", default=None)
    subparser.add_argument("--question-id-value", default=None)
    subparser.add_argument("--question-text-col", default=None)
    subparser.add_argument("--question-text-value", default=None)
    subparser.add_argument("--answer-text-col", default=None)
    subparser.add_argument("--answer-text-value", default=None)
    subparser.add_argument("--mask-emails", action="store_true")
    subparser.add_argument("--mask-urls", action="store_true")
    subparser.add_argument("--mask-phones", action="store_true")

    subparser = subparsers.add_parser("validate-processed", help="Validate 01_processed")
    add_project_argument(subparser)

    subparser = subparsers.add_parser("screening", help="Generate 02_screening/screened_responses.csv")
    add_project_argument(subparser)

    subparser = subparsers.add_parser("validate-screening", help="Validate 02_screening")
    add_project_argument(subparser)

    subparser = subparsers.add_parser("embeddings", help="Generate 03_embeddings artifacts")
    add_project_argument(subparser)
    add_question_argument(subparser)
    subparser.add_argument("--model", default="text-embedding-3-small")
    subparser.add_argument("--batch-size", type=int, default=100)
    subparser.add_argument("--max-retries", type=int, default=3)
    subparser.add_argument("--retry-base-seconds", type=float, default=1.0)
    subparser.add_argument("--force", action="store_true")

    subparser = subparsers.add_parser("clustering", help="Generate 04_clustering artifacts")
    add_project_argument(subparser)
    add_question_argument(subparser)
    subparser.add_argument("--umap-n-neighbors", type=int, default=15)
    subparser.add_argument("--umap-n-components", type=int, default=5)
    subparser.add_argument("--hdbscan-min-cluster-size", type=int, default=10)
    subparser.add_argument("--hdbscan-min-samples", type=int, default=5)
    subparser.add_argument("--random-state", type=int, default=42)
    subparser.add_argument("--force", action="store_true")

    subparser = subparsers.add_parser("classification", help="Generate 05_classification/final_labels.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("review", help="Generate 06_review/review_log.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("validate-question", help="Validate all question-level artifacts")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("validate-project", help="Validate all project-level artifacts")
    add_project_argument(subparser)

    subparser = subparsers.add_parser("validate-log", help="Validate a JSONL log file under 99_logs")
    add_project_argument(subparser)
    subparser.add_argument("--log-name", default="pipeline.log")

    return parser


def main() -> None:
    try:
        args = build_parser().parse_args()
        validate_public_identifiers(args)
        validate_command_context(args)
    except (FileExistsError, FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise SystemExit(str(exc))

    if args.command == "init-project":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="init_project.py",
            script_args=["--project-name", args.project_name],
        )
        return

    if args.command == "init-question":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="init_question.py",
            script_args=["--project-dir", str(project_dir(args.project_name)), "--question-id", args.question_id],
        )
        return

    if args.command == "normalize":
        current_project_dir = project_dir(args.project_name)
        command_args = [
            "--input",
            args.input,
            "--output",
            str(current_project_dir / "01_processed" / "responses_normalized.csv"),
            "--mapping-log",
            str(current_project_dir / "99_logs" / "raw_to_processed_mapping.md"),
            "--run-log",
            str(current_project_dir / "99_logs" / "raw_to_processed.log"),
        ]
        for option_name in [
            "response_id_col",
            "response_id_value",
            "question_id_col",
            "question_id_value",
            "question_text_col",
            "question_text_value",
            "answer_text_col",
            "answer_text_value",
        ]:
            option_value = getattr(args, option_name)
            if option_value is not None:
                command_args.extend([f"--{option_name.replace('_', '-')}", option_value])
        for option_name in ["mask_emails", "mask_urls", "mask_phones"]:
            if getattr(args, option_name):
                command_args.append(f"--{option_name.replace('_', '-')}")
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="normalize_processed.py",
            script_args=command_args,
        )
        return

    if args.command == "validate-processed":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="normalize_processed.py",
            script_args=[
                "--input",
                str(project_dir(args.project_name) / "01_processed" / "responses_normalized.csv"),
                "--validate",
            ],
        )
        return

    if args.command == "screening":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="screening.py",
            script_args=[
                "--input",
                str(project_dir(args.project_name) / "01_processed" / "responses_normalized.csv"),
                "--output",
                str(project_dir(args.project_name) / "02_screening" / "screened_responses.csv"),
                "--log",
                str(project_log_path(args.project_name, "screening.log")),
            ],
        )
        return

    if args.command == "validate-screening":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="screening.py",
            script_args=[
                "--input",
                str(project_dir(args.project_name) / "02_screening" / "screened_responses.csv"),
                "--validate",
            ],
        )
        return

    if args.command == "embeddings":
        command_args = [
            "--input",
            str(project_dir(args.project_name) / "02_screening" / "screened_responses.csv"),
            "--question-id",
            args.question_id,
            "--output-dir",
            str(question_dir(args.project_name, args.question_id) / "03_embeddings"),
            "--model",
            args.model,
            "--batch-size",
            str(args.batch_size),
            "--max-retries",
            str(args.max_retries),
            "--retry-base-seconds",
            str(args.retry_base_seconds),
            "--log",
            str(question_stage_log_path(args.project_name, args.question_id, "03_embeddings", "embedding.log")),
        ]
        if args.force:
            command_args.append("--force")
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="embeddings.py",
            script_args=command_args,
        )
        return

    if args.command == "clustering":
        current_question_dir = question_dir(args.project_name, args.question_id)
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="clustering.py",
            script_args=[
                "--input",
                str(project_dir(args.project_name) / "02_screening" / "screened_responses.csv"),
                "--question-id",
                args.question_id,
                "--embeddings",
                str(current_question_dir / "03_embeddings" / "embeddings.npy"),
                "--output-dir",
                str(current_question_dir / "04_clustering"),
                "--umap-n-neighbors",
                str(args.umap_n_neighbors),
                "--umap-n-components",
                str(args.umap_n_components),
                "--hdbscan-min-cluster-size",
                str(args.hdbscan_min_cluster_size),
                "--hdbscan-min-samples",
                str(args.hdbscan_min_samples),
                "--random-state",
                str(args.random_state),
                *(["--force"] if args.force else []),
                "--log",
                str(question_stage_log_path(args.project_name, args.question_id, "04_clustering", "clustering.log")),
            ],
        )
        return

    if args.command == "classification":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="classification.py",
            script_args=[
                "--input",
                str(project_dir(args.project_name) / "02_screening" / "screened_responses.csv"),
                "--question-id",
                args.question_id,
                "--category-master",
                str(question_dir(args.project_name, args.question_id) / "05_classification" / "category_master.csv"),
                "--output",
                str(question_dir(args.project_name, args.question_id) / "05_classification" / "final_labels.csv"),
                "--log",
                str(question_stage_log_path(args.project_name, args.question_id, "05_classification", "classification.log")),
            ],
        )
        return

    if args.command == "review":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="review_prep.py",
            script_args=[
                "--input",
                str(question_dir(args.project_name, args.question_id) / "05_classification" / "final_labels.csv"),
                "--output",
                str(question_dir(args.project_name, args.question_id) / "06_review" / "review_log.csv"),
                "--screened",
                str(project_dir(args.project_name) / "02_screening" / "screened_responses.csv"),
                "--log",
                str(question_stage_log_path(args.project_name, args.question_id, "06_review", "review.log")),
            ],
        )
        return

    if args.command == "validate-question":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_question_artifacts.py",
            script_args=[
                "--question-dir",
                str(question_dir(args.project_name, args.question_id)),
            ],
        )
        return

    if args.command == "validate-project":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_project_artifacts.py",
            script_args=[
                "--project-dir",
                str(project_dir(args.project_name)),
            ],
        )
        return

    if args.command == "validate-log":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_jsonl_log.py",
            script_args=[
                "--input",
                str(project_log_path(args.project_name, args.log_name)),
            ],
        )
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
