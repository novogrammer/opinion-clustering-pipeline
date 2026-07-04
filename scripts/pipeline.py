#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

from common import (
    append_jsonl,
    utc_now_iso,
    validate_identifier,
    validate_project_scaffold,
    validate_question_scaffold,
)


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


def run_pipeline_command(
    *,
    project_name: str,
    command_name: str,
    script_name: str,
    script_args: list[str],
) -> None:
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

    if args.command == "init-question":
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

    subparser = subparsers.add_parser("validate-mapping", help="Validate raw_to_processed mapping")
    add_project_argument(subparser)

    subparser = subparsers.add_parser("screening", help="Generate 02_screening/screened_responses.csv")
    add_project_argument(subparser)

    subparser = subparsers.add_parser("duplicate-check", help="Generate 02_screening/duplicate_responses.csv")
    add_project_argument(subparser)

    subparser = subparsers.add_parser("validate-screening", help="Validate 02_screening")
    add_project_argument(subparser)

    subparser = subparsers.add_parser("validate-duplicates", help="Validate 02_screening/duplicate_responses.csv")
    add_project_argument(subparser)

    subparser = subparsers.add_parser("embeddings", help="Generate 03_embeddings artifacts")
    add_project_argument(subparser)
    add_question_argument(subparser)
    subparser.add_argument("--model", default="text-embedding-3-small")
    subparser.add_argument("--batch-size", type=int, default=100)
    subparser.add_argument("--max-retries", type=int, default=3)
    subparser.add_argument("--retry-base-seconds", type=float, default=1.0)
    subparser.add_argument("--force", action="store_true")
    subparser.add_argument("--prepare-only", action="store_true")

    subparser = subparsers.add_parser("validate-embedding-requests", help="Validate embedding_requests.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("validate-embedding-metadata", help="Validate embedding_metadata.json")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("validate-embedding-failures", help="Validate embedding_failures.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("validate-embeddings-array", help="Validate embeddings.npy")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("clustering", help="Generate 04_clustering artifacts")
    add_project_argument(subparser)
    add_question_argument(subparser)
    subparser.add_argument("--umap-n-neighbors", type=int, default=15)
    subparser.add_argument("--umap-n-components", type=int, default=5)
    subparser.add_argument("--hdbscan-min-cluster-size", type=int, default=10)
    subparser.add_argument("--hdbscan-min-samples", type=int, default=5)
    subparser.add_argument("--random-state", type=int, default=42)
    subparser.add_argument("--force", action="store_true")

    subparser = subparsers.add_parser("validate-clusters", help="Validate clusters.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("validate-cluster-summary", help="Validate cluster_summary.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("validate-clustering-metadata", help="Validate clustering_metadata.json")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("scaffold-category-master", help="Scaffold category_master.csv from cluster_summary.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)
    subparser.add_argument("--category-prefix", default="CAT")
    subparser.add_argument("--include-outlier", action="store_true")

    subparser = subparsers.add_parser("category-conflicts", help="Generate 05_classification/category_conflicts.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("validate-category-master", help="Validate category_master.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("validate-override-rules", help="Validate manual_override_rules.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("validate-category-conflicts", help="Validate category_conflicts.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("classification", help="Generate 05_classification/final_labels.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("validate-final-labels", help="Validate final_labels.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("review", help="Generate 06_review/review_log.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("review-summary", help="Generate 06_review/review_summary.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("review-priorities", help="Generate 06_review/category_review_priorities.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("review-samples", help="Generate 06_review/review_samples.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)
    subparser.add_argument("--medium-per-category", type=int, default=3)
    subparser.add_argument("--include-low", action="store_true")
    subparser.add_argument("--priority-trigger", default=None)

    subparser = subparsers.add_parser("review-corrections", help="Generate 06_review/review_corrections.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("override-candidates", help="Generate 05_classification/manual_override_candidates.csv from review_corrections.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("promote-override-candidates", help="Promote approved override candidates into manual_override_rules.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("override-rule-hits", help="Generate 05_classification/override_rule_hits.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("override-rule-summary", help="Generate 05_classification/override_rule_summary.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("validate-review-log", help="Validate review_log.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("validate-review-summary", help="Validate review_summary.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("validate-review-priorities", help="Validate category_review_priorities.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("validate-review-samples", help="Validate review_samples.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("validate-review-corrections", help="Validate review_corrections.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("validate-override-candidates", help="Validate manual_override_candidates.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("validate-override-rule-hits", help="Validate override_rule_hits.csv")
    add_project_argument(subparser)
    add_question_argument(subparser)

    subparser = subparsers.add_parser("validate-override-rule-summary", help="Validate override_rule_summary.csv")
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
            script_name="validate_processed.py",
            script_args=["--input", str(project_dir(args.project_name) / "01_processed" / "responses_normalized.csv")],
        )
        return

    if args.command == "validate-mapping":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_raw_to_processed_mapping.py",
            script_args=["--input", str(project_dir(args.project_name) / "99_logs" / "raw_to_processed_mapping.md")],
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
            script_name="validate_screened_responses.py",
            script_args=["--input", str(project_dir(args.project_name) / "02_screening" / "screened_responses.csv")],
        )
        return

    if args.command == "duplicate-check":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="duplicate_responses.py",
            script_args=[
                "--input",
                str(project_dir(args.project_name) / "02_screening" / "screened_responses.csv"),
                "--output",
                str(project_dir(args.project_name) / "02_screening" / "duplicate_responses.csv"),
                "--log",
                str(project_log_path(args.project_name, "screening.log")),
            ],
        )
        return

    if args.command == "validate-duplicates":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_duplicate_responses.py",
            script_args=["--input", str(project_dir(args.project_name) / "02_screening" / "duplicate_responses.csv")],
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
        if args.prepare_only:
            command_args.append("--prepare-only")
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="embeddings.py",
            script_args=command_args,
        )
        return

    if args.command == "validate-embedding-requests":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_embedding_requests.py",
            script_args=["--input", str(question_dir(args.project_name, args.question_id) / "03_embeddings" / "embedding_requests.csv")],
        )
        return

    if args.command == "validate-embedding-metadata":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_embedding_metadata.py",
            script_args=[
                "--input",
                str(question_dir(args.project_name, args.question_id) / "03_embeddings" / "embedding_metadata.json"),
                "--screened",
                str(project_dir(args.project_name) / "02_screening" / "screened_responses.csv"),
                "--requests",
                str(question_dir(args.project_name, args.question_id) / "03_embeddings" / "embedding_requests.csv"),
                "--embeddings",
                str(question_dir(args.project_name, args.question_id) / "03_embeddings" / "embeddings.npy"),
                "--failures",
                str(question_dir(args.project_name, args.question_id) / "03_embeddings" / "embedding_failures.csv"),
            ],
        )
        return

    if args.command == "validate-embedding-failures":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_embedding_failures.py",
            script_args=["--input", str(question_dir(args.project_name, args.question_id) / "03_embeddings" / "embedding_failures.csv")],
        )
        return

    if args.command == "validate-embeddings-array":
        embedding_dir = question_dir(args.project_name, args.question_id) / "03_embeddings"
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_embeddings_array.py",
            script_args=[
                "--input",
                str(embedding_dir / "embeddings.npy"),
                "--requests",
                str(embedding_dir / "embedding_requests.csv"),
                "--metadata",
                str(embedding_dir / "embedding_metadata.json"),
            ],
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
                str(current_question_dir / "03_embeddings" / "embedding_requests.csv"),
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

    if args.command == "validate-clusters":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_clusters.py",
            script_args=["--input", str(question_dir(args.project_name, args.question_id) / "04_clustering" / "clusters.csv")],
        )
        return

    if args.command == "validate-cluster-summary":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_cluster_summary.py",
            script_args=[
                "--input",
                str(question_dir(args.project_name, args.question_id) / "04_clustering" / "cluster_summary.csv"),
                "--clusters",
                str(question_dir(args.project_name, args.question_id) / "04_clustering" / "clusters.csv"),
            ],
        )
        return

    if args.command == "validate-clustering-metadata":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_clustering_metadata.py",
            script_args=[
                "--input",
                str(question_dir(args.project_name, args.question_id) / "04_clustering" / "clustering_metadata.json"),
                "--requests",
                str(question_dir(args.project_name, args.question_id) / "03_embeddings" / "embedding_requests.csv"),
                "--embeddings",
                str(question_dir(args.project_name, args.question_id) / "03_embeddings" / "embeddings.npy"),
                "--clusters",
                str(question_dir(args.project_name, args.question_id) / "04_clustering" / "clusters.csv"),
                "--summary",
                str(question_dir(args.project_name, args.question_id) / "04_clustering" / "cluster_summary.csv"),
            ],
        )
        return

    if args.command == "scaffold-category-master":
        current_question_dir = question_dir(args.project_name, args.question_id)
        command_args = [
            "--input",
            str(current_question_dir / "04_clustering" / "cluster_summary.csv"),
            "--output",
            str(current_question_dir / "05_classification" / "category_master.csv"),
            "--category-prefix",
            args.category_prefix,
            "--log",
            str(question_stage_log_path(args.project_name, args.question_id, "05_classification", "classification.log")),
        ]
        if args.include_outlier:
            command_args.append("--include-outlier")
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="scaffold_category_master.py",
            script_args=command_args,
        )
        return

    if args.command == "category-conflicts":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="category_master_conflicts.py",
            script_args=[
                "--input",
                str(question_dir(args.project_name, args.question_id) / "05_classification" / "category_master.csv"),
                "--output",
                str(question_dir(args.project_name, args.question_id) / "05_classification" / "category_conflicts.csv"),
                "--log",
                str(question_stage_log_path(args.project_name, args.question_id, "05_classification", "classification.log")),
            ],
        )
        return

    if args.command == "validate-category-master":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_category_master.py",
            script_args=["--input", str(question_dir(args.project_name, args.question_id) / "05_classification" / "category_master.csv")],
        )
        return

    if args.command == "validate-override-rules":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_override_rules.py",
            script_args=["--input", str(question_dir(args.project_name, args.question_id) / "05_classification" / "manual_override_rules.csv")],
        )
        return

    if args.command == "validate-category-conflicts":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_category_conflicts.py",
            script_args=["--input", str(question_dir(args.project_name, args.question_id) / "05_classification" / "category_conflicts.csv")],
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

    if args.command == "validate-final-labels":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_final_labels.py",
            script_args=["--input", str(question_dir(args.project_name, args.question_id) / "05_classification" / "final_labels.csv")],
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
                "--duplicates",
                str(project_dir(args.project_name) / "02_screening" / "duplicate_responses.csv"),
                "--log",
                str(question_stage_log_path(args.project_name, args.question_id, "06_review", "review.log")),
            ],
        )
        return

    if args.command == "review-summary":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="review_summary.py",
            script_args=[
                "--input",
                str(question_dir(args.project_name, args.question_id) / "06_review" / "review_log.csv"),
                "--output",
                str(question_dir(args.project_name, args.question_id) / "06_review" / "review_summary.csv"),
                "--log",
                str(question_stage_log_path(args.project_name, args.question_id, "06_review", "review.log")),
            ],
        )
        return

    if args.command == "review-priorities":
        current_question_dir = question_dir(args.project_name, args.question_id)
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="category_review_priorities.py",
            script_args=[
                "--review-summary",
                str(current_question_dir / "06_review" / "review_summary.csv"),
                "--category-conflicts",
                str(current_question_dir / "05_classification" / "category_conflicts.csv"),
                "--output",
                str(current_question_dir / "06_review" / "category_review_priorities.csv"),
                "--log",
                str(question_stage_log_path(args.project_name, args.question_id, "06_review", "review.log")),
            ],
        )
        return

    if args.command == "review-samples":
        current_question_dir = question_dir(args.project_name, args.question_id)
        command_args = [
            "--input",
            str(current_question_dir / "06_review" / "review_log.csv"),
            "--output",
            str(current_question_dir / "06_review" / "review_samples.csv"),
            "--medium-per-category",
            str(args.medium_per_category),
            "--log",
            str(question_stage_log_path(args.project_name, args.question_id, "06_review", "review.log")),
        ]
        if args.include_low:
            command_args.append("--include-low")
        if args.priority_trigger is not None:
            command_args.extend(["--priority-trigger", args.priority_trigger])
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="review_samples.py",
            script_args=command_args,
        )
        return

    if args.command == "review-corrections":
        current_question_dir = question_dir(args.project_name, args.question_id)
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="review_corrections.py",
            script_args=[
                "--input",
                str(current_question_dir / "06_review" / "review_log.csv"),
                "--output",
                str(current_question_dir / "06_review" / "review_corrections.csv"),
                "--log",
                str(question_stage_log_path(args.project_name, args.question_id, "06_review", "review.log")),
            ],
        )
        return

    if args.command == "override-candidates":
        current_question_dir = question_dir(args.project_name, args.question_id)
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="review_override_candidates.py",
            script_args=[
                "--input",
                str(current_question_dir / "06_review" / "review_corrections.csv"),
                "--output",
                str(current_question_dir / "05_classification" / "manual_override_candidates.csv"),
                "--log",
                str(question_stage_log_path(args.project_name, args.question_id, "05_classification", "classification.log")),
            ],
        )
        return

    if args.command == "promote-override-candidates":
        current_question_dir = question_dir(args.project_name, args.question_id)
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="promote_override_candidates.py",
            script_args=[
                "--input",
                str(current_question_dir / "05_classification" / "manual_override_candidates.csv"),
                "--output",
                str(current_question_dir / "05_classification" / "manual_override_rules.csv"),
                "--log",
                str(question_stage_log_path(args.project_name, args.question_id, "05_classification", "classification.log")),
            ],
        )
        return

    if args.command == "override-rule-hits":
        current_question_dir = question_dir(args.project_name, args.question_id)
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="override_rule_hits.py",
            script_args=[
                "--input",
                str(project_dir(args.project_name) / "02_screening" / "screened_responses.csv"),
                "--question-id",
                args.question_id,
                "--override-rules",
                str(current_question_dir / "05_classification" / "manual_override_rules.csv"),
                "--output",
                str(current_question_dir / "05_classification" / "override_rule_hits.csv"),
                "--log",
                str(question_stage_log_path(args.project_name, args.question_id, "05_classification", "classification.log")),
            ],
        )
        return

    if args.command == "override-rule-summary":
        current_question_dir = question_dir(args.project_name, args.question_id)
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="override_rule_summary.py",
            script_args=[
                "--input",
                str(current_question_dir / "05_classification" / "override_rule_hits.csv"),
                "--output",
                str(current_question_dir / "05_classification" / "override_rule_summary.csv"),
                "--log",
                str(question_stage_log_path(args.project_name, args.question_id, "05_classification", "classification.log")),
            ],
        )
        return

    if args.command == "validate-review-log":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_review_log.py",
            script_args=["--input", str(question_dir(args.project_name, args.question_id) / "06_review" / "review_log.csv")],
        )
        return

    if args.command == "validate-review-summary":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_review_summary.py",
            script_args=["--input", str(question_dir(args.project_name, args.question_id) / "06_review" / "review_summary.csv")],
        )
        return

    if args.command == "validate-review-priorities":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_category_review_priorities.py",
            script_args=["--input", str(question_dir(args.project_name, args.question_id) / "06_review" / "category_review_priorities.csv")],
        )
        return

    if args.command == "validate-review-samples":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_review_samples.py",
            script_args=["--input", str(question_dir(args.project_name, args.question_id) / "06_review" / "review_samples.csv")],
        )
        return

    if args.command == "validate-review-corrections":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_review_corrections.py",
            script_args=["--input", str(question_dir(args.project_name, args.question_id) / "06_review" / "review_corrections.csv")],
        )
        return

    if args.command == "validate-override-candidates":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_override_candidates.py",
            script_args=["--input", str(question_dir(args.project_name, args.question_id) / "05_classification" / "manual_override_candidates.csv")],
        )
        return

    if args.command == "validate-override-rule-hits":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_override_rule_hits.py",
            script_args=["--input", str(question_dir(args.project_name, args.question_id) / "05_classification" / "override_rule_hits.csv")],
        )
        return

    if args.command == "validate-override-rule-summary":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_override_rule_summary.py",
            script_args=["--input", str(question_dir(args.project_name, args.question_id) / "05_classification" / "override_rule_summary.csv")],
        )
        return

    if args.command == "validate-question":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_question_artifacts.py",
            script_args=["--question-dir", str(question_dir(args.project_name, args.question_id))],
        )
        return

    if args.command == "validate-project":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_project_artifacts.py",
            script_args=["--project-dir", str(project_dir(args.project_name))],
        )
        return

    if args.command == "validate-log":
        run_pipeline_command(
            project_name=args.project_name,
            command_name=args.command,
            script_name="validate_jsonl_log.py",
            script_args=["--input", str(project_dir(args.project_name) / "99_logs" / args.log_name)],
        )
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
