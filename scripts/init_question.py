from __future__ import annotations

import argparse
from pathlib import Path

from common import copy_template, validate_identifier, validate_project_scaffold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a new question directory from templates/question")
    parser.add_argument("--project-dir", required=True, type=Path, help="Path to the project directory")
    parser.add_argument("--question-id", required=True, help="Question directory name")
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=Path("templates/question"),
        help="Question template directory",
    )
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        validate_identifier(args.question_id, "question_id")
        validate_project_scaffold(args.project_dir)
        question_dir = args.project_dir / "questions" / args.question_id
        copy_template(args.template_dir, question_dir)
    except (FileExistsError, FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
