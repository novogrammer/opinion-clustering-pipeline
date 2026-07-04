from __future__ import annotations

import argparse
from pathlib import Path

from common import copy_template, validate_identifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a new project directory from templates/project")
    parser.add_argument("--project-name", required=True, help="Project directory name")
    parser.add_argument(
        "--projects-dir",
        type=Path,
        default=Path("projects"),
        help="Base directory for project outputs",
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=Path("templates/project"),
        help="Project template directory",
    )
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        validate_identifier(args.project_name, "project_name")
        project_dir = args.projects_dir / args.project_name
        copy_template(args.template_dir, project_dir)
    except (FileExistsError, FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
