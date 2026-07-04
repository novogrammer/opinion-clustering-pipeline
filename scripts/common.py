from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_RESPONSE_COLUMNS = [
    "response_id",
    "question_id",
    "question_text",
    "answer_text",
]

REQUIRED_LOG_KEYS = [
    "event",
    "created_at",
]

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_template(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("*.sample.*", "__pycache__", "*.pyc", ".DS_Store"),
    )


def validate_identifier(value: str, label: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must contain only ASCII letters, digits, and underscores: {value}")
    return value


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_parent_dir(path)
    df.to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)


def write_json(payload: dict, path: Path) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(payload: dict, path: Path) -> None:
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class ProjectPaths:
    project_dir: Path
    raw_dir: Path
    processed_dir: Path
    screening_dir: Path
    questions_dir: Path
    logs_dir: Path


def build_project_paths(project_dir: Path) -> ProjectPaths:
    return ProjectPaths(
        project_dir=project_dir,
        raw_dir=project_dir / "00_raw",
        processed_dir=project_dir / "01_processed",
        screening_dir=project_dir / "02_screening",
        questions_dir=project_dir / "questions",
        logs_dir=project_dir / "99_logs",
    )


def validate_project_scaffold(project_dir: Path) -> None:
    paths = build_project_paths(project_dir)
    if not paths.project_dir.exists():
        raise FileNotFoundError(f"Project directory does not exist: {paths.project_dir}")
    if not paths.project_dir.is_dir():
        raise NotADirectoryError(f"Project path is not a directory: {paths.project_dir}")

    required_dirs = [
        paths.raw_dir,
        paths.processed_dir,
        paths.screening_dir,
        paths.questions_dir,
        paths.logs_dir,
    ]
    missing_dirs = [path for path in required_dirs if not path.is_dir()]
    if missing_dirs:
        joined = ", ".join(str(path) for path in missing_dirs)
        raise FileNotFoundError(f"Project scaffold is incomplete: {joined}")


def validate_question_scaffold(project_dir: Path, question_id: str) -> None:
    validate_project_scaffold(project_dir)
    question_root = project_dir / "questions" / question_id
    if not question_root.exists():
        raise FileNotFoundError(f"Question directory does not exist: {question_root}")
    if not question_root.is_dir():
        raise NotADirectoryError(f"Question path is not a directory: {question_root}")

    required_dirs = [
        question_root / "03_embeddings",
        question_root / "04_clustering",
        question_root / "05_classification",
        question_root / "06_review",
    ]
    missing_dirs = [path for path in required_dirs if not path.is_dir()]
    if missing_dirs:
        joined = ", ".join(str(path) for path in missing_dirs)
        raise FileNotFoundError(f"Question scaffold is incomplete: {joined}")
