from __future__ import annotations

import argparse
import re
from pathlib import Path

from common import append_jsonl, utc_now_iso


REQUIRED_PATTERNS = {
    "title": re.compile(r"^# raw_to_processed mapping$", re.MULTILINE),
    "source_file": re.compile(r"^- source_file:\s*`?.+`?$", re.MULTILINE),
    "output_file": re.compile(r"^- output_file:\s*`?.+`?$", re.MULTILINE),
    "row_count": re.compile(r"^- row_count:\s*\d+$", re.MULTILINE),
    "column_mapping": re.compile(r"^- column_mapping:\s*$", re.MULTILINE),
}


def run_validations(content: str) -> list[str]:
    errors: list[str] = []
    for label, pattern in REQUIRED_PATTERNS.items():
        if pattern.search(content) is None:
            errors.append(f"Missing or invalid section: {label}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 99_logs/raw_to_processed_mapping.md")
    parser.add_argument("--input", required=True, type=Path, help="Path to raw_to_processed_mapping.md")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append validation results as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    content = args.input.read_text(encoding="utf-8")
    errors = run_validations(content)
    log_payload = {
        "event": "validate_raw_to_processed_mapping",
        "input": str(args.input),
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
