from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import REQUIRED_LOG_KEYS, append_jsonl, utc_now_iso


def run_validations(lines: list[str]) -> list[str]:
    errors: list[str] = []
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped == "":
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid JSON at line {idx}: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"Line {idx} is not a JSON object")
            continue
        for key in REQUIRED_LOG_KEYS:
            if key not in payload:
                errors.append(f"Missing key '{key}' at line {idx}")
        if "event" in payload and str(payload["event"]).strip() == "":
            errors.append(f"Blank event at line {idx}")
        if "created_at" in payload and str(payload["created_at"]).strip() == "":
            errors.append(f"Blank created_at at line {idx}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate JSONL log files used by pipeline scripts")
    parser.add_argument("--input", required=True, type=Path, help="Path to .log JSONL file")
    parser.add_argument("--log", type=Path, default=None, help="Optional path to append validation results as JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lines = args.input.read_text(encoding="utf-8").splitlines()
    errors = run_validations(lines)
    log_payload = {
        "event": "validate_jsonl_log",
        "input": str(args.input),
        "line_count": len(lines),
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
