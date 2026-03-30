#!/usr/bin/env python3
"""Resolve a metastudy target and launch the interactive autoresearch planner."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
METASTUDIES = REPO / "metastudies"


def resolve_metastudy(raw_target: str) -> Path:
    candidate = Path(raw_target)
    if candidate.is_absolute():
        return candidate.resolve()
    if raw_target.startswith(".") or "/" in raw_target:
        return (REPO / candidate).resolve()
    return (METASTUDIES / raw_target).resolve()


def repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def build_command(metastudy_path: Path) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "autoresearch/scripts/launch-metastudy-planner.py",
        repo_relative(metastudy_path),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch the interactive autoresearch metastudy planner.",
    )
    parser.add_argument(
        "metastudy",
        help="Metastudy name or path. Bare names resolve under metastudies/.",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the resolved planner command without executing it.",
    )
    args = parser.parse_args()

    metastudy_path = resolve_metastudy(args.metastudy)
    if not metastudy_path.is_dir():
        print(f"ERROR: metastudy directory not found: {metastudy_path}", file=sys.stderr)
        return 1

    objective_path = metastudy_path / "objective.md"
    if not objective_path.is_file():
        print(f"ERROR: objective.md not found: {objective_path}", file=sys.stderr)
        return 1

    command = build_command(metastudy_path)
    if args.print_command:
        print(shlex.join(command))
        return 0

    return subprocess.run(command, cwd=REPO).returncode


if __name__ == "__main__":
    sys.exit(main())
