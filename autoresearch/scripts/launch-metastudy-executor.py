#!/usr/bin/env python3
"""User-facing entrypoint: launch the metastudy-executor for a given metastudy.

Usage:
    uv run python autoresearch/scripts/launch-metastudy-executor.py metastudies/<name>

The metastudy directory must already contain objective.md.
The log is written to <metastudy-path>/run.log.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launch_common import launch

ROLE = "metastudy-executor"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch the metastudy-executor Codex agent.",
        epilog="Example: uv run python autoresearch/scripts/launch-metastudy-executor.py metastudies/multimodal-gaussian-baseline",
    )
    parser.add_argument("metastudy_path", type=Path, help="Path to the metastudy directory")
    args = parser.parse_args()

    metastudy_path = args.metastudy_path.resolve()
    if not metastudy_path.is_dir():
        print(f"ERROR: {metastudy_path} is not a directory", file=sys.stderr)
        return 1
    if not (metastudy_path / "objective.md").exists():
        print(f"ERROR: {metastudy_path}/objective.md not found", file=sys.stderr)
        return 1

    log_path = metastudy_path / "run.log"
    return launch(ROLE, metastudy_path, log_path)


if __name__ == "__main__":
    sys.exit(main())
