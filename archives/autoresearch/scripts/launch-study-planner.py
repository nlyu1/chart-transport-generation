#!/usr/bin/env python3
"""Launch the study-planner Codex agent.

Called by: study-executor (when plan.md is missing).

Usage:
    python autoresearch/launch-study-planner.py <study-path> --log <log-path>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launch_common import launch

ROLE = "study-planner"


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the study-planner Codex agent.")
    parser.add_argument("study_path", type=Path, help="Path to the study directory")
    parser.add_argument("--log", type=Path, required=True, help="Path to the metastudy run.log")
    args = parser.parse_args()

    return launch(ROLE, args.study_path.resolve(), args.log.resolve())


if __name__ == "__main__":
    sys.exit(main())
