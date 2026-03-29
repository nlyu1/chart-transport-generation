#!/usr/bin/env python3
"""Launch the metastudy-planner as an interactive Codex session.

Run this before launch-metastudy-executor.py. The interactive session lets you
review and steer the study decomposition before execution begins.

Usage:
    uv run python autoresearch/scripts/launch-metastudy-planner.py metastudies/<name>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launch_common import launch

ROLE = "metastudy-planner"


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the metastudy-planner as an interactive Codex session.")
    parser.add_argument("metastudy_path", type=Path, help="Path to the metastudy directory")
    parser.add_argument("--log", type=Path, help="Path to run.log (default: <metastudy-path>/run.log)")
    args = parser.parse_args()

    metastudy_path = args.metastudy_path.resolve()
    log_path = args.log.resolve() if args.log else metastudy_path / "run.log"
    return launch(ROLE, metastudy_path, log_path, interactive=True)


if __name__ == "__main__":
    sys.exit(main())
