#!/usr/bin/env python3
"""Launch the substudy-executor Codex agent.

Called by: study-executor (once per substudy in the study plan).
This agent is the only one that actually runs training code.

Usage:
    python autoresearch/launch-substudy-executor.py <substudy-path> --log <log-path>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launch_common import launch

ROLE = "substudy-executor"


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the substudy-executor Codex agent.")
    parser.add_argument("substudy_path", type=Path, help="Path to the substudy directory")
    parser.add_argument("--log", type=Path, required=True, help="Path to the metastudy run.log")
    args = parser.parse_args()

    return launch(ROLE, args.substudy_path.resolve(), args.log.resolve())


if __name__ == "__main__":
    sys.exit(main())
