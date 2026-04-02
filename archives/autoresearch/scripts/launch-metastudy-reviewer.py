#!/usr/bin/env python3
"""Launch the metastudy-reviewer Codex agent.

Called by: metastudy-executor (after all studies complete).

Usage:
    python autoresearch/launch-metastudy-reviewer.py <metastudy-path> --log <log-path>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launch_common import launch

ROLE = "metastudy-reviewer"


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the metastudy-reviewer Codex agent.")
    parser.add_argument("metastudy_path", type=Path, help="Path to the metastudy directory")
    parser.add_argument("--log", type=Path, required=True, help="Path to the metastudy run.log")
    args = parser.parse_args()

    return launch(ROLE, args.metastudy_path.resolve(), args.log.resolve())


if __name__ == "__main__":
    sys.exit(main())
