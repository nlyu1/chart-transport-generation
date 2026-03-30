#!/usr/bin/env python3
"""Resolve a metastudy target and print planner context for the current session."""

from __future__ import annotations

import argparse
import json
import shlex
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


def build_planner_context(*, metastudy_path: Path) -> dict[str, object]:
    objective_path = metastudy_path / "objective.md"
    plan_path = metastudy_path / "plan.md"
    state_path = metastudy_path / "state.md"
    return {
        "repo_root": str(REPO),
        "metastudy_path": str(metastudy_path),
        "metastudy_repo_relative": repo_relative(metastudy_path),
        "objective_path": str(objective_path),
        "objective_repo_relative": repo_relative(objective_path),
        "plan_path": str(plan_path),
        "plan_repo_relative": repo_relative(plan_path),
        "plan_exists": plan_path.is_file(),
        "state_path": str(state_path),
        "state_repo_relative": repo_relative(state_path),
        "state_exists": state_path.is_file(),
        "planner_prompt_path": str(REPO / "autoresearch/prompts/metastudy-planner.md"),
        "theory_path": str(REPO / "theory/proposal.typ"),
        "shared_metastudy_agents_path": str(REPO / "metastudies/AGENTS.md"),
        "repo_agents_path": str(REPO / "AGENTS.md"),
        "legacy_nested_command": [
            "uv",
            "run",
            "python",
            "autoresearch/scripts/launch-metastudy-planner.py",
            repo_relative(metastudy_path),
        ],
    }


def render_text_context(*, context: dict[str, object]) -> str:
    legacy_nested_command = shlex.join(context["legacy_nested_command"])
    return "\n".join(
        [
            "Metastudy planner context",
            f"repo: {context['repo_root']}",
            f"metastudy: {context['metastudy_repo_relative']}",
            f"objective: {context['objective_repo_relative']}",
            f"plan: {context['plan_repo_relative']} (exists: {str(context['plan_exists']).lower()})",
            f"state: {context['state_repo_relative']} (exists: {str(context['state_exists']).lower()})",
            "",
            "Use the current Codex session as the planner.",
            "Read these first:",
            f"- {Path(str(context['planner_prompt_path'])).relative_to(REPO)}",
            f"- {Path(str(context['shared_metastudy_agents_path'])).relative_to(REPO)}",
            f"- {Path(str(context['repo_agents_path'])).relative_to(REPO)}",
            f"- {Path(str(context['objective_path'])).relative_to(REPO)}",
            f"- {Path(str(context['theory_path'])).relative_to(REPO)}",
            "",
            "Do not launch a nested interactive Codex session unless the user explicitly requests the legacy behavior.",
            f"Legacy nested command: {legacy_nested_command}",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve a metastudy target and print planner context.",
    )
    parser.add_argument(
        "metastudy",
        help="Metastudy name or path. Bare names resolve under metastudies/.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format for the validated planner context.",
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

    context = build_planner_context(metastudy_path=metastudy_path)
    if args.format == "json":
        print(json.dumps(context, indent=2, sort_keys=True))
        return 0

    print(render_text_context(context=context))
    return 0


if __name__ == "__main__":
    sys.exit(main())
