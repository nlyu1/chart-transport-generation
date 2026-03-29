from __future__ import annotations

import argparse
from pathlib import Path

from autoresearch.contracts import MetastudyState, StudyState, SubstudyState
from autoresearch.filesystem import AutoresearchFilesystem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autoresearch scaffold CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    metastudy_parser = subparsers.add_parser(
        "init-metastudy",
        help="Create a metastudy scaffold under artifacts/metastudies.",
    )
    metastudy_parser.add_argument("--slug", required=True)
    metastudy_parser.add_argument("--title", required=True)
    metastudy_parser.add_argument("--question-file", type=Path, required=True)

    study_parser = subparsers.add_parser(
        "init-study",
        help="Create a study scaffold under artifacts/studies.",
    )
    study_parser.add_argument("--metastudy-slug", required=True)
    study_parser.add_argument("--study-id", required=True)
    study_parser.add_argument("--study-slug", required=True)
    study_parser.add_argument("--title", required=True)
    study_parser.add_argument("--question-file", type=Path, required=True)

    substudy_parser = subparsers.add_parser(
        "init-substudy",
        help="Create a substudy scaffold under artifacts/studies/<study>/substudies.",
    )
    substudy_parser.add_argument("--study-slug", required=True)
    substudy_parser.add_argument("--substudy-id", required=True)
    substudy_parser.add_argument("--substudy-slug", required=True)
    substudy_parser.add_argument("--title", required=True)
    substudy_parser.add_argument("--task-file", type=Path, required=True)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path.cwd()
    filesystem = AutoresearchFilesystem(repo_root=repo_root)

    if args.command == "init-metastudy":
        question_markdown = args.question_file.read_text(encoding="utf-8")
        filesystem.initialize_metastudy(
            metastudy_slug=args.slug,
            title=args.title,
            question_markdown=question_markdown,
            state=MetastudyState(
                metastudy_id=args.slug,
                metastudy_slug=args.slug,
                status="drafting",
            ),
        )
        return

    if args.command == "init-study":
        question_markdown = args.question_file.read_text(encoding="utf-8")
        filesystem.initialize_study(
            study_slug=args.study_slug,
            title=args.title,
            question_markdown=question_markdown,
            state=StudyState(
                study_id=args.study_id,
                study_slug=args.study_slug,
                metastudy_slug=args.metastudy_slug,
                status="drafting",
            ),
        )
        return

    if args.command == "init-substudy":
        task_markdown = args.task_file.read_text(encoding="utf-8")
        filesystem.initialize_substudy(
            study_slug=args.study_slug,
            substudy_slug=args.substudy_slug,
            title=args.title,
            task_markdown=task_markdown,
            state=SubstudyState(
                substudy_id=args.substudy_id,
                substudy_slug=args.substudy_slug,
                study_slug=args.study_slug,
                status="drafting",
            ),
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
