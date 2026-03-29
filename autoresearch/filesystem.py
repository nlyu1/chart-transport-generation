from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import TypeVar

from src.config.base import BaseConfig

from autoresearch.contracts import (
    ArtifactRef,
    MetastudyPlan,
    MetastudyState,
    ReviewerVerdict,
    StudyPlan,
    StudyState,
    StudySynthesis,
    SubstudyResult,
    SubstudyState,
)


ModelT = TypeVar("ModelT", bound=BaseConfig)


class AutoresearchFilesystem:
    def __init__(
        self,
        *,
        repo_root: Path,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.metastudies_root = self.repo_root / "artifacts" / "metastudies"
        self.studies_root = self.repo_root / "artifacts" / "studies"

    def ensure_roots(self) -> None:
        self.metastudies_root.mkdir(parents=True, exist_ok=True)
        self.studies_root.mkdir(parents=True, exist_ok=True)

    def metastudy_root(self, *, metastudy_slug: str) -> Path:
        return self.metastudies_root / metastudy_slug

    def study_root(self, *, study_slug: str) -> Path:
        return self.studies_root / study_slug

    def substudy_root(self, *, study_slug: str, substudy_slug: str) -> Path:
        return self.study_root(study_slug=study_slug) / "substudies" / substudy_slug

    def initialize_metastudy(
        self,
        *,
        metastudy_slug: str,
        title: str,
        question_markdown: str,
        state: MetastudyState,
    ) -> Path:
        self.ensure_roots()
        root = self.metastudy_root(metastudy_slug=metastudy_slug)
        if root.exists():
            raise FileExistsError(f"Metastudy already exists: {root}")
        root.mkdir(parents=True)
        self.write_markdown(
            path=root / "question.md",
            content=question_markdown.rstrip() + "\n",
        )
        self.write_markdown(
            path=root / "plan.md",
            content=self.render_metastudy_plan_stub(title=title),
        )
        self.write_json(path=root / "state.json", model=state)
        self.write_markdown(
            path=root / "journal.md",
            content="# Journal\n\n- Metastudy scaffold created.\n",
        )
        self.write_markdown(
            path=root / "dashboard.md",
            content=self.render_dashboard_stub(title=title),
        )
        return root

    def initialize_study(
        self,
        *,
        study_slug: str,
        title: str,
        question_markdown: str,
        state: StudyState,
    ) -> Path:
        self.ensure_roots()
        root = self.study_root(study_slug=study_slug)
        if root.exists():
            raise FileExistsError(f"Study already exists: {root}")
        (root / "substudies").mkdir(parents=True)
        (root / "scripts").mkdir()
        (root / "data").mkdir()
        (root / "figures").mkdir()
        self.write_markdown(
            path=root / "question.md",
            content=question_markdown.rstrip() + "\n",
        )
        self.write_markdown(
            path=root / "plan.md",
            content=self.render_study_plan_stub(title=title),
        )
        self.write_markdown(
            path=root / "report.md",
            content=self.render_study_report_stub(title=title),
        )
        self.write_json(path=root / "state.json", model=state)
        return root

    def initialize_substudy(
        self,
        *,
        study_slug: str,
        substudy_slug: str,
        title: str,
        task_markdown: str,
        state: SubstudyState,
    ) -> Path:
        root = self.substudy_root(study_slug=study_slug, substudy_slug=substudy_slug)
        if root.exists():
            raise FileExistsError(f"Substudy already exists: {root}")
        (root / "scripts").mkdir(parents=True)
        (root / "data").mkdir()
        (root / "figures").mkdir()
        self.write_markdown(
            path=root / "task.md",
            content=task_markdown.rstrip() + "\n",
        )
        self.write_markdown(
            path=root / "notes.md",
            content=f"# {title}\n\n## Notes\n\n",
        )
        self.write_markdown(
            path=root / "summary.md",
            content=f"# {title}\n\n## Summary\n\nPending execution.\n",
        )
        self.write_json(path=root / "state.json", model=state)
        return root

    def write_metastudy_plan(
        self,
        *,
        metastudy_slug: str,
        plan: MetastudyPlan,
    ) -> None:
        root = self.metastudy_root(metastudy_slug=metastudy_slug)
        self.write_json(path=root / "plan.json", model=plan)
        self.write_markdown(path=root / "plan.md", content=self.render_metastudy_plan(plan))

    def write_metastudy_review(
        self,
        *,
        metastudy_slug: str,
        verdict: ReviewerVerdict,
    ) -> None:
        root = self.metastudy_root(metastudy_slug=metastudy_slug)
        self.write_json(path=root / "last_review.json", model=verdict)
        self.append_journal(
            path=root / "journal.md",
            line=f"- Metastudy review `{verdict.verdict}`: {verdict.summary}",
        )

    def write_study_plan(
        self,
        *,
        study_slug: str,
        plan: StudyPlan,
    ) -> None:
        root = self.study_root(study_slug=study_slug)
        self.write_json(path=root / "plan.json", model=plan)
        self.write_markdown(path=root / "plan.md", content=self.render_study_plan(plan))

    def write_study_review(
        self,
        *,
        study_slug: str,
        verdict: ReviewerVerdict,
    ) -> None:
        root = self.study_root(study_slug=study_slug)
        self.write_json(path=root / "last_review.json", model=verdict)

    def write_substudy_result(
        self,
        *,
        study_slug: str,
        substudy_slug: str,
        result: SubstudyResult,
    ) -> None:
        root = self.substudy_root(study_slug=study_slug, substudy_slug=substudy_slug)
        self.write_json(path=root / "result.json", model=result)
        self.write_markdown(path=root / "summary.md", content=self.render_substudy_summary(result))

    def write_substudy_review(
        self,
        *,
        study_slug: str,
        substudy_slug: str,
        verdict: ReviewerVerdict,
    ) -> None:
        root = self.substudy_root(study_slug=study_slug, substudy_slug=substudy_slug)
        self.write_json(path=root / "last_review.json", model=verdict)

    def write_study_synthesis(
        self,
        *,
        study_slug: str,
        synthesis: StudySynthesis,
    ) -> None:
        root = self.study_root(study_slug=study_slug)
        self.write_json(path=root / "synthesis.json", model=synthesis)

    def write_json(self, *, path: Path, model: BaseConfig) -> None:
        path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")

    def read_json(self, *, path: Path, model_type: type[ModelT]) -> ModelT:
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))

    def write_markdown(self, *, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")

    def append_journal(self, *, path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as journal_file:
            journal_file.write(line.rstrip() + "\n")

    def render_metastudy_plan_stub(self, *, title: str) -> str:
        return dedent(
            f"""\
            # {title}

            ## Problem

            Pending metastudy planning.

            ## Active Hypotheses

            Pending review.

            ## Study Backlog

            Pending review.

            ## Completion Criteria

            Pending review.

            ## Next Decision

            Draft the first metastudy plan.
            """
        )

    def render_dashboard_stub(self, *, title: str) -> str:
        return dedent(
            f"""\
            # {title}

            ## Status

            - Lifecycle: drafting
            - Active study: none

            ## Current Focus

            Pending first metastudy plan.
            """
        )

    def render_study_plan_stub(self, *, title: str) -> str:
        return dedent(
            f"""\
            # {title}

            ## Question

            Pending study planning.

            ## Hypotheses

            Pending study planning.

            ## Substudies

            Pending study planning.

            ## Review Trigger

            Submit the study plan for review before any substudy runs.
            """
        )

    def render_study_report_stub(self, *, title: str) -> str:
        return dedent(
            f"""\
            # {title}

            ## Question

            State the exact study question.

            ## Short Answer

            Pending study execution.

            ## Scope

            - Pending study execution.

            ## Hypotheses

            1. Pending study execution.

            ## Findings

            ### Finding 1

            - Observation.
            - Interpretation.
            - Evidence: `data/...`, `figures/...`, `scripts/...`

            ## Isolating Experiments

            ### Experiment 1

            - Hypothesis tested:
            - Isolation strategy:
            - Inputs:
            - Procedure:
            - Result:
            - Decision:

            ## Evidence Index

            - `figures/...`
            - `data/...`
            - `scripts/...`

            ## Reproduction

            List the commands that rebuild the study-local artifacts.

            ## Open Questions

            State the next isolating experiment.
            """
        )

    def render_metastudy_plan(self, plan: MetastudyPlan) -> str:
        backlog_lines = [
            f"- `{study.study_slug}`: {study.title} — {study.question}"
            for study in plan.study_backlog
        ]
        hypothesis_lines = [
            f"- `{hypothesis.hypothesis_id}`: {hypothesis.statement}"
            for hypothesis in plan.active_hypotheses
        ]
        completion_lines = [f"- {criterion}" for criterion in plan.completion_criteria]
        theory_lines = [f"- {anchor}" for anchor in plan.theory_anchors]
        return dedent(
            f"""\
            # {plan.title}

            ## Problem

            {plan.problem_statement}

            ## Theory Anchors

            {self._render_lines(theory_lines)}

            ## Active Hypotheses

            {self._render_lines(hypothesis_lines)}

            ## Study Backlog

            {self._render_lines(backlog_lines)}

            ## Completion Criteria

            {self._render_lines(completion_lines)}

            ## Next Decision

            {plan.next_decision}
            """
        )

    def render_study_plan(self, plan: StudyPlan) -> str:
        hypothesis_lines = [
            f"- `{hypothesis.hypothesis_id}`: {hypothesis.statement}"
            for hypothesis in plan.hypotheses
        ]
        substudy_lines = [
            f"- `{task.substudy_slug}`: {task.title} — {task.question}"
            for task in plan.substudies
        ]
        theory_lines = [f"- {anchor}" for anchor in plan.theory_anchors]
        synthesis_lines = [
            f"- {question}" for question in plan.expected_synthesis_questions
        ]
        return dedent(
            f"""\
            # {plan.title}

            ## Question

            {plan.question}

            ## Theory Anchors

            {self._render_lines(theory_lines)}

            ## Hypotheses

            {self._render_lines(hypothesis_lines)}

            ## Substudies

            {self._render_lines(substudy_lines)}

            ## Expected Synthesis Questions

            {self._render_lines(synthesis_lines)}

            ## Review Trigger

            {plan.next_review_trigger}
            """
        )

    def render_substudy_summary(self, result: SubstudyResult) -> str:
        fact_lines = [f"- {fact}" for fact in result.observed_facts]
        interpretation_lines = [f"- {item}" for item in result.interpretations]
        open_question_lines = [f"- {item}" for item in result.open_questions]
        artifact_lines = [self._render_artifact_ref(ref=artifact) for artifact in result.artifacts]
        return dedent(
            f"""\
            # {result.substudy_id}

            ## Status

            {result.status}

            ## Summary

            {result.summary}

            ## Observed Facts

            {self._render_lines(fact_lines)}

            ## Interpretations

            {self._render_lines(interpretation_lines)}

            ## Artifacts

            {self._render_lines(artifact_lines)}

            ## Open Questions

            {self._render_lines(open_question_lines)}
            """
        )

    def _render_artifact_ref(self, *, ref: ArtifactRef) -> str:
        return f"- `{ref.path}`: {ref.description}"

    def _render_lines(self, lines: list[str]) -> str:
        if lines:
            return "\n".join(lines)
        return "- None yet."


__all__ = ["AutoresearchFilesystem"]
