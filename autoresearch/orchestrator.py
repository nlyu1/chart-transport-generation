from __future__ import annotations

from pathlib import Path
from typing import Any

from autoresearch.contracts import (
    MetastudyPlan,
    MetastudyState,
    ReviewerVerdict,
    StudyDirective,
    StudyPlan,
    StudyState,
    StudySynthesis,
    SubstudyResult,
    SubstudyState,
    SubstudyTask,
)
from autoresearch.filesystem import AutoresearchFilesystem
from autoresearch.model_client import ModelClient
from autoresearch.prompts import PromptLibrary


class AutoresearchOrchestrator:
    def __init__(
        self,
        *,
        filesystem: AutoresearchFilesystem,
        model_client: ModelClient,
        prompt_library: PromptLibrary,
    ) -> None:
        self.filesystem = filesystem
        self.model_client = model_client
        self.prompt_library = prompt_library

    @classmethod
    def build(
        cls,
        *,
        repo_root: Path,
        model_client: ModelClient,
    ) -> "AutoresearchOrchestrator":
        return cls(
            filesystem=AutoresearchFilesystem(repo_root=repo_root),
            model_client=model_client,
            prompt_library=PromptLibrary.build_default(),
        )

    def bootstrap_metastudy(
        self,
        *,
        metastudy_slug: str,
        title: str,
        question_markdown: str,
    ) -> Path:
        state = MetastudyState(
            metastudy_id=metastudy_slug,
            metastudy_slug=metastudy_slug,
            status="drafting",
        )
        return self.filesystem.initialize_metastudy(
            metastudy_slug=metastudy_slug,
            title=title,
            question_markdown=question_markdown,
            state=state,
        )

    def plan_metastudy(self, *, metastudy_slug: str) -> MetastudyPlan:
        root = self.filesystem.metastudy_root(metastudy_slug=metastudy_slug)
        question_markdown = (root / "question.md").read_text(encoding="utf-8")
        state = self.filesystem.read_json(
            path=root / "state.json",
            model_type=MetastudyState,
        )
        plan = self.model_client.generate_structured(
            role="metastudy_agent",
            system_prompt=self.prompt_library.load(role="metastudy_agent"),
            input_sections={
                "question_markdown": question_markdown,
                "state": state.model_dump(mode="json"),
            },
            output_type=MetastudyPlan,
        )
        self.filesystem.write_metastudy_plan(metastudy_slug=metastudy_slug, plan=plan)
        self.filesystem.write_json(
            path=root / "state.json",
            model=state.model_copy(
                update={
                    "status": "in_review",
                    "plan_revision": state.plan_revision + 1,
                }
            ),
        )
        return plan

    def review_metastudy(self, *, metastudy_slug: str) -> ReviewerVerdict:
        root = self.filesystem.metastudy_root(metastudy_slug=metastudy_slug)
        state = self.filesystem.read_json(
            path=root / "state.json",
            model_type=MetastudyState,
        )
        plan = self.filesystem.read_json(
            path=root / "plan.json",
            model_type=MetastudyPlan,
        )
        verdict = self.model_client.generate_structured(
            role="metastudy_reviewer",
            system_prompt=self.prompt_library.load(role="metastudy_reviewer"),
            input_sections={
                "plan": plan.model_dump(mode="json"),
                "state": state.model_dump(mode="json"),
                "question_markdown": (root / "question.md").read_text(encoding="utf-8"),
            },
            output_type=ReviewerVerdict,
        )
        next_status = self._status_from_verdict(verdict=verdict)
        self.filesystem.write_metastudy_review(
            metastudy_slug=metastudy_slug,
            verdict=verdict,
        )
        self.filesystem.write_json(
            path=root / "state.json",
            model=state.model_copy(
                update={
                    "status": next_status,
                    "last_verdict": verdict.verdict,
                    "pending_study_slugs": [
                        study.study_slug for study in plan.study_backlog
                    ],
                }
            ),
        )
        return verdict

    def bootstrap_study_from_directive(
        self,
        *,
        metastudy_slug: str,
        directive: StudyDirective,
    ) -> Path:
        study_state = StudyState(
            study_id=directive.study_id,
            study_slug=directive.study_slug,
            metastudy_slug=metastudy_slug,
            status="drafting",
        )
        question_markdown = self._render_study_question(directive=directive)
        root = self.filesystem.initialize_study(
            study_slug=directive.study_slug,
            title=directive.title,
            question_markdown=question_markdown,
            state=study_state,
        )
        metastudy_root = self.filesystem.metastudy_root(metastudy_slug=metastudy_slug)
        metastudy_state = self.filesystem.read_json(
            path=metastudy_root / "state.json",
            model_type=MetastudyState,
        )
        pending_study_slugs = [
            slug
            for slug in metastudy_state.pending_study_slugs
            if slug != directive.study_slug
        ]
        self.filesystem.write_json(
            path=metastudy_root / "state.json",
            model=metastudy_state.model_copy(
                update={
                    "active_study_slug": directive.study_slug,
                    "pending_study_slugs": pending_study_slugs,
                    "status": "active",
                }
            ),
        )
        return root

    def plan_study(self, *, study_slug: str) -> StudyPlan:
        root = self.filesystem.study_root(study_slug=study_slug)
        question_markdown = (root / "question.md").read_text(encoding="utf-8")
        state = self.filesystem.read_json(path=root / "state.json", model_type=StudyState)
        plan = self.model_client.generate_structured(
            role="study_agent",
            system_prompt=self.prompt_library.load(role="study_agent"),
            input_sections={
                "question_markdown": question_markdown,
                "state": state.model_dump(mode="json"),
            },
            output_type=StudyPlan,
        )
        self.filesystem.write_study_plan(study_slug=study_slug, plan=plan)
        self.filesystem.write_json(
            path=root / "state.json",
            model=state.model_copy(
                update={
                    "status": "in_review",
                    "plan_revision": state.plan_revision + 1,
                    "pending_substudy_slugs": [
                        task.substudy_slug for task in plan.substudies
                    ],
                }
            ),
        )
        return plan

    def review_study(self, *, study_slug: str) -> ReviewerVerdict:
        root = self.filesystem.study_root(study_slug=study_slug)
        state = self.filesystem.read_json(path=root / "state.json", model_type=StudyState)
        plan = self.filesystem.read_json(path=root / "plan.json", model_type=StudyPlan)
        verdict = self.model_client.generate_structured(
            role="study_reviewer",
            system_prompt=self.prompt_library.load(role="study_reviewer"),
            input_sections={
                "plan": plan.model_dump(mode="json"),
                "state": state.model_dump(mode="json"),
                "question_markdown": (root / "question.md").read_text(encoding="utf-8"),
            },
            output_type=ReviewerVerdict,
        )
        self.filesystem.write_study_review(study_slug=study_slug, verdict=verdict)
        self.filesystem.write_json(
            path=root / "state.json",
            model=state.model_copy(
                update={
                    "status": self._status_from_verdict(verdict=verdict),
                    "last_verdict": verdict.verdict,
                }
            ),
        )
        return verdict

    def bootstrap_substudy_from_task(
        self,
        *,
        study_slug: str,
        task: SubstudyTask,
    ) -> Path:
        state = SubstudyState(
            substudy_id=task.substudy_id,
            substudy_slug=task.substudy_slug,
            study_slug=study_slug,
            status="drafting",
        )
        task_markdown = self._render_substudy_task(task=task)
        root = self.filesystem.initialize_substudy(
            study_slug=study_slug,
            substudy_slug=task.substudy_slug,
            title=task.title,
            task_markdown=task_markdown,
            state=state,
        )
        study_root = self.filesystem.study_root(study_slug=study_slug)
        study_state = self.filesystem.read_json(path=study_root / "state.json", model_type=StudyState)
        pending_substudy_slugs = [
            slug
            for slug in study_state.pending_substudy_slugs
            if slug != task.substudy_slug
        ]
        self.filesystem.write_json(
            path=study_root / "state.json",
            model=study_state.model_copy(
                update={
                    "active_substudy_slug": task.substudy_slug,
                    "pending_substudy_slugs": pending_substudy_slugs,
                    "status": "active",
                }
            ),
        )
        return root

    def record_substudy_result(
        self,
        *,
        study_slug: str,
        substudy_slug: str,
        result: SubstudyResult,
    ) -> SubstudyResult:
        substudy_root = self.filesystem.substudy_root(
            study_slug=study_slug,
            substudy_slug=substudy_slug,
        )
        state = self.filesystem.read_json(
            path=substudy_root / "state.json",
            model_type=SubstudyState,
        )
        self.filesystem.write_substudy_result(
            study_slug=study_slug,
            substudy_slug=substudy_slug,
            result=result,
        )
        self.filesystem.write_json(
            path=substudy_root / "state.json",
            model=state.model_copy(
                update={
                    "status": "in_review",
                    "attempt_count": state.attempt_count + 1,
                    "result_artifact_paths": [artifact.path for artifact in result.artifacts],
                }
            ),
        )
        return result

    def review_substudy_result(
        self,
        *,
        study_slug: str,
        substudy_slug: str,
    ) -> ReviewerVerdict:
        substudy_root = self.filesystem.substudy_root(
            study_slug=study_slug,
            substudy_slug=substudy_slug,
        )
        result = self.filesystem.read_json(
            path=substudy_root / "result.json",
            model_type=SubstudyResult,
        )
        state = self.filesystem.read_json(
            path=substudy_root / "state.json",
            model_type=SubstudyState,
        )
        verdict = self.model_client.generate_structured(
            role="substudy_reviewer",
            system_prompt=self.prompt_library.load(role="substudy_reviewer"),
            input_sections={
                "result": result.model_dump(mode="json"),
                "state": state.model_dump(mode="json"),
                "task_markdown": (substudy_root / "task.md").read_text(encoding="utf-8"),
            },
            output_type=ReviewerVerdict,
        )
        next_substudy_status = "accepted" if verdict.verdict == "pass" else "revise_requested"
        if verdict.verdict == "reject":
            next_substudy_status = "blocked"
        self.filesystem.write_substudy_review(
            study_slug=study_slug,
            substudy_slug=substudy_slug,
            verdict=verdict,
        )
        self.filesystem.write_json(
            path=substudy_root / "state.json",
            model=state.model_copy(
                update={
                    "status": next_substudy_status,
                    "last_verdict": verdict.verdict,
                }
            ),
        )
        study_root = self.filesystem.study_root(study_slug=study_slug)
        study_state = self.filesystem.read_json(path=study_root / "state.json", model_type=StudyState)
        accepted_substudy_slugs = list(study_state.accepted_substudy_slugs)
        completed_substudy_slugs = list(study_state.completed_substudy_slugs)
        if verdict.verdict == "pass":
            if substudy_slug not in accepted_substudy_slugs:
                accepted_substudy_slugs.append(substudy_slug)
            if substudy_slug not in completed_substudy_slugs:
                completed_substudy_slugs.append(substudy_slug)
        self.filesystem.write_json(
            path=study_root / "state.json",
            model=study_state.model_copy(
                update={
                    "accepted_substudy_slugs": accepted_substudy_slugs,
                    "completed_substudy_slugs": completed_substudy_slugs,
                    "active_substudy_slug": "",
                    "last_verdict": verdict.verdict,
                }
            ),
        )
        return verdict

    def synthesize_study(self, *, study_slug: str) -> StudySynthesis:
        study_root = self.filesystem.study_root(study_slug=study_slug)
        state = self.filesystem.read_json(path=study_root / "state.json", model_type=StudyState)
        accepted_results = [
            self.filesystem.read_json(
                path=self.filesystem.substudy_root(
                    study_slug=study_slug,
                    substudy_slug=substudy_slug,
                )
                / "result.json",
                model_type=SubstudyResult,
            ).model_dump(mode="json")
            for substudy_slug in state.accepted_substudy_slugs
        ]
        synthesis = self.model_client.generate_structured(
            role="substudy_synthesizer",
            system_prompt=self.prompt_library.load(role="substudy_synthesizer"),
            input_sections={
                "study_state": state.model_dump(mode="json"),
                "accepted_results": accepted_results,
                "study_question_markdown": (study_root / "question.md").read_text(
                    encoding="utf-8"
                ),
            },
            output_type=StudySynthesis,
        )
        self.filesystem.write_study_synthesis(study_slug=study_slug, synthesis=synthesis)
        self.filesystem.write_json(
            path=study_root / "state.json",
            model=state.model_copy(
                update={
                    "status": "completed" if synthesis.status == "completed" else "active",
                }
            ),
        )
        return synthesis

    def _render_study_question(self, *, directive: StudyDirective) -> str:
        input_lines = "\n".join(f"- {item}" for item in directive.suggested_inputs)
        success_lines = "\n".join(f"- {item}" for item in directive.success_criteria)
        if not input_lines:
            input_lines = "- None specified."
        if not success_lines:
            success_lines = "- Produce a directly answerable study report."
        return (
            f"# {directive.title}\n\n"
            f"{directive.question}\n\n"
            "## Rationale\n\n"
            f"{directive.rationale}\n\n"
            "## Inputs\n\n"
            f"{input_lines}\n\n"
            "## Success Criteria\n\n"
            f"{success_lines}\n"
        )

    def _render_substudy_task(self, *, task: SubstudyTask) -> str:
        return (
            f"# {task.title}\n\n"
            "## Question\n\n"
            f"{task.question}\n\n"
            "## Scope\n\n"
            f"{self._render_markdown_list(task.scope)}\n\n"
            "## Non-Goals\n\n"
            f"{self._render_markdown_list(task.non_goals)}\n\n"
            "## Inputs\n\n"
            f"{self._render_markdown_list(task.inputs)}\n\n"
            "## Theory Anchors\n\n"
            f"{self._render_markdown_list(task.theory_anchors)}\n\n"
            "## Allowed Read Roots\n\n"
            f"{self._render_markdown_list(task.allowed_read_roots)}\n\n"
            "## Allowed Write Roots\n\n"
            f"{self._render_markdown_list(task.allowed_write_roots)}\n\n"
            "## Expected Artifacts\n\n"
            f"{self._render_markdown_list(task.expected_artifacts)}\n\n"
            "## Success Criteria\n\n"
            f"{self._render_markdown_list(task.success_criteria)}\n\n"
            "## Required Report Sections\n\n"
            f"{self._render_markdown_list(task.required_report_sections)}\n\n"
            "## Run Budget\n\n"
            f"- max_attempts: {task.run_budget.max_attempts}\n"
            f"- max_model_calls: {task.run_budget.max_model_calls}\n"
            f"- max_commands: {task.run_budget.max_commands}\n"
            f"- max_minutes: {task.run_budget.max_minutes}\n"
        )

    def _render_markdown_list(self, items: list[str]) -> str:
        if items:
            return "\n".join(f"- {item}" for item in items)
        return "- None specified."

    def _status_from_verdict(self, *, verdict: ReviewerVerdict) -> str:
        if verdict.verdict == "pass":
            return "active"
        if verdict.verdict == "revise":
            return "drafting"
        return "blocked"


__all__ = ["AutoresearchOrchestrator"]
