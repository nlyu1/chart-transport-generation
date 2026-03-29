from __future__ import annotations

from typing import Literal

from pydantic import Field

from src.config.base import BaseConfig


class ArtifactRef(BaseConfig):
    path: str
    description: str


class RunBudget(BaseConfig):
    max_attempts: int = 1
    max_model_calls: int = 8
    max_commands: int = 8
    max_minutes: int = 120


class Hypothesis(BaseConfig):
    hypothesis_id: str
    statement: str
    rationale: str = ""
    evidence_needed: list[str] = Field(default_factory=list)


class StudyDirective(BaseConfig):
    study_id: str
    study_slug: str
    title: str
    question: str
    rationale: str
    suggested_inputs: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)


class MetastudyPlan(BaseConfig):
    metastudy_id: str
    title: str
    problem_statement: str
    theory_anchors: list[str] = Field(default_factory=list)
    active_hypotheses: list[Hypothesis] = Field(default_factory=list)
    study_backlog: list[StudyDirective] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)
    next_decision: str


class ReviewerVerdict(BaseConfig):
    subject_id: str
    subject_kind: Literal[
        "metastudy_plan",
        "study_plan",
        "substudy_result",
        "study_synthesis",
    ]
    verdict: Literal["pass", "revise", "reject"]
    summary: str
    issues: list[str] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)


class SubstudyTask(BaseConfig):
    substudy_id: str
    study_id: str
    substudy_slug: str
    title: str
    question: str
    scope: list[str] = Field(default_factory=list)
    non_goals: list[str] = Field(default_factory=list)
    theory_anchors: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    allowed_read_roots: list[str] = Field(default_factory=list)
    allowed_write_roots: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    required_report_sections: list[str] = Field(default_factory=list)
    run_budget: RunBudget = Field(default_factory=RunBudget)


class StudyPlan(BaseConfig):
    study_id: str
    metastudy_id: str
    title: str
    question: str
    theory_anchors: list[str] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    substudies: list[SubstudyTask] = Field(default_factory=list)
    expected_synthesis_questions: list[str] = Field(default_factory=list)
    next_review_trigger: str


class SubstudyResult(BaseConfig):
    substudy_id: str
    status: Literal["completed", "blocked", "needs_revision"]
    summary: str
    observed_facts: list[str] = Field(default_factory=list)
    interpretations: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    suggested_followups: list[str] = Field(default_factory=list)


class StudySynthesis(BaseConfig):
    study_id: str
    status: Literal["active", "completed", "needs_revision"]
    answer: str
    key_findings: list[str] = Field(default_factory=list)
    supported_hypotheses: list[str] = Field(default_factory=list)
    rejected_hypotheses: list[str] = Field(default_factory=list)
    unresolved_hypotheses: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    recommended_next_substudies: list[str] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)


class MetastudyState(BaseConfig):
    metastudy_id: str
    metastudy_slug: str
    status: Literal["drafting", "in_review", "active", "blocked", "completed"]
    plan_revision: int = 0
    active_study_slug: str = ""
    pending_study_slugs: list[str] = Field(default_factory=list)
    completed_study_slugs: list[str] = Field(default_factory=list)
    last_verdict: str = ""


class StudyState(BaseConfig):
    study_id: str
    study_slug: str
    metastudy_slug: str
    status: Literal["drafting", "in_review", "active", "blocked", "completed"]
    plan_revision: int = 0
    active_substudy_slug: str = ""
    pending_substudy_slugs: list[str] = Field(default_factory=list)
    completed_substudy_slugs: list[str] = Field(default_factory=list)
    accepted_substudy_slugs: list[str] = Field(default_factory=list)
    last_verdict: str = ""


class SubstudyState(BaseConfig):
    substudy_id: str
    substudy_slug: str
    study_slug: str
    status: Literal[
        "drafting",
        "running",
        "in_review",
        "accepted",
        "revise_requested",
        "blocked",
    ]
    attempt_count: int = 0
    last_verdict: str = ""
    result_artifact_paths: list[str] = Field(default_factory=list)


__all__ = [
    "ArtifactRef",
    "Hypothesis",
    "MetastudyPlan",
    "MetastudyState",
    "ReviewerVerdict",
    "RunBudget",
    "StudyDirective",
    "StudyPlan",
    "StudyState",
    "StudySynthesis",
    "SubstudyResult",
    "SubstudyState",
    "SubstudyTask",
]
