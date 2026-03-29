# Autoresearch TODOs

## High priority

- Decide the exact minimal schema for `artifacts/metastudies/<slug>/studies.json`.
- Decide the exact minimal schema for `artifacts/metastudies/<slug>/session_registry.json`.
- Decide whether task IDs should be numeric per study, globally unique, or both.
- Decide the review status enum for `results/*.review.json`.
- Decide the exact required sections for `results/<task_id>.md`.
- Decide the exact required sections for `synthesis.md`.

## Codex workflow design

- Write the repo-level `AGENTS.md` guidance for autoresearch usage.
- Collapse the current role prompts into a smaller skills model:
  - `planner`
  - `executor`
  - `reviewer`
  - optional `synthesizer`
- Decide whether these should live as real Codex skills under `.agents/skills/` or remain in `autoresearch/` first.
- Define how a persistent metastudy session should be resumed from files.
- Define how a study session should claim ownership of one study slug.
- Define how reviewer sessions stay independent from executor sessions.

## Task contract details

- Decide the final JSON shape for `tasks/*.json`.
- Decide whether tasks should include explicit command budgets.
- Decide whether tasks should include explicit theory anchors.
- Decide how much prior artifact context should be copied into the task file versus discovered by Codex.
- Decide whether tasks should explicitly list forbidden write roots or only allowed write roots.

## File conventions

- Decide whether `results/<id>.md` should include a fixed header template.
- Decide whether `report.md` should be updated incrementally or only after synthesis.
- Decide whether study-level `state.json` is still needed or whether the task/review files are enough.
- Decide whether `journal.md` should exist at the study level as well as the metastudy level.
- Decide how to link from metastudy plans to study reports in a stable way.

## Thin Python helpers

- Simplify `autoresearch` Python helpers so they reflect the Codex-centric design rather than a full internal runtime.
- Decide which current helper modules are genuinely useful versus overbuilt.
- Add a tiny CLI for:
  - creating a metastudy skeleton,
  - creating a study skeleton,
  - creating a substudy task file,
  - updating a dashboard or study index.
- Add small template emitters for `question.md`, `plan.md`, `task.json`, `result.md`, and `review.json`.

## Multimodal Gaussian bootstrap

- Write the first metastudy `question.md` for the multimodal Gaussian problem.
- Define the first study slug and question.
- Define the first 2-3 substudy task files.
- Decide the standard metrics and artifact checklist each multimodal Gaussian substudy must produce.
- Decide the initial set of transport-related levers to consider, including `T_MIN`.

## Nice to have

- Add a helper that summarizes accepted study results into the metastudy dashboard.
- Add a helper that verifies every study report cites local artifacts.
- Add a helper that checks for orphaned tasks with no result or review file.
- Add a small example metastudy under `artifacts/metastudies/` once the schemas are locked.
