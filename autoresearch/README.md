# Autoresearch

Autoresearch is a thin scaffold for running research with Codex as the worker.

The intended split is:

- Codex does the actual planning, coding, command execution, and iteration inside a task.
- Files under `artifacts/` hold the durable state.
- Small helpers under `autoresearch/` make the workflow repeatable.

This is deliberately simpler than a full multi-agent runtime.

## Top-level model

There are three levels of work:

1. `metastudy`: the long-running research thread you discuss directly.
2. `study`: one focused question inside the metastudy.
3. `substudy`: one concrete work item executed by one Codex task or session.

Recommended behavior split:

- planner: updates `plan.md` and creates study/task files,
- executor: does one substudy,
- reviewer: checks whether the substudy answered the question,
- synthesizer: combines accepted results into a study report.

These are roles, not separate runtime systems.

## Source of truth

Persistent Codex sessions are useful, but they are not canonical.

The canonical state is always on disk:

- metastudy files under `artifacts/metastudies/`,
- study files under `artifacts/studies/`,
- study-local scripts, data, and figures under the relevant study folder.

If a Codex session disappears, a new one should be able to resume from files alone.

## Recommended directory usage

### Metastudy

```text
artifacts/metastudies/<metastudy_slug>/
  question.md
  plan.md
  journal.md
  dashboard.md
  studies.json
  session_registry.json
```

### Study

```text
artifacts/studies/<study_slug>/
  question.md
  plan.md
  report.md
  state.json
  tasks/
  results/
  synthesis.md
  scripts/
  data/
  figures/
```

## How to use this

### 1. Start a metastudy

Create:

- `artifacts/metastudies/<slug>/question.md`

This should contain:

- the high-level goal,
- pointers to relevant theory, code, and artifacts,
- constraints,
- what kind of answer you eventually want.

Then ask Codex, in a metastudy-planner session, to produce:

- `plan.md`
- `studies.json`
- an initial `dashboard.md`

### 2. Create one study at a time

For the next highest-priority question, create:

- `artifacts/studies/<study_slug>/question.md`
- `artifacts/studies/<study_slug>/plan.md`
- `artifacts/studies/<study_slug>/tasks/*.json`

Keep the study narrow. One study should answer one related topic, not the entire research agenda.

### 3. Run substudies through Codex

Each `tasks/<id>.json` is one Codex work order.

The executor session should:

- read the task file,
- inspect repo context,
- write study-local scripts/configs/results,
- keep all writes under `artifacts/` or `/tmp`,
- produce `results/<id>.md`.

### 4. Review before synthesis

A reviewer session should read:

- `tasks/<id>.json`
- `results/<id>.md`

and write:

- `results/<id>.review.json`

Only accepted results should feed synthesis.

### 5. Synthesize and update the plan

After enough accepted results exist, a synthesizer session should write:

- `synthesis.md`
- updates to `report.md`

Then the metastudy planner can update the top-level plan and queue the next study.

## Persistent Codex sessions

The workflow is simpler if you keep a few stable Codex sessions alive:

- one for metastudy planning,
- one per active study,
- optionally one reviewer session reused across studies.

Track these in:

- `artifacts/metastudies/<slug>/session_registry.json`

Suggested contents:

- role name,
- scope slug,
- Codex session or thread identifier,
- notes about what that session owns.

This avoids losing continuity while keeping continuity explicit.

## What the Python scaffold is for

The Python code under `autoresearch/` should stay thin.

Useful responsibilities:

- initialize folder skeletons,
- create task file templates,
- keep status indexes up to date,
- render dashboards,
- provide light convenience commands.

Avoid turning it into a replacement for Codex’s own execution loop.

## Current repo constraints

- Do not modify shared reusable library code under `src/` as part of autoresearch execution.
- Study work should happen in `artifacts/` and `/tmp`.
- Prefer study-local scripts, copied configs, and wrappers.
- Every important claim must point to local artifacts.

## Suggested next step

The immediate next use of this scaffold should be the first multimodal Gaussian metastudy:

- bootstrap the metastudy question,
- draft the first metastudy plan,
- define the first study around low-dimensional verification and instrumentation quality.
