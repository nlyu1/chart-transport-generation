# Autoresearch specs

## Core decision

The unit of work is a Codex task or Codex session.

The scaffold does not try to implement its own rich agent runtime. It uses:

- markdown files for human-readable intent and synthesis,
- JSON files for lightweight machine-readable task contracts,
- Codex instructions, `AGENTS.md`, and skills for behavior,
- thin Python helpers only for file creation, status tracking, and convenience.

## Mental model

- `metastudy`: the long-running research thread you discuss directly.
- `study`: one focused topic inside the metastudy.
- `substudy`: one concrete experiment or analysis task executed by one Codex worker.

Codex is responsible for planning inside a task, reading code, editing study-local files, running commands, iterating, and producing artifacts.

The scaffold is responsible for:

- giving Codex a clean task contract,
- storing durable state in files,
- separating planning from execution from review,
- making it easy to resume or branch work across persistent Codex sessions.

## What is simplified

v1 does not need:

- an internal multi-agent runtime,
- programmatic handoffs,
- structured inter-agent RPC,
- a heavyweight orchestrator loop,
- hidden transcript state.

v1 does need:

- a clear folder layout,
- stable markdown and JSON contracts,
- reusable role instructions,
- explicit review gates,
- a way to keep one Codex session per active branch of work.

## Filesystem model

### Metastudy root

Each long-running research effort lives in:

```text
artifacts/metastudies/<metastudy_slug>/
  question.md
  plan.md
  journal.md
  dashboard.md
  studies.json
  session_registry.json
```

Purpose:

- `question.md`: the top-level goal, pointers, constraints, and current user intent.
- `plan.md`: the current metastudy plan. This is revised over time.
- `journal.md`: append-only high-level decisions and important findings.
- `dashboard.md`: short current status for the human.
- `studies.json`: machine-readable index of study status.
- `session_registry.json`: optional mapping from role or study slug to persistent Codex session/thread identifiers.

### Study root

Concrete work continues under:

```text
artifacts/studies/<study_slug>/
  question.md
  plan.md
  report.md
  state.json
  tasks/
    001.json
    002.json
  results/
    001.md
    001.review.json
    002.md
    002.review.json
  synthesis.md
  scripts/
  data/
  figures/
```

Purpose:

- `question.md`: the study contract.
- `plan.md`: the current study plan and study-specific hypotheses.
- `report.md`: the final cleaned study report.
- `state.json`: current lifecycle state.
- `tasks/*.json`: concrete substudy task contracts.
- `results/*.md`: substudy outputs written for human inspection.
- `results/*.review.json`: reviewer verdicts on substudy outputs.
- `synthesis.md`: study-level synthesis over accepted substudy results.

### Minimal substudy contract

Each `tasks/<id>.json` should contain only what the Codex worker actually needs:

- `task_id`
- `title`
- `question`
- `scope`
- `inputs`
- `expected_artifacts`
- `allowed_write_roots`
- `success_criteria`
- `review_rubric`

This should stay intentionally small. The task file is a work order, not a full ontology.

## Role model

Roles are behavioral modes, not framework objects.

### Metastudy planner

Input:

- metastudy `question.md`
- prior `plan.md`
- prior study reports

Output:

- updated metastudy `plan.md`
- updated `studies.json`
- journal entry if priorities changed

Primary job:

- decide which study to run next,
- keep the overall effort coherent,
- incorporate findings from completed studies.

### Study planner

Input:

- metastudy plan
- one study question
- relevant theory and prior artifacts

Output:

- `artifacts/studies/<study_slug>/plan.md`
- `tasks/*.json`

Primary job:

- convert one study question into a small sequence of substudies,
- make each substudy narrow and reviewable.

### Executor

Input:

- one substudy task file
- relevant repo context
- relevant prior artifacts

Output:

- `results/<task_id>.md`
- study-local scripts/data/figures as needed

Primary job:

- do the work,
- keep changes local to `artifacts/` and `/tmp`,
- use Codex’s normal code-reading/editing/run loop.

### Reviewer

Input:

- one substudy result
- original substudy task

Output:

- `results/<task_id>.review.json`

Primary job:

- pass, revise, or reject,
- identify unsupported claims,
- ensure the artifact trail is local and sufficient.

### Synthesizer

Input:

- accepted substudy results

Output:

- `synthesis.md`
- updated `report.md`

Primary job:

- combine only accepted findings,
- expose unresolved questions,
- recommend the next study or the next sharp substudy.

## Session model

Persistent Codex sessions are useful, but they are not the source of truth.

Recommended pattern:

- one metastudy session for top-level planning,
- one study session per active study,
- one executor session per active substudy when needed,
- optional separate reviewer session if you want strict separation.

The file system remains canonical. Sessions are convenience memory.

That means:

- every important output must be written to markdown or JSON,
- a dead session should be recoverable by starting a new Codex session from files alone,
- session identifiers belong in `session_registry.json`, not in your head.

## Python scope

Python remains thin.

Useful helpers:

- initialize a metastudy folder,
- initialize a study folder,
- create a task JSON skeleton,
- update `studies.json`,
- maintain `session_registry.json`,
- render dashboards and status summaries.

Not useful in v1:

- reimplementing Codex’s planning and execution loop,
- building a rich orchestration engine,
- forcing all reasoning through Pydantic models,
- trying to replace Codex threads with custom Python agents.

## Codex instruction strategy

Behavior should primarily live in:

- repo-level `AGENTS.md`,
- reusable Codex skills,
- local task markdown and JSON files.

The current `autoresearch/prompts/*.md` files can be treated as draft role instructions. They should likely be collapsed into a smaller number of skills:

- `planner`
- `executor`
- `reviewer`

Potentially a fourth:

- `synthesizer`

That is enough for v1.

## Constraints

- Shared reusable code under `src/` is read-only for the autoresearch harness.
- Codex workers may write under `artifacts/` and `/tmp`.
- Study-local wrappers, copied configs, and monkey-patches are allowed.
- Every material claim in a result or report must point to local artifacts.
- Reviews should happen before synthesis consumes a result.

## Initial target workflow

The first metastudy targets multimodal Gaussian chart transport.

The first study sequence should likely follow this shape:

1. confirm low-dimensional behavior and debugging visibility,
2. scale dimension in controlled sweeps,
3. inspect conditioning, transport, critic, and loss evolution,
4. isolate key levers such as `T_MIN` and related transport settings,
5. synthesize hypotheses about what matters for one-shot high-dimensional generation.

The key idea is not “run many experiments.” It is “run the minimum sharp sequence of studies that reveals the real levers.”
