# autoresearch

A hierarchical agentic research framework for structured experimental investigation. Metastudy planning now lives in the repo-local skill at `.agents/skills/metastudy-planner/`; `autoresearch` starts once a metastudy already has `plan.md`, then executes and reviews the work across study and substudy levels.

## Hierarchy

```
metastudy/          ← research campaign (weeks-to-months scope)
  studies/
    <study>/        ← research question (days scope)
      substudies/
        <substudy>/ ← single training run (hours scope)
```

## Agent Roles

| Agent | Role | Invoked by |
|---|---|---|
| `metastudy-executor` | Orchestrates studies end-to-end; synthesizes the final report; may spawn follow-up studies | User |
| `metastudy-reviewer` | Critically assesses the completed metastudy against its objective | `metastudy-executor` |
| `study-planner` | Decomposes a study objective into concrete, independent single-GPU substudies | `study-executor` |
| `study-executor` | Queues substudies across the GPUs present on the machine; tracks state; writes study report; invokes reviewer | `metastudy-executor` |
| `study-reviewer` | Critically assesses the completed study against its research question | `study-executor` |
| `substudy-executor` | Runs one experiment on one assigned GPU: writes config, runs training, writes report | `study-executor` |

## Invocation

First, plan the metastudy in the current Codex session with `.agents/skills/metastudy-planner/`. Once `plan.md` exists, the user invokes `metastudy-executor` with a metastudy directory path. Everything after that point is automatic:

```
metastudy-executor(<metastudy-dir>)
  └── study-executor × N         [one per study in plan]
        └── study-planner        [if plan.md missing]
        └── substudy-executor × M    [queued in parallel, up to one active substudy per present GPU]
  └── metastudy-reviewer
  └── [follow-up study-executors, if review warrants]
```

## Per-Level File Convention

Each level (`metastudy/`, `study/`, `substudy/`) maintains the same five files:

| File | Written by | Mutable? | Purpose |
|---|---|---|---|
| `objective.md` | Parent agent or user | Never | Immutable goal specification |
| `plan.md` | Planner agent | Never after creation | Decomposition into sub-units |
| `state.md` | Executor agent | Append-to-top only | Live execution log; safe to read mid-run |
| `report.md` | Executor agent | Once on completion | Synthesized findings |
| `review.md` | Reviewer agent | Once on completion | Critical assessment |

Substudies additionally contain:
- `config.py` — Python experiment config written by `substudy-executor`
- `artifacts/` — training outputs (created by the training run itself)

## state.md Format

`state.md` is an append-to-top log: new entries are **prepended** to the top of the file. Reading the file top-to-bottom shows newest-to-oldest. This lets you inspect which runs are active and see partial results without waiting for the campaign to finish.

Entry format:
```markdown
---
## [YYYY-MM-DD HH:MM UTC] <Event type>: <name>
**Outcome**: ...
**Key metrics**: ...
**Notes**: ...
```

## Resumability

Executors check `state.md` at startup to identify already-completed sub-units and skip them. An interrupted run can be re-invoked from the top level and will pick up where it left off, assuming the metastudy plan already exists.

## Shared Constraints

Inherited from `metastudies/AGENTS.md`:
- **Do not modify shared library code** in `src/`. All experiment customization lives in substudy `config.py` files or temporary scripts in `/tmp/`.
- **Stripe experiments across the GPUs present on the machine.** The `study-executor` manages the per-study queue and assigns at most one active substudy to each GPU; each `substudy-executor` remains a simple single-GPU run.

## Dashboard

The dashboard lives in `autoresearch/dashboard/` and gives a live-ish operator view over metastudies, agent runs, logs, and past Codex conversations.

Run it with `uv`:

```bash
uv run python autoresearch/dashboard/server.py
```

Then open:

```text
http://127.0.0.1:8765
```

### What it shows

- **Metastudy selector**: choose any directory under `metastudies/`
- **Left pane: filesystem tree**: browse the selected metastudy directory and open files like `objective.md`, `plan.md`, `state.md`, `report.md`, `review.md`, `config.py`, and `training.log`
- **Right pane: agent spawn tree**: reconstructed from `run.log`, grouped by metastudy, study, and substudy directories
- **Inspector**:
  - tails `run.log` and `training.log`
  - opens study/substudy markdown files
  - loads past Codex conversations by matching `run.log` session IDs to local `~/.codex/sessions/**/*.jsonl`

### How the conversation view works

`autoresearch/scripts/_launch_common.py` writes `INFO` lines such as:

```text
codex resume --include-non-interactive <session-id>
```

The dashboard parses those session IDs out of `run.log`, finds the corresponding local JSONL transcript under `~/.codex/sessions/`, and renders the parsed conversation in the inspector. The raw JSONL is also available from the same view.

### Notes

- The UI auto-selects the most recently active metastudy.
- The inspector can auto-tail the currently opened log file.
- Parsed transcripts hide the giant bootstrap prompt blocks by default; use the transcript toggle to reveal them.
