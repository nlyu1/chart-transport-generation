# autoresearch

A hierarchical agentic research framework for structured experimental investigation. Agents plan, execute, and review experiments at three levels of granularity, chaining automatically from a single user invocation.

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
| `metastudy-planner` | Decomposes a metastudy objective into an ordered sequence of studies | `metastudy-executor` (or user) |
| `metastudy-executor` | Orchestrates studies end-to-end; synthesizes the final report; may spawn follow-up studies | User |
| `metastudy-reviewer` | Critically assesses the completed metastudy against its objective | `metastudy-executor` |
| `study-planner` | Decomposes a study objective into concrete substudies (each = one training run) | `study-executor` |
| `study-executor` | Runs substudies in order; tracks state; writes study report; invokes reviewer | `metastudy-executor` |
| `study-reviewer` | Critically assesses the completed study against its research question | `study-executor` |
| `substudy-executor` | Runs one experiment: selects GPU, writes config, runs training, writes report | `study-executor` |

## Invocation

The user invokes `metastudy-executor` with a metastudy directory path. Everything else is automatic:

```
metastudy-executor(<metastudy-dir>)
  └── metastudy-planner          [if plan.md missing]
  └── study-executor × N         [one per study in plan]
        └── study-planner        [if plan.md missing]
        └── substudy-executor × M    [one per substudy in plan]
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

Executors check `state.md` at startup to identify already-completed sub-units and skip them. An interrupted run can be re-invoked from the top level and will pick up where it left off.

## Shared Constraints

Inherited from `metastudies/AGENTS.md`:
- **Do not modify shared library code** in `src/`. All experiment customization lives in substudy `config.py` files or temporary scripts in `/tmp/`.
- **Stripe experiments across available GPUs.** The `substudy-executor` selects the least-utilized GPU at runtime via `nvidia-smi`.
