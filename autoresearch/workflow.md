# Autoresearch Workflow

How to initiate and run a fully autonomous research campaign.

---

## Prerequisites

- `metastudies/AGENTS.md` — shared agent constraints (read once)
- `autoresearch/README.md` — framework overview
- Two GPUs available; training uses `bfloat16` and `uv`

---

## Step 1 — Create the metastudy

Create a directory and write the objective:

```bash
mkdir -p metastudies/<name>
```

Write `metastudies/<name>/objective.md`. The objective should specify:
- The research question or goal
- Explicit success criteria (quantitative where possible)
- Any known constraints or starting points

See `metastudies/multimodal-gaussian-baseline/objective.md` as an example.

---

## Step 2 — Run the planner interactively

The planner runs as an **interactive Codex session** — you can converse with it, steer the decomposition, and approve the result before execution begins.

The session should begin by:
- restating its interpretation of your objective
- surfacing any planning-relevant ambiguities
- asking clarifying questions when they materially affect the decomposition
- proposing a sharper `objective.md` if the current one is too vague

If you approve a sharpened rewrite, the planner may update `objective.md` before writing the plan. It should end with a short handoff telling you what it wrote, what assumptions it used, what to review, and the next command to run.

```bash
uv run python autoresearch/scripts/launch-metastudy-planner.py metastudies/<name>
```

The session will read `objective.md`, explore the library, and propose a decomposition into 3–8 studies. When the session ends, it will have written:
- `metastudies/<name>/studies/<study-name>/objective.md` for each study
- `metastudies/<name>/plan.md`

Review `plan.md` and edit study `objective.md` files if needed. **The executor requires `plan.md` to exist and will not start without it.**

---

## Step 3 — Launch the executor

```bash
uv run python autoresearch/scripts/launch-metastudy-executor.py metastudies/<name>
```

The executor runs non-interactively from this point. It will:

1. Run each study in order via `study-executor` (which auto-invokes `study-planner` if needed)
   - Each study invokes `study-planner` (if needed), then queues independent substudies across the GPUs present on the machine
   - Each `substudy-executor` owns exactly one GPU, writes `config.py`, runs training, and writes `report.md`
   - `study-reviewer` runs after each study completes
3. Invoke `metastudy-reviewer` after all studies complete
4. Optionally spawn follow-up studies if the reviewer identifies unresolved hypotheses
5. Write the final `metastudies/<name>/report.md`

All invocations are logged to `metastudies/<name>/run.log`.

---

## Monitoring progress

```bash
# Live log of agent invocations and completions
tail -f metastudies/<name>/run.log

# High-level metastudy state (newest entries first)
cat metastudies/<name>/state.md

# Per-study state
cat metastudies/<name>/studies/<study-name>/state.md
```

Log line format:
```
TIMESTAMP             ACTION  ROLE                PATH                              [exit:N  Ns]
2026-03-29T17:00:00Z  START   metastudy-planner   .../multimodal-gaussian-baseline
2026-03-29T17:02:30Z  DONE    metastudy-planner   .../multimodal-gaussian-baseline  exit:0  149s
2026-03-29T17:05:00Z  START   metastudy-executor  .../multimodal-gaussian-baseline
2026-03-29T17:05:01Z  START   study-executor      .../studies/2d-baseline
2026-03-29T17:05:02Z  START   study-planner       .../studies/2d-baseline
2026-03-29T17:06:10Z  DONE    study-planner       .../studies/2d-baseline           exit:0  68s
2026-03-29T17:06:11Z  START   substudy-executor   .../substudies/step-size-0p1
2026-03-29T17:06:12Z  START   substudy-executor   .../substudies/step-size-0p05
2026-03-29T18:10:00Z  DONE    substudy-executor   .../substudies/step-size-0p05     exit:0  3830s
2026-03-29T18:36:11Z  DONE    substudy-executor   .../substudies/step-size-0p1      exit:0  5400s
```

---

## Resuming an interrupted run

The executors are idempotent: they check `state.md` for completed studies/substudies and skip them. Simply re-run the same command:

```bash
uv run python autoresearch/scripts/launch-metastudy-executor.py metastudies/<name>
```

If a substudy's training crashed mid-run, its `report.md` will contain the error. The study-executor will still log it as complete and move on; the study-reviewer will flag it in `review.md`.

---

## Running a single study or substudy manually

```bash
# Run one study (calls planner if needed, runs all substudies, calls reviewer)
uv run python autoresearch/scripts/launch-study-executor.py \
  metastudies/<name>/studies/<study-name> \
  --log metastudies/<name>/run.log

# Run one substudy (writes config.py, trains, writes report.md)
uv run python autoresearch/scripts/launch-substudy-executor.py \
  metastudies/<name>/studies/<study-name>/substudies/<substudy-name> \
  --log metastudies/<name>/run.log
```

---

## Overriding the model

All agents use `gpt-5.4` by default. Override via:

```bash
AUTORESEARCH_MODEL=o3 uv run python autoresearch/scripts/launch-metastudy-executor.py metastudies/<name>
```

---

## Output locations

| Artifact | Location |
|---|---|
| Invocation log | `metastudies/<name>/run.log` |
| Metastudy plan | `metastudies/<name>/plan.md` |
| Metastudy state | `metastudies/<name>/state.md` |
| Metastudy report | `metastudies/<name>/report.md` |
| Metastudy review | `metastudies/<name>/review.md` |
| Study report | `metastudies/<name>/studies/<study>/report.md` |
| Substudy config | `metastudies/<name>/studies/<study>/substudies/<sub>/config.py` |
| Training log | `metastudies/<name>/studies/<study>/substudies/<sub>/training.log` |
| Model artifacts | `metastudies/<name>/studies/<study>/substudies/<sub>/artifacts/` |
