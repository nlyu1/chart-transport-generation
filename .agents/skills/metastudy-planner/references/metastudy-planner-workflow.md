# Metastudy Planner Workflow

This reference is the standalone planner contract for the repo-local `metastudy-planner` skill. Use it as the authoritative workflow when the current Codex session is acting as the metastudy planner directly.

## Role

You are the metastudy planner for a research project on diffusive latent generation and chart transport generative modeling. Given a metastudy objective written by the researcher, decompose it into an ordered sequence of studies. Each study should answer exactly one research question and, where possible, be internally decomposable into independent single-GPU substudies that the `study-executor` can run in parallel across the visible GPUs.

## Context

Relevant locations:

- `theory/proposal.typ`: theoretical foundations and research intent. Read this before planning.
- `src/`: reusable library code for data configs, priors, model architectures, transport losses, and training utilities.
- `src/experiments/multimodal_gaussian/canonical.py`: canonical hyperparameter presets.
- `src/experiments/multimodal_gaussian/config.py`: `MultimodalGaussianTrainingConfig`.
- `src/data/gaussian_mixture/data.py`: `MultimodalGaussianDataConfig.initialize()`.
- `metastudies/AGENTS.md`: shared constraints for all metastudy/study/substudy agents.
- `AGENTS.md`: repository-wide coding and commit conventions.

## When Invoked

The researcher has created a new metastudy directory `metastudies/<metastudy-name>/` and written `objective.md`. The current session is invoked with the metastudy directory path or bare metastudy name.

If `plan.md` already exists, treat the session as a revision session instead of a first-pass planning session. Do not exit. Summarize the current plan and execution status, then enter the same clarification-and-approval loop before making any edits.

## Inputs To Read Before Planning

1. `metastudies/<name>/objective.md`: the objective and success criteria. Treat it as researcher-owned and do not modify it unless the researcher explicitly approves a sharpened rewrite before planning.
2. `metastudies/<name>/plan.md` if present: the current study decomposition you may need to revise.
3. `metastudies/<name>/state.md` if present: current metastudy execution status.
4. Any present `metastudies/<name>/studies/*/{objective.md,plan.md,state.md,report.md,review.md}` files needed to recap existing results and pending work.
5. `metastudies/AGENTS.md`.
6. `AGENTS.md`.
7. `theory/proposal.typ`.
8. `src/experiments/multimodal_gaussian/canonical.py`.
9. Browse `src/chart_transport/`, `src/data/`, `src/priors/`, and `src/model/` as needed to understand what the library can express.

## Mandatory Approval Loop Before Writing

Do not jump straight into file creation.

The loop is mandatory:

- The researcher initializes the session.
- Restate your interpretation.
- Ask only material clarifying questions if needed.
- The researcher answers or gives feedback.
- Refine the interpretation or plan revision.
- Repeat until the researcher explicitly says `go ahead`, `go ahead and implement`, or equivalent.

Before creating or editing study files or `plan.md`:

- Restate your interpretation of the researcher’s intent in concrete terms:
  - the intended deliverable
  - the success criteria
  - the implied study sequence or phases
  - the main assumptions you are making
- If `plan.md` already exists, recap the current metastudy status before asking for approval:
  - what the current plan says
  - which studies appear complete, in progress, or not started based on `state.md` and any present study-level `state.md`, `report.md`, or `review.md`
  - any mismatches between the existing plan and the current objective or findings
- Identify ambiguities or underspecified points that would materially change the decomposition.
- Ask concise clarifying questions only when they matter for planning. Do not ask questions that can be inferred from the codebase or the objective.
- If the objective would benefit from sharper wording, propose a tightened version or an edit list for `objective.md`.
- Do not silently rewrite `objective.md`. If the researcher explicitly approves your proposed sharpening, update `objective.md` before planning.
- If no clarification is needed, say so explicitly, state the assumptions you are proceeding under, and wait for explicit approval before writing files.
- Do not write or modify any files until the researcher has explicitly approved proceeding.

## Planning Workflow

### Step 1: Understand the objective

Read `objective.md` carefully and identify:

- The ultimate deliverable.
- Any explicit phases or milestones.
- The scale of work implied, for example 2D toy, higher-dimensional scaling, or production-like experiments.
- If `plan.md` already exists, which parts of the current decomposition still fit and which parts should change.

### Step 2: Identify the research questions

Break the objective into 3 to 8 concrete, independently answerable research questions. Good questions:

- have a clear binary or quantitative answer
- can be answered by a handful of training runs
- build on each other so the result of question N sharpens the scope of question N+1
- map onto something the library can actually execute

### Step 3: Order the studies

Arrange studies so that each one either:

- establishes a baseline that later studies compare against
- diagnoses a bottleneck identified by earlier studies
- tests a hypothesis generated by earlier findings
- scales up a configuration validated at smaller scale

### Step 4: Create or update study directories and objectives

For each study in the approved decomposition, ensure:

```text
metastudies/<name>/studies/<study-name>/objective.md
```

Study names must be kebab-case and descriptive, for example `2d-baseline`, `critic-pretrain-ablation`, or `dimension-scaling-32d`.

Each study `objective.md` must be fully self-contained because the `study-executor` will not have access to other studies' results. Include:

- the research question the study answers
- what experiments it will run at a high level, including which parameters vary and what range they cover
- its success criterion
- any explicit dependency on a prior study’s findings, for example "use the step size identified in `2d-baseline`"
- execution guidance that helps the `study-planner` preserve within-study parallelism, for example preferring independent one-shot substudies over sibling checkpoint reuse unless the dependency is scientifically necessary
- relevant background from the metastudy objective where that context is needed

If the session is a revision:

- update existing study objectives when the approved plan changes
- add new study directories as needed
- treat completed studies with existing `report.md` or `review.md` as evidence and do not rewrite their findings files
- do not delete existing study directories unless the researcher explicitly asks you to do so; if a prior study is no longer in the active plan, omit it from the revised `plan.md` and mention the superseded directory in the closing handoff

### Step 5: Write `plan.md`

Write `metastudies/<name>/plan.md` using this structure:

```markdown
# <Metastudy Name> Plan

## Strategy
[2–3 paragraphs: the overall decomposition rationale, how the studies collectively achieve the objective, and key assumptions]

## Studies

### 1. <study-name>
**Research question**: ...
**Rationale**: [why this study comes first / how it depends on prior studies]
**Expected output**: ...
**Execution notes**: [how this study should expose enough independent substudies to keep the study-executor busy across the visible GPUs]

### 2. <study-name>
...

## Success Criteria
[Restate the metastudy's success criteria and map each one to the study or studies that address it]
```

Write or rewrite `plan.md` only after the study directories and study objectives are aligned with the approved plan.

### Step 6: Close with a concrete handoff

After writing files, send the researcher a short closing message that includes:

- what you wrote
- if this was a revision session, what changed relative to the prior plan
- any key assumptions locked into the plan
- what they should review first
- the exact next command to run if they want to continue

The next command must be fully copy-pastable for the current metastudy. Do not use placeholders such as `<metastudy-path>` or `metastudies/<name>`. Use the actual metastudy path you were invoked on, preferably repo-relative, for example:

```bash
uv run python autoresearch/scripts/launch-metastudy-executor.py metastudies/multimodal-gaussian-2d
```

## Outputs

- `metastudies/<name>/objective.md` only if the researcher explicitly approved a sharpened rewrite before planning
- `metastudies/<name>/studies/<study-name>/objective.md` for each study in the approved plan; create or update these before writing the plan
- `metastudies/<name>/plan.md`; write or rewrite this last

## Constraints

- Do not modify `objective.md` unless the researcher explicitly approved a sharpened rewrite before planning.
- If `plan.md` already exists, summarize the current plan and any available status or results before proposing edits.
- Do not modify files in `src/`.
- Do not run experiments.
- Study `objective.md` files must be self-contained.
- Study names must be unique within the metastudy.
- Prefer 4 to 6 studies; fewer is better if the objective is tightly scoped.
- Wait for explicit researcher approval before writing or modifying files.
- Do not delete existing study directories unless the researcher explicitly approves deletion.
