# Autoresearch agent handbook

This is a research repository with strict rules for orchestration and role delegation.
The user specifies metastudies with clear top-level objectives, and agents plan and design studies and substudies. The levels of abstraction, and role interactions, are detailed as follows:

- **Metastudies**: top-level objectives.
    - Involved agents: metastudy executor reviewer, and planner
- **Studies**: answering specific questions, or exploring a specific direction.
    - We make this semantic separation because study inputs are "a specific semantic, or scientific, question that needs to be answered", while a substudy is a particular engineering experiment which needs to be done, with semantic output.
- **Substudies**: one, or a closely semantically aligned list, of experiments or implementations which need to be done.

## Workflow details

Here, we adopt a programmatic specification of the workflow and clearly define the pieces of data and information-processing nodes. Declarations below.

Information processing nodes:

1. Metastudy architect
2. Metastudy planner
3. Metastudy executor
4. Metastudy reviewer
5. Study planner
6. Study executor
7. Study reviewer
8. Substudy executor
9. Substudy reviewer

Passed data:

1. Metastudy `spec`, `plan`, `state`, `audit`. Artifacts belong in the metastudy folder.
2. For each study, study `plan`, `state`, `audit`. Artifacts belong in the study subfolder.
3. For each substudy, substudy `state`, `audit`. Artifacts belong in in the substudy sub-subfolder.

### 1. Metastudy architect

#### Semantic policy

- Reads: user-provided objective material, relevant theory/codebase context, and any existing metastudy artifacts when revising.
- Sharpen the top-level research objective into a metastudy-level semantic contract.
- Iterate the objective with progressive sharpening and proposed questions until the user explicitly approves.

Read the theory codebase and existing metastudy results / artifacts / status report to contextualize; identify ambiguities, hidden assumptions, non-goals, and success criteria; iterate with the user until the metastudy objective is crisp enough to drive downstream planning.

- Output meaning: the metastudy `spec` should be the human-readable source of truth for what the metastudy is trying to answer and what counts as answering it.

#### Interactions
- Interacts with: user.

- Boundary: the metastudy architect is not responsible for decomposition, scheduling, or operational book-keeping.

#### Side effects
- Creates or revises the metastudy `spec` markdown file.
- Does not write metastudy `plan`, `state`, or `audit`.
- Hands off only through the metastudy `spec`.

### 2. Metastudy planner

#### Semantic policy

- Reads: metastudy `spec`, metastudy `state`, and current metastudy `plan`.
- Convert the metastudy-level semantic contract into an explicit list of studies that together should answer the metastudy objective.
- Output meaning: the metastudy `plan` is the current decomposition of the metastudy into child studies, together with the semantic rationale for why those studies are the right next set of questions or directions.
- The planner should work at the study level of abstraction, not at the substudy or experiment-design level.
- The planner may refine an existing plan in light of current metastudy state by proposing studies, but **should not directly mutate metastudy state** itself.

#### Interactions
- Interacts with: metastudy executor.

- Boundary: the metastudy planner does not execute studies or judge completion.

#### Side effects
- Creates or revises the metastudy `plan` markdown file.
- Does not write metastudy `state` or metastudy `audit`. Does not populate or create metastudy artifacts.
- Hands off only through the metastudy `plan` and any newly materialized study specs.

### 3. Metastudy executor

#### Semantic policy

- Reads: metastudy `spec`, metastudy `plan`, metastudy `state`, metastudy `audit`, and child study `state` artifacts.
- Maintain the metastudy as a long-lived stateful process which keeps driving the study loop until the metastudy-level question is either answered or explicitly sent back for revision.
- Output meaning: the metastudy `state` is the operational source of truth for where the metastudy currently stands, which studies have returned, what the current high-level findings are, and what the next action should be.
- The executor is the only metastudy-level node which should translate reviewer directives back into ongoing process state.

#### Interactions
- Interacts with: metastudy planner, study executor, and metastudy reviewer.

- Boundary: the metastudy executor does not define the metastudy objective, does not decompose the metastudy semantically on its own, and does not make the final judgement of pass versus revision.

#### Side effects
- Creates or revises the metastudy `state` markdown file.
- Initializes metastudy `state` when entering the driving loop if it does not yet exist.
- Invokes the metastudy planner when the current metastudy `plan` is missing, stale, or insufficient for the next pass through the studies.
- Invokes the study executor sequentially over the studies implied by the current metastudy `plan`.
- Updates metastudy `state` after each child study returns, so that the metastudy can be resumed without needing to inspect child audits.
- Invokes the metastudy reviewer after the relevant child studies for the current plan pass have returned.
- Reads metastudy `audit` and translates its directives into updated metastudy `state`.
- Does not write metastudy `spec`, metastudy `plan`, or metastudy `audit`.

### 4. Metastudy reviewer

#### Semantic policy

- Reads: metastudy `spec`, metastudy `plan`, metastudy `state`, and child study `state` artifacts.
- Judge whether the metastudy, at its current level of evidence and synthesis, has actually answered the top-level objective well enough to pass.
- Output meaning: the metastudy `audit` is the metastudy-level judgement artifact. It should clearly state whether the metastudy passes or requires revision, and why.
- The reviewer should reason from the metastudy objective and the accumulated study states, not from child audits. The metastudy should depend on study states as the upward-facing semantic summaries.
- The reviewer should be strict. It should not grant pass merely because all planned studies completed; it should evaluate whether the metastudy objective was in fact answered and push for modifications or more brainstorming as needed.
- If something is not explained concisely with standalone, compelling references (e.g. graphics) and numbers, then it's not ready.
- Check that metadata human deliverables are standalone and provides enough context.

#### Interactions
- Interacts with: metastudy executor.

- Boundary: the reviewer does not update metastudy `state`, does not revise the plan itself, and does not execute any further studies.

#### Side effects
- Creates or revises the metastudy `audit` markdown file.
- Emits a clear metastudy-level judgement of `pass` or `revision`, together with directives for what remains unresolved when revision is required.
- Hands off only through the metastudy `audit`.
- Does not write metastudy `spec`, metastudy `plan`, or metastudy `state`.

### 5. Study planner

#### Semantic policy

- Reads: metastudy `plan`, study `state`, and current study `plan`.
- Convert the parent metastudy direction for this study into a concrete list of substudies that together should answer the study-level question.
- Output meaning: the study `plan` is the current decomposition of the study into child substudies, together with the semantic rationale for why those substudies are the right next experiments or implementations.
- The planner should work at the substudy level of abstraction, not at the individual low-level training-step or coding-step level.
- The planner may refine an existing study plan in light of current study state by proposing substudies, but **should not directly mutate study state** itself.

#### Interactions
- Interacts with: study executor.

- Boundary: the study planner does not execute substudies or judge completion.

#### Side effects
- Creates or revises the study `plan` markdown file.
- Does not write study `state` or study `audit`.
- Hands off only through the study `plan`.

### 6. Study executor

#### Semantic policy

- Reads: metastudy `plan`, study `plan`, study `state`, study `audit`, and child substudy `state` artifacts.
- Maintain the study as a long-lived stateful process which keeps driving the substudy loop until the study-level question is either answered or explicitly sent back for revision.
- Output meaning: the study `state` is the operational source of truth for where the study currently stands, which substudies have returned, what the current study-level findings are, and what the next action should be.
- The executor is the only study-level node which should translate reviewer directives back into ongoing process state.

#### Interactions
- Interacts with: study planner, substudy executor, and study reviewer.

- Boundary: the study executor does not define the study question, does not decompose the study semantically on its own, and does not make the final judgement of pass versus revision.

#### Side effects
- Creates or revises the study `state` markdown file.
- Initializes study `state` when entering the driving loop if it does not yet exist.
- Invokes the study planner when the current study `plan` is missing, stale, or insufficient for the next pass through the substudies.
- Invokes the substudy executor over the substudies implied by the current study `plan`.
- Updates study `state` after each child substudy returns, so that the study can be resumed without needing to inspect child audits.
- Invokes the study reviewer after the relevant child substudies for the current plan pass have returned.
- Reads study `audit` and translates its directives into updated study `state`.
- Does not write study `plan` or study `audit`.

### 7. Study reviewer

#### Semantic policy

- Reads: metastudy `plan`, study `state`, and study `plan`.
- Judge whether the study, at its current level of evidence and synthesis, has actually answered the study-level question well enough to pass.
- Output meaning: the study `audit` is the study-level judgement artifact. It should clearly state whether the study passes or requires revision, and why.
- The reviewer should reason from the study question as situated within the metastudy plan, and from the accumulated substudy results as represented in study state, not from substudy audits.
- The reviewer should be strict. It should not grant pass merely because all currently planned substudies completed; it should evaluate whether the study question was in fact answered and push for further experiments or reframing as needed.

#### Interactions
- Interacts with: study executor.

- Boundary: the reviewer does not update study `state`, does not revise the plan itself, and does not execute any further substudies.

#### Side effects
- Creates or revises the study `audit` markdown file.
- Emits a clear study-level judgement of `pass` or `revision`, together with directives for what remains unresolved when revision is required.
- Hands off only through the study `audit`.
- Does not write study `plan` or study `state`.

### 8. Substudy executor

#### Semantic policy

- Reads: study `plan`, substudy `state`, and, when present, parent study context as carried through the study plan.
- Carry out a single concrete substudy or a tightly aligned bundle of concrete experiments/implementations until the substudy has produced a meaningful result to report upward.
- Output meaning: the substudy `state` is the operational and semantic record of what was attempted, what happened, what artifacts were produced, and what the substudy currently concludes.
- The substudy executor operates at the lowest abstraction level in the hierarchy. It is responsible for concrete experimental action, not for planning broader scientific direction.

#### Interactions
- Interacts with: study executor and substudy reviewer.

- Boundary: the substudy executor does not redefine the study question, does not create the study plan, and does not make the final judgement of whether the substudy is adequate as returned.

#### Side effects
- Creates or revises the substudy `state` markdown file.
- Initializes substudy `state` when the substudy is first launched if it does not yet exist.
- Runs the concrete experiment or implementation work implied by the substudy.
- Records concrete outcomes, produced artifacts, failures, and provisional conclusions in substudy `state`.
- Invokes the substudy reviewer when the substudy has reached a point where it can be judged.
- Reads reviewer guidance as reflected back into substudy `state` or via the study executor's control flow.
- Does not write study `plan`, study `state`, or any `audit` artifact.

### 9. Substudy reviewer

#### Semantic policy

- Reads: study `plan` and substudy `state`.
- Judge whether the substudy, as executed, is sufficient to report back to the parent study, or whether it should continue with revision.
- Output meaning: the substudy `audit` is the substudy-level judgement artifact. It should clearly state whether the substudy should return upward or continue under revision, and why.
- The reviewer should reason from the substudy's role within the study plan and from the actual concrete outcomes recorded in substudy state.
- The reviewer should be strict. It should not accept a substudy merely because some experiment was run; it should evaluate whether the substudy produced meaningful evidence relative to its intended role in the study.

#### Interactions
- Interacts with: substudy executor.

- Boundary: the reviewer does not update substudy `state`, does not revise the study plan itself, and does not execute any further experimental work.

#### Side effects
- Creates or revises the substudy `audit` markdown file.
- Emits a clear substudy-level judgement of `report back` or `continue with revision`, together with directives for what remains unresolved when further work is required.
- Hands off only through the substudy `audit`.
- Does not write study `plan` or substudy `state`.

### Metastudy layout

All files and md artifacts should be concise, informative, and **prefer robust interlinking over monolithic bloating**. If you can't explain or document something clearly, then it's not ready.

It is strongly encouraged to back up `state.md` and `audit.md` with figures whenever that materially improves clarity. Visual and other report-supporting artifacts should go under the local `figures/` directory at the corresponding metastudy / study / substudy level.

```text
metastudies/
└── <metastudy_slug>/
    ├── spec.md
    ├── plan.md
    ├── state.md
    ├── audit.md
    ├── figures/
    ├── scripts/
    └── studies/
        └── <study_slug>/
            ├── plan.md
            ├── state.md
            ├── audit.md
            ├── figures/
            ├── scripts/
            └── substudies/
                └── <substudy_slug>/
                    ├── state.md
                    ├── audit.md
                    ├── artifacts/
                    ├── figures/
                    └── scripts/
```

#### Spec file

The spec file should clearly include the following:

1. Clear re-iteration of the objectives so that the reviewer has clear instructions and context.
2. Enough guidelines, or specifications, to get a planner to implement a concrete study.
    - In case of a metastudy spec file, enough guidance for the study executor to make a plan.
    - In case of a substudy spec file, clear guidance and specification of the substudy so that the executor can implement it.
3. Indices to resource files and context.

#### Plan file format

1. Link to the spec file
2. Clear, standalone reiteration of the metastudy / study objectives.
3. An clear list of lower-level primitives (study / substudy) specifications, see "spec file" subsection.
4. Spec list should be append-only: executor and planners can, based on reviewer feedback, add to the plan file, but not modify existing sections.

#### State file format

1. Link to spec and plan.
2. Append-only list of metastudy / study / substudy results:
    - Metastudy: standalone report of study outcomes with reference as necessary, along the lines of how it has progressed along our objective.
    - Study: same as metastudy
    - Substudy: concrete experiment / implementation results.
3. State remains executor-owned at every level.
4. If review changes the process state, the executor should append a standalone summary of what the review implies for next steps.
5. Reviewer-authored judgement should live in `audit`, not be written directly into `state`.
6. When tables, plots, diagrams, or visual comparisons materially clarify the state, place them under `figures/` and reference them from `state.md` rather than embedding bulky content inline.

#### Audit file format

1. Reiterate the sharp objective provided in the plan, or audit context
2. Append-only list for each round of review:
    - Clear verdict:
      - Metastudy / study: `pass` or `revision`
      - Substudy: `report back` or `continue with revision`
    - Synthesis (high-level) and details of execution artifacts, with clear analysis of their relevance. Draw conclusions, extrapolate, and relate **experimental** / lower-level outcomes with progress along the high-level objective.
    - If revision: clear instructions for sharpened intermediate objectives and proposed studies / experiments.
3. When visual evidence or report-style artifacts materially strengthen the review, place them under `figures/` and reference them from `audit.md`.

## Holistic workflow

- User provides short instruction, or markdown file for study specs.
- User iterates with **metastudy architect** to produce a concise but clear metastudy spec.
- We enter the stateful metastudy-executor driving loop:
    - Initialize with empty `metastudy state` and `plan`
    - Repeat until the **metastudy reviewer** explicitly approves the study
        - Metastudy planner consumes `state`, `spec`, and `plan` to produce new `plan`.
        - **Metastudy executor** starts the _study-executor_ loop sequentially across study specifications:
            - Initialize with empty `study state` and `plan`
            - Repeat until the **study reviewer** explicitly approves the study
                - Study planner consumes (1) `metastudy-plan` with study-index info, (2) `study state`, (3) `study plan` to produce new `study plan`
                - For each substudy in the plan, **study executor** launches **substudy executor**
                    - Substudy initialized with empty `substudy state`
                    - Substudy executor consumes (1) `study-plan` with substudy index, (2) `substudy state`, to produce updated `substudy state`
                    - Substudy reviewer consumes `study-plan`, `substudy-state` to produce `substudy-audit` detailing `continue with revision` or `report back`
                    - Substudy executor reads `substudy-audit` and updates `substudy-state` accordingly
                - After a substudy returns, **study executor** updates the `study state`. If all substudy returns, study executor invokes **study reviewer**
                - **study reviewer** consumes `metastudy-plan`, `study-state` to produce `study-audit` with directives and a clear "pass / revision". Returns to the **study executor**.
                - **Study executor** views the `study-audit` and produces updated `study state` with all the relevant info; metastudy should not need to read the audit, just the state is enough. In case of pass, return. In case of revision, go back to the planner step
            - After a study returns, **metastudy executor** updates the `metastudy state`. If all study returned, invokes **metastudy reviewer**
            - **metastudy reviewer** consumes `metastudy-plan`, `metastudy-spec`, and `study-state` for all child studies (but not audits!!) to produce a clear `metastudy-audit` with directives and pass / revision. Yields to **metastudy executor**
            - **metastudy executor** views the audit and updates `metastudy state`. If pass, loop finishes, or report back
