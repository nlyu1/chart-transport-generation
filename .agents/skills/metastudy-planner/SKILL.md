---
name: metastudy-planner
description: Plan or revise a metastudy in the current Codex session for a named metastudy or metastudy path. Use when Codex should become the repository's metastudy planner directly inside this session.
---

# Metastudy Planner

## Overview

Use this skill to make the current Codex session act as the metastudy planner. Resolve the user's metastudy name or path, validate that `objective.md` exists, then follow the repository's metastudy planning workflow in-session rather than spawning `codex` recursively.

## Workflow

1. Expect either a bare metastudy name such as `multimodal-gaussian-2d` or a path such as `metastudies/multimodal-gaussian-2d`.
2. If the user gives only a bare name, resolve it to `metastudies/<name>`.
3. Validate the target from the repo root with the helper script:

```bash
uv run python .agents/skills/metastudy-planner/scripts/launch.py <metastudy-name-or-path>
```

4. After validation, read:
   - `.agents/skills/metastudy-planner/references/metastudy-planner-workflow.md`
   - `metastudies/AGENTS.md`
   - `AGENTS.md`
   - the target metastudy's `objective.md`
   - `plan.md`, `state.md`, and relevant study-level files if present
   - `theory/proposal.typ`
   - relevant experiment/config files referenced by the workflow
5. Treat `.agents/skills/metastudy-planner/references/metastudy-planner-workflow.md` as the authoritative workflow, but execute it in the current thread. The mandatory clarification-and-approval loop still applies.
6. Keep planning in the current session. Do not defer to a nested planner launcher.

## Failure Handling

- If the metastudy directory does not exist, stop and report the missing path.
- If `objective.md` is missing, stop and tell the user to create it before launching.
- If validation fails, summarize the error and show the exact command that was attempted.

## Notes

- The current Codex session owns the clarification loop, existing-plan recap, plan edits, and the final handoff command.
- The helper script is only a resolver and validator. It must not spawn a nested Codex session.
- This skill is repo-local. Use paths relative to `/home/nlyu/Code/diffusive-latent-generation/`.

## Resource

- `scripts/launch.py`: resolve a metastudy name or path, validate the target, and print planner context for the current session.
- `references/metastudy-planner-workflow.md`: authoritative planner workflow to follow in-session.
