---
name: metastudy-planner
description: Launch the repository's interactive autoresearch metastudy planner for a named metastudy or metastudy path. Use when Codex should start or resume metastudy planning or plan revision in this repository, especially when the user gives a metastudy name and wants the existing `autoresearch` planner workflow rather than an in-session rewrite.
---

# Metastudy Planner

## Overview

Use this skill as a thin entrypoint to the existing `autoresearch` metastudy planner. Resolve the user's metastudy name or path, validate that `objective.md` exists, and launch the planner session instead of re-implementing the planner logic in the current thread.

## Workflow

1. Expect either a bare metastudy name such as `multimodal-gaussian-2d` or a path such as `metastudies/multimodal-gaussian-2d`.
2. If the user gives only a bare name, resolve it to `metastudies/<name>`.
3. Launch the helper script from the repo root:

```bash
uv run python skills/metastudy-planner/scripts/launch.py <metastudy-name-or-path>
```

4. Treat the launcher as the primary implementation. Do not inline the planner workflow in the current session unless the launcher is unavailable or the user explicitly asks not to start a nested interactive planner session.
5. If the user wants a different model, pass `AUTORESEARCH_MODEL=<model>` through the environment before launching the helper script.

## Failure Handling

- If the metastudy directory does not exist, stop and report the missing path.
- If `objective.md` is missing, stop and tell the user to create it before launching.
- If launch fails, summarize the error and show the exact command that was attempted.

## Notes

- The planner prompt owns the clarification loop, existing-plan recap, plan edits, and the final handoff command.
- The helper script supports `--print-command` for dry runs and validation without starting the nested Codex session.
- This skill is repo-local. Use paths relative to `/home/nlyu/Code/diffusive-latent-generation/`.

## Resource

- `scripts/launch.py`: resolve a metastudy name or path, validate the target, optionally print the underlying planner command, and launch `autoresearch/scripts/launch-metastudy-planner.py`.
