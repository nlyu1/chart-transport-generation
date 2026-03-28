# Background

- You're working in a uv library, always use `uv run python` for Python invocations.
- Always use top-level bash instead of sandbox, requiring escalation as necessary.
- When specifying a cuda device, it isn't sufficient to move model and data `to(device)`; wrap the whole context in `with torch.cuda.device(device), torch.device(device), torch.autocast(device_type=..., dtype=...)` (or TE equivalent); transformer engine MLP kernel breaks on float32.
- Unless otherwise specified, always run training & benchmarking in bfloat16 or transformer engine fp8 (training default is Fabric precision `bf16-mixed` in `prometheus/python/prometheus/training/config.py`).
- Unless otherwise specified, do not set default config values during development, and prefer keyword arguments
- Unless otherwise specified, keep __init__.py optimal and only export submodules. Imports should clearly specify the full path.

# Style

- Strongly prefer pydantic dataclasses and clear typing; disprefer "A | B" permissive type signatures. Do not use any `Type | None` field signatures.
- Prefer keyword-only arguments; strongly dis-prefer "Type | None" patterns.
- Unless absolutely needed, strongly disprefer default instance values.
- Prefer strict failure to permissive try/except
- Do not write over-defensive validation which clutter code.
- Strongly prefer Jaxtyping tensors. Prefer useful typing and faithful usage over over-defensive checks.
- We strongly prefer empty "__init__.py" and use full-path imports.
- **Do not** write backward-compatible alias / patches when making changes. Make change as if it's implemented for the first time, and it suffices to explicitly inform me of breaking changes

# Testing

- This is a research repo, so testing is not needed.
- You are encouraged to snippet-test your code by "uv run python -c", or putting one-time tests in /tmp.

# Project scope

- Read `theory/proposal.typ` for theoretical specifications of the model. JUST DO IT.
- We're building a highly structured, modular, and well-maintained research codebase for generative modeling experiments.
- Experiments are fully specified by stateless, immutable, composable configs (pydantic classes), which are instantiated to stateful objects during runtime.