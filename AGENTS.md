# Repository Guidelines

## Project Structure & Module Organization
This repo is organized as a Cartesian product of three moving pieces:

- Common reusable tools in `src/`: `src/model/` for function approximators, `src/data/` for dataset access and configs, `src/priors/` for latent priors, `src/chart_transport/` for objectives and training utilities, and `src/monitoring/` for generative diagnostics.
- Stateless method and experiment specification: immutable Pydantic config objects define architectures, losses, schedules, priors, and monitoring without hidden runtime state.
- Stateful runtime instantiation in `src/experiments/`: experiment packages such as `src/experiments/mnist/` and `src/experiments/multimodal_gaussian/` turn configs into models, optimizers, Fabric state, and training loops.
- Theory and intent live in `theory/`. Read `theory/proposal.typ` before changing objectives, priors, or transport logic.
- Outputs belong in `artifacts/`; old code and notebooks belong in `archives/`.

## Build, Test, and Development Commands
Always use `uv`.

- `uv sync --dev`: install runtime and dev dependencies.
- `uv run ruff check .`: lint the codebase.
- `uv run python -c "import src.experiments.multimodal_gaussian.integrated as m; print(m.__file__)"`: trace implementation paths.
- `uv run python -c "from src.experiments.multimodal_gaussian.canonical import get_canonical_chart_transport_configs"`: quick import smoke test.

For debugging, prefer `uv run python -c ...` snippets or short one-off scripts in `/tmp`.

## GPU & Precision Usage
Use GPU code deliberately. Moving tensors and modules with `.to(device)` is not enough; wrap execution in `torch.cuda.device(device)`, `torch.device(device)`, and `torch.autocast(device_type=..., dtype=torch.bfloat16)` contexts. Default training and benchmarking to bfloat16 rather than float32.

## Coding Style & Naming Conventions
Use Python 3.12, 4-space indentation, full-path imports, and minimal `__init__.py` files. Prefer immutable Pydantic configs and `pydantic.dataclasses.dataclass(kw_only=True)` for runtime containers. Use keyword-only APIs, avoid permissive unions like `Type | None`, and prefer informative tensor typing, including Jaxtyping where useful.

Prefer explicit failure over permissive branching or broad `try/except`. Prefer centralized canonical configs over hidden defaults scattered across modules; default instance values should be rare. When changing an API, make the clean breaking change instead of adding backward-compatible aliases or compatibility shims.

Do not code up `__all__ = ...`

## Testing Guidelines
This is a research repo, so snippet validation is the default. Use focused `uv run python -c ...` checks to validate imports, shapes, and control flow. If you add tests, place them in `tests/` and name files `test_<feature>.py`.
