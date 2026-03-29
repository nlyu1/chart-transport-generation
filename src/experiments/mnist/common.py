from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.experiments.mnist.state import MNISTTrainingRuntime


@contextlib.contextmanager
def runtime_precision_context(
    *,
    rt: "MNISTTrainingRuntime",
) -> Iterator[None]:
    with contextlib.ExitStack() as stack:
        if rt.device.type == "cuda":
            stack.enter_context(torch.cuda.device(rt.device))
        stack.enter_context(torch.device(str(rt.device)))
        stack.enter_context(
            torch.autocast(
                device_type=rt.device.type,
                dtype=torch.bfloat16,
            )
        )
        yield


def format_metrics_summary(
    *,
    metrics: dict[str, float],
) -> str:
    return ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())


def should_log_monitor(
    *,
    step: int,
    total_steps: int,
    every_n_steps: int,
) -> bool:
    return step == 1 or step % every_n_steps == 0 or step == total_steps


def log_wandb_scalars(
    *,
    rt: "MNISTTrainingRuntime",
    stage: str,
    step: int,
    metrics: dict[str, float],
) -> None:
    payload = {f"{stage}/{key}": value for key, value in metrics.items()}
    payload[f"{stage}/local_step"] = step
    rt.wandb_run.log(payload)


__all__ = [
    "format_metrics_summary",
    "log_wandb_scalars",
    "runtime_precision_context",
    "should_log_monitor",
]
