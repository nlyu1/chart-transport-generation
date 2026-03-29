from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    import torch.nn as nn

    from src.experiments.multimodal_gaussian.state import MultimodalTrainingRuntime


@dataclass(
    config=ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    ),
    kw_only=True,
)
class CycleLosses:
    data_batch: Tensor
    prior_batch: Tensor
    data_latents: Tensor
    data_reconstruction: Tensor
    prior_reconstruction: Tensor
    prior_latents: Tensor
    data_cycle_loss: Tensor
    prior_cycle_loss: Tensor


@contextlib.contextmanager
def runtime_precision_context(
    *,
    rt: "MultimodalTrainingRuntime",
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


@contextlib.contextmanager
def preserve_module_train_states(
    *,
    modules: list["nn.Module"],
) -> Iterator[None]:
    training_states = [module.training for module in modules]
    try:
        yield
    finally:
        for module, was_training in zip(modules, training_states, strict=True):
            module.train(was_training)


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
    rt: "MultimodalTrainingRuntime",
    stage: str,
    step: int,
    metrics: dict[str, float],
) -> None:
    payload = {f"{stage}/{key}": value for key, value in metrics.items()}
    payload[f"{stage}/local_step"] = step
    rt.wandb_run.log(payload)


def detach_metrics(
    *,
    losses: dict[str, Tensor],
) -> dict[str, float]:
    return {key: value.detach().item() for key, value in losses.items()}


def optimizer_step_(
    *,
    rt: "MultimodalTrainingRuntime",
    loss: Tensor,
) -> None:
    rt.optimizer.zero_grad(set_to_none=True)
    rt.fabric.backward(loss)
    rt.fabric.clip_gradients(
        rt.chart_transport_model,
        rt.optimizer,
        max_norm=rt.tc.chart_transport_config.architecture_config.grad_clip_norm,
        error_if_nonfinite=False,
    )
    rt.optimizer.step()


def compute_cycle_losses(
    *,
    rt: "MultimodalTrainingRuntime",
    batch_size: int,
) -> CycleLosses:
    constraint_config = rt.tc.chart_transport_config.loss_config.constraint_config
    prior_config = rt.tc.chart_transport_config.prior_config

    data_batch = rt.runtime_data_config.sample_unconditional(
        batch_size=batch_size,
    )
    prior_batch = prior_config.sample(
        batch_size=batch_size,
    ).to(device=rt.device, dtype=torch.float32)

    data_latents = rt.chart_transport_model.encoder(data_batch)
    data_reconstruction = rt.chart_transport_model.decoder(data_latents)
    prior_reconstruction = rt.chart_transport_model.decoder(prior_batch)
    prior_latents = rt.chart_transport_model.encoder(prior_reconstruction)

    data_cycle_loss = F.huber_loss(
        data_reconstruction,
        data_batch,
        delta=constraint_config.huber_delta,
        reduction="mean",
    )
    prior_cycle_loss = F.huber_loss(
        prior_latents,
        prior_batch,
        delta=constraint_config.huber_delta,
        reduction="mean",
    )
    return CycleLosses(
        data_batch=data_batch,
        prior_batch=prior_batch,
        data_latents=data_latents,
        data_reconstruction=data_reconstruction,
        prior_reconstruction=prior_reconstruction,
        prior_latents=prior_latents,
        data_cycle_loss=data_cycle_loss,
        prior_cycle_loss=prior_cycle_loss,
    )


__all__ = [
    "CycleLosses",
    "compute_cycle_losses",
    "detach_metrics",
    "format_metrics_summary",
    "log_wandb_scalars",
    "optimizer_step_",
    "preserve_module_train_states",
    "runtime_precision_context",
    "should_log_monitor",
]
