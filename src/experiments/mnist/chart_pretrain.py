from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from src.experiments.mnist.common import (
    format_metrics_summary,
    log_wandb_scalars,
    runtime_precision_context,
    should_log_monitor,
)
from src.monitoring.utils import MonitorStage

if TYPE_CHECKING:
    from src.experiments.mnist.state import MNISTTrainingRuntime


def _compute_chart_pretrain_losses(
    *,
    rt: "MNISTTrainingRuntime",
    batch_size: int,
) -> dict[str, torch.Tensor]:
    chart_transport_config = rt.tc.chart_transport_config
    chart_pretrain_config = chart_transport_config.loss_config.chart_pretrain_config
    constraint_config = chart_transport_config.loss_config.constraint_config
    prior_config = chart_transport_config.prior_config

    encoder = rt.chart_transport_model.encoder
    decoder = rt.chart_transport_model.decoder

    data_batch = rt.runtime_data_config.sample_unconditional(
        batch_size=batch_size,
    )
    prior_batch = prior_config.sample(
        batch_size=batch_size,
    ).to(device=rt.device, dtype=torch.float32)

    data_latents = encoder(data_batch)
    decoder_outputs = decoder(
        torch.cat(
            [data_latents, prior_batch],
            dim=0,
        )
    )
    data_reconstruction, prior_reconstruction = decoder_outputs.split(
        [batch_size, batch_size],
        dim=0,
    )
    prior_latents = encoder(prior_reconstruction)

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
    zero_mean_loss = F.huber_loss(
        data_latents.mean(),
        torch.zeros((), device=rt.device, dtype=data_latents.dtype),
        delta=1.0,
        reduction="mean",
    )
    latent_norms = data_latents.norm(dim=-1)
    latent_norm_loss = F.huber_loss(
        latent_norms,
        torch.zeros_like(latent_norms),
        delta=chart_pretrain_config.latent_norm_delta,
        reduction="mean",
    )

    chart_loss = data_cycle_loss + prior_cycle_loss
    chart_loss = (
        chart_loss
        + chart_pretrain_config.zero_mean_weight * zero_mean_loss
        + chart_pretrain_config.latent_norm_weight * latent_norm_loss
    )

    return {
        "chart_loss": chart_loss,
        "data_cycle_loss": data_cycle_loss,
        "prior_cycle_loss": prior_cycle_loss,
        "zero_mean_loss": zero_mean_loss,
        "latent_norm_loss": latent_norm_loss,
    }


def _detach_metrics(
    *,
    losses: dict[str, torch.Tensor],
) -> dict[str, float]:
    return {key: value.detach().item() for key, value in losses.items()}


def chart_pretrain_train_step_(
    *,
    rt: "MNISTTrainingRuntime",
) -> dict[str, float]:
    encoder = rt.chart_transport_model.encoder
    decoder = rt.chart_transport_model.decoder
    critic = rt.chart_transport_model.critic

    encoder.train()
    decoder.train()
    critic.eval()

    rt.optimizer.zero_grad(set_to_none=True)
    with runtime_precision_context(rt=rt):
        losses = _compute_chart_pretrain_losses(
            rt=rt,
            batch_size=rt.tc.train_batch_size,
        )
    rt.fabric.backward(losses["chart_loss"])
    rt.fabric.clip_gradients(
        rt.chart_transport_model,
        rt.optimizer,
        max_norm=rt.tc.chart_transport_config.architecture_config.grad_clip_norm,
        error_if_nonfinite=False,
    )
    rt.optimizer.step()
    return _detach_metrics(losses=losses)


def chart_pretrain_eval_step_(
    *,
    rt: "MNISTTrainingRuntime",
    step: int,
) -> dict[str, float]:
    encoder = rt.chart_transport_model.encoder
    decoder = rt.chart_transport_model.decoder

    encoder_was_training = encoder.training
    decoder_was_training = decoder.training
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        with runtime_precision_context(rt=rt):
            monitor_metrics = rt.tc.monitor_config.constraint_monitor_config.apply_to(
                rt=rt,
                step=step,
                stage=MonitorStage.CHART,
            )

    if encoder_was_training:
        encoder.train()
    if decoder_was_training:
        decoder.train()

    return monitor_metrics


def chart_pretrain_(
    *,
    rt: "MNISTTrainingRuntime",
) -> dict[str, float]:
    total_steps = rt.tc.chart_transport_config.scheduling_config.pretrain_chart_n_steps
    log_every_n_steps = (
        rt.tc.monitor_config.schedule_config.log_every_n_steps_chart_pretrain
    )

    latest_metrics: dict[str, float] = {}
    progress = tqdm(
        range(1, total_steps + 1),
        desc="pretrain",
    )
    for step in progress:
        train_metrics = rt._chart_pretrain_train_step()
        latest_metrics = {f"train_{key}": value for key, value in train_metrics.items()}
        log_wandb_scalars(
            rt=rt,
            stage="pretrain",
            step=step,
            metrics=train_metrics,
        )
        progress.set_postfix(
            chart_loss=f"{train_metrics['chart_loss']:.4f}",
            data_cycle=f"{train_metrics['data_cycle_loss']:.4f}",
            prior_cycle=f"{train_metrics['prior_cycle_loss']:.4f}",
        )

        if should_log_monitor(
            step=step,
            total_steps=total_steps,
            every_n_steps=log_every_n_steps,
        ):
            monitor_metrics = rt._chart_pretrain_eval_step(step=step)
            latest_metrics.update(
                {f"monitor_{key}": value for key, value in monitor_metrics.items()}
            )
            log_wandb_scalars(
                rt=rt,
                stage="pretrain",
                step=step,
                metrics={
                    f"monitor_{key}": value for key, value in monitor_metrics.items()
                },
            )
            rt.fabric.print(
                f"[pretrain] step {step}/{total_steps}: "
                f"train: {format_metrics_summary(metrics=train_metrics)}; "
                f"monitor: {format_metrics_summary(metrics=monitor_metrics)}"
            )

    return latest_metrics


__all__ = [
    "chart_pretrain_",
    "chart_pretrain_eval_step_",
    "chart_pretrain_train_step_",
]
