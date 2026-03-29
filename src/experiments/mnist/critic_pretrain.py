from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from src.chart_transport.training import sample_transport_times
from src.experiments.mnist.common import (
    format_metrics_summary,
    log_wandb_scalars,
    runtime_precision_context,
    should_log_monitor,
)
from src.monitoring.utils import MonitorStage

if TYPE_CHECKING:
    from src.experiments.mnist.state import MNISTTrainingRuntime


def _compute_critic_pretrain_loss(
    *,
    rt: "MNISTTrainingRuntime",
    batch_size: int,
) -> dict[str, torch.Tensor]:
    encoder = rt.chart_transport_model.encoder
    critic = rt.chart_transport_model.critic

    data_batch = rt.runtime_data_config.sample_unconditional(
        batch_size=batch_size,
    )
    with torch.no_grad():
        data_latents = encoder(data_batch)

    t = sample_transport_times(
        transport_config=rt.tc.chart_transport_config.loss_config.transport_config,
        device=rt.device,
        batch_shape=(data_latents.shape[0],),
    )
    eps = torch.randn_like(data_latents)
    noised_latents = (1.0 - t).unsqueeze(-1) * data_latents + t.unsqueeze(-1) * eps
    predicted_noise = critic(noised_latents, t)
    critic_loss = F.mse_loss(predicted_noise, eps)
    return {
        "critic_loss": critic_loss,
    }


def _detach_metrics(
    *,
    losses: dict[str, torch.Tensor],
) -> dict[str, float]:
    return {key: value.detach().item() for key, value in losses.items()}


def critic_pretrain_train_step_(
    *,
    rt: "MNISTTrainingRuntime",
) -> dict[str, float]:
    encoder = rt.chart_transport_model.encoder
    decoder = rt.chart_transport_model.decoder
    critic = rt.chart_transport_model.critic

    encoder.eval()
    decoder.eval()
    critic.train()

    rt.optimizer.zero_grad(set_to_none=True)
    with runtime_precision_context(rt=rt):
        losses = _compute_critic_pretrain_loss(
            rt=rt,
            batch_size=rt.tc.train_batch_size,
        )
    rt.fabric.backward(losses["critic_loss"])
    rt.fabric.clip_gradients(
        rt.chart_transport_model,
        rt.optimizer,
        max_norm=rt.tc.chart_transport_config.architecture_config.grad_clip_norm,
        error_if_nonfinite=False,
    )
    rt.optimizer.step()
    return _detach_metrics(losses=losses)


def critic_pretrain_eval_step_(
    *,
    rt: "MNISTTrainingRuntime",
    step: int,
) -> dict[str, float]:
    critic = rt.chart_transport_model.critic

    critic_was_training = critic.training
    critic.eval()

    with torch.no_grad():
        with runtime_precision_context(rt=rt):
            monitor_metrics = rt.tc.monitor_config.critic_monitor_config.apply_to(
                rt=rt,
                step=step,
                stage=MonitorStage.CRITIC,
            )

    if critic_was_training:
        critic.train()

    return monitor_metrics


def critic_pretrain_(
    *,
    rt: "MNISTTrainingRuntime",
) -> dict[str, float]:
    total_steps = rt.tc.chart_transport_config.scheduling_config.pretrain_critic_n_steps
    log_every_n_steps = (
        rt.tc.monitor_config.schedule_config.log_every_n_steps_critic_pretrain
    )

    latest_metrics: dict[str, float] = {}
    progress = tqdm(
        range(1, total_steps + 1),
        desc="critic_pretrain",
    )
    for step in progress:
        train_metrics = rt._critic_pretrain_train_step()
        latest_metrics = {f"train_{key}": value for key, value in train_metrics.items()}
        log_wandb_scalars(
            rt=rt,
            stage="critic_pretrain",
            step=step,
            metrics=train_metrics,
        )
        progress.set_postfix(
            critic_loss=f"{train_metrics['critic_loss']:.4f}",
        )

        if should_log_monitor(
            step=step,
            total_steps=total_steps,
            every_n_steps=log_every_n_steps,
        ):
            monitor_metrics = rt._critic_pretrain_eval_step(step=step)
            latest_metrics.update(
                {f"monitor_{key}": value for key, value in monitor_metrics.items()}
            )
            log_wandb_scalars(
                rt=rt,
                stage="critic_pretrain",
                step=step,
                metrics={
                    f"monitor_{key}": value for key, value in monitor_metrics.items()
                },
            )
            rt.fabric.print(
                f"[critic_pretrain] step {step}/{total_steps}: "
                f"train: {format_metrics_summary(metrics=train_metrics)}; "
                f"monitor: {format_metrics_summary(metrics=monitor_metrics)}"
            )

    return latest_metrics


__all__ = [
    "critic_pretrain_",
    "critic_pretrain_eval_step_",
    "critic_pretrain_train_step_",
]
