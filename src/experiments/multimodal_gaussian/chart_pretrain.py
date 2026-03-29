from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

if TYPE_CHECKING:
    from src.experiments.multimodal_gaussian.state import MultimodalTrainingRuntime


@contextlib.contextmanager
def _runtime_precision_context(
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


def _format_metrics_summary(
    *,
    metrics: dict[str, float],
) -> str:
    return ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())


def _should_log_monitor(
    *,
    step: int,
    total_steps: int,
    every_n_steps: int,
) -> bool:
    return step == 1 or step % every_n_steps == 0 or step == total_steps


def _log_wandb_scalars(
    *,
    rt: "MultimodalTrainingRuntime",
    stage: str,
    step: int,
    metrics: dict[str, float],
) -> None:
    payload = {f"{stage}/{key}": value for key, value in metrics.items()}
    payload[f"{stage}/local_step"] = step
    rt.wandb_run.log(payload)


def _compute_chart_pretrain_losses(
    *,
    rt: "MultimodalTrainingRuntime",
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
    rt: "MultimodalTrainingRuntime",
) -> dict[str, float]:
    encoder = rt.chart_transport_model.encoder
    decoder = rt.chart_transport_model.decoder

    encoder.train()
    decoder.train()

    rt.optimizer.zero_grad(set_to_none=True)
    with _runtime_precision_context(rt=rt):
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
    rt: "MultimodalTrainingRuntime",
    step: int,
    run_constraint_monitor: bool,
    run_conditioning_monitor: bool,
) -> dict[str, float]:
    encoder = rt.chart_transport_model.encoder
    decoder = rt.chart_transport_model.decoder

    encoder_was_training = encoder.training
    decoder_was_training = decoder.training
    encoder.eval()
    decoder.eval()

    monitor_metrics: dict[str, float] = {}
    with _runtime_precision_context(rt=rt):
        if run_constraint_monitor:
            monitor_metrics.update(
                rt.tc.monitor_config.constraint_monitor_config.apply_to(
                    rt=rt,
                    step=step,
                )
            )
        if run_conditioning_monitor:
            monitor_metrics.update(
                rt.tc.monitor_config.conditioning_monitor_config.apply_to(
                    rt=rt,
                    step=step,
                )
            )

    if encoder_was_training:
        encoder.train()
    if decoder_was_training:
        decoder.train()

    return monitor_metrics


def chart_pretrain_(
    *,
    rt: "MultimodalTrainingRuntime",
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
        _log_wandb_scalars(
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

        if _should_log_monitor(
            step=step,
            total_steps=total_steps,
            every_n_steps=log_every_n_steps,
        ):
            monitor_metrics = rt._chart_pretrain_eval_step(
                step=step,
                run_constraint_monitor=True,
                run_conditioning_monitor=True,
            )
            latest_metrics.update(
                {f"monitor_{key}": value for key, value in monitor_metrics.items()}
            )
            _log_wandb_scalars(
                rt=rt,
                stage="pretrain",
                step=step,
                metrics={
                    f"monitor_{key}": value for key, value in monitor_metrics.items()
                },
            )
            monitor_summary_metrics = {
                key: value
                for key, value in monitor_metrics.items()
                if "_mode_" not in key
            }
            rt.fabric.print(
                f"[pretrain] step {step}/{total_steps}: "
                f"train: {_format_metrics_summary(metrics=train_metrics)}; "
                f"monitor: {_format_metrics_summary(metrics=monitor_summary_metrics)}"
            )

    return latest_metrics


__all__ = [
    "chart_pretrain_",
    "chart_pretrain_eval_step_",
    "chart_pretrain_train_step_",
]
