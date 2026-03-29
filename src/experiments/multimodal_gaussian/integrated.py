from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from src.chart_transport.constraint import (
    LagrangianConstraintConfig,
    LossConstraintConfig,
)
from src.chart_transport.training import estimate_transport_targets
from src.experiments.multimodal_gaussian.chart_pretrain import (
    _format_metrics_summary,
    _log_wandb_scalars,
    _runtime_precision_context,
    _should_log_monitor,
)

if TYPE_CHECKING:
    from src.experiments.multimodal_gaussian.state import MultimodalTrainingRuntime


def _detach_metrics(
    *,
    losses: dict[str, torch.Tensor],
) -> dict[str, float]:
    return {key: value.detach().item() for key, value in losses.items()}


def _constraint_budget(
    *,
    budget_per_dim: float,
    numel: int,
) -> float:
    return budget_per_dim * (numel**0.5)


def _compute_cycle_losses(
    *,
    rt: "MultimodalTrainingRuntime",
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    return data_cycle_loss, prior_cycle_loss


def _compute_constraint_repair_losses(
    *,
    rt: "MultimodalTrainingRuntime",
    batch_size: int,
) -> dict[str, torch.Tensor]:
    constraint_method = rt.tc.chart_transport_config.loss_config.constraint_config.constraint_method
    data_cycle_loss, prior_cycle_loss = _compute_cycle_losses(
        rt=rt,
        batch_size=batch_size,
    )

    if isinstance(constraint_method, LossConstraintConfig):
        repair_loss = (
            constraint_method.data_loss_weight * data_cycle_loss
            + constraint_method.prior_loss_weight * prior_cycle_loss
        )
        return {
            "repair_loss": repair_loss,
            "data_cycle_loss": data_cycle_loss,
            "prior_cycle_loss": prior_cycle_loss,
        }

    data_budget = _constraint_budget(
        budget_per_dim=constraint_method.data_constraint_budget_per_dim,
        numel=rt.runtime_data_config.data_numel(),
    )
    prior_budget = _constraint_budget(
        budget_per_dim=constraint_method.prior_constraint_budget_per_dim,
        numel=rt.tc.chart_transport_config.prior_config.latent_numel(),
    )
    repair_loss = data_cycle_loss + prior_cycle_loss
    repair_loss = repair_loss + rt.data_dual * (data_cycle_loss - data_budget)
    repair_loss = repair_loss + rt.prior_dual * (prior_cycle_loss - prior_budget)
    return {
        "repair_loss": repair_loss,
        "data_cycle_loss": data_cycle_loss,
        "prior_cycle_loss": prior_cycle_loss,
    }


def _update_constraint_duals_(
    *,
    rt: "MultimodalTrainingRuntime",
    data_cycle_loss: torch.Tensor,
    prior_cycle_loss: torch.Tensor,
) -> None:
    constraint_method = rt.tc.chart_transport_config.loss_config.constraint_config.constraint_method
    if not isinstance(constraint_method, LagrangianConstraintConfig):
        return

    data_budget = _constraint_budget(
        budget_per_dim=constraint_method.data_constraint_budget_per_dim,
        numel=rt.runtime_data_config.data_numel(),
    )
    prior_budget = _constraint_budget(
        budget_per_dim=constraint_method.prior_constraint_budget_per_dim,
        numel=rt.tc.chart_transport_config.prior_config.latent_numel(),
    )
    rt.data_dual = (
        rt.data_dual
        + constraint_method.dual_variable_lr
        * (data_cycle_loss.detach() - data_budget)
    ).clamp_min(0.0)
    rt.prior_dual = (
        rt.prior_dual
        + constraint_method.dual_variable_lr
        * (prior_cycle_loss.detach() - prior_budget)
    ).clamp_min(0.0)


def integrated_constraint_repair_step_(
    *,
    rt: "MultimodalTrainingRuntime",
) -> dict[str, float]:
    encoder = rt.chart_transport_model.encoder
    decoder = rt.chart_transport_model.decoder
    critic = rt.chart_transport_model.critic

    encoder.train()
    decoder.train()
    critic.eval()

    rt.optimizer.zero_grad(set_to_none=True)
    with _runtime_precision_context(rt=rt):
        losses = _compute_constraint_repair_losses(
            rt=rt,
            batch_size=rt.tc.train_batch_size,
        )
    rt.fabric.backward(losses["repair_loss"])
    rt.fabric.clip_gradients(
        rt.chart_transport_model,
        rt.optimizer,
        max_norm=rt.tc.chart_transport_config.architecture_config.grad_clip_norm,
        error_if_nonfinite=False,
    )
    rt.optimizer.step()
    _update_constraint_duals_(
        rt=rt,
        data_cycle_loss=losses["data_cycle_loss"],
        prior_cycle_loss=losses["prior_cycle_loss"],
    )
    metrics = _detach_metrics(losses=losses)
    metrics["data_dual"] = rt.data_dual.detach().item()
    metrics["prior_dual"] = rt.prior_dual.detach().item()
    return metrics


def _compute_transport_losses(
    *,
    rt: "MultimodalTrainingRuntime",
    batch_size: int,
) -> dict[str, torch.Tensor]:
    transport_config = rt.tc.chart_transport_config.loss_config.transport_config
    prior_config = rt.tc.chart_transport_config.prior_config

    data_batch = rt.runtime_data_config.sample_unconditional(
        batch_size=batch_size,
    )
    data_latents = rt.chart_transport_model.encoder(data_batch)

    with torch.no_grad():
        transported_latents, transport_field, transport_field_norm = (
            estimate_transport_targets(
                critic=rt.chart_transport_model.critic,
                prior_config=prior_config,
                transport_config=transport_config,
                clean_latents=data_latents,
            )
        )
        generated_prior_batch = prior_config.sample(
            batch_size=batch_size,
        ).to(device=rt.device, dtype=torch.float32)
        generated_samples = rt.chart_transport_model.decoder(generated_prior_batch)
        avg_generated_log_likelihood = rt.runtime_data_config.log_likelihood(
            generated_samples.float(),
        ).mean()

    encoder_transport_loss = F.mse_loss(
        data_latents,
        transported_latents,
    )
    decoder_transport_loss = F.mse_loss(
        rt.chart_transport_model.decoder(transported_latents),
        data_batch.detach(),
    )
    transport_loss = (
        transport_config.encoder_transport_weight * encoder_transport_loss
        + transport_config.decoder_transport_weight * decoder_transport_loss
    )
    return {
        "transport_loss": transport_loss,
        "encoder_transport_loss": encoder_transport_loss,
        "decoder_transport_loss": decoder_transport_loss,
        "transport_field_norm": transport_field_norm.mean(),
        "avg_generated_log_likelihood": avg_generated_log_likelihood,
    }


def integrated_transport_step_(
    *,
    rt: "MultimodalTrainingRuntime",
) -> dict[str, float]:
    encoder = rt.chart_transport_model.encoder
    decoder = rt.chart_transport_model.decoder
    critic = rt.chart_transport_model.critic

    encoder.train()
    decoder.train()
    critic.eval()

    rt.optimizer.zero_grad(set_to_none=True)
    with _runtime_precision_context(rt=rt):
        losses = _compute_transport_losses(
            rt=rt,
            batch_size=rt.tc.train_batch_size,
        )
    rt.fabric.backward(losses["transport_loss"])
    rt.fabric.clip_gradients(
        rt.chart_transport_model,
        rt.optimizer,
        max_norm=rt.tc.chart_transport_config.architecture_config.grad_clip_norm,
        error_if_nonfinite=False,
    )
    rt.optimizer.step()
    return _detach_metrics(losses=losses)


def _average_metrics(
    *,
    metrics_sequence: list[dict[str, float]],
) -> dict[str, float]:
    if len(metrics_sequence) == 0:
        raise ValueError("metrics_sequence must be non-empty")
    keys = metrics_sequence[0].keys()
    return {
        key: sum(metrics[key] for metrics in metrics_sequence) / len(metrics_sequence)
        for key in keys
    }


def _num_critic_updates_every_transport_step(
    *,
    rt: "MultimodalTrainingRuntime",
) -> int:
    n_updates = (
        rt.tc.chart_transport_config.scheduling_config.n_critic_updates_every_transport_step
    )
    if n_updates <= 0:
        raise ValueError(
            "n_critic_updates_every_transport_step must be positive during integrated training"
        )
    return n_updates


def integrated_train_step_(
    *,
    rt: "MultimodalTrainingRuntime",
) -> dict[str, float]:
    repair_metrics = rt._integrated_constraint_repair_step()
    critic_metrics = _average_metrics(
        metrics_sequence=[
            rt._critic_pretrain_train_step()
            for _ in range(_num_critic_updates_every_transport_step(rt=rt))
        ]
    )
    transport_metrics = rt._integrated_transport_step()
    return {
        **critic_metrics,
        **repair_metrics,
        **transport_metrics,
    }


def integrated_eval_step_(
    *,
    rt: "MultimodalTrainingRuntime",
    step: int,
) -> dict[str, float]:
    encoder = rt.chart_transport_model.encoder
    decoder = rt.chart_transport_model.decoder
    critic = rt.chart_transport_model.critic

    encoder_was_training = encoder.training
    decoder_was_training = decoder.training
    critic_was_training = critic.training
    encoder.eval()
    decoder.eval()
    critic.eval()

    with torch.no_grad():
        with _runtime_precision_context(rt=rt):
            monitor_metrics = rt.tc.monitor_config.constraint_monitor_config.apply_to(
                rt=rt,
                step=step,
            )
            monitor_metrics.update(
                rt.tc.monitor_config.critic_monitor_config.apply_to(
                    rt=rt,
                    step=step,
                    stage="integrated",
                )
            )
            monitor_metrics.update(
                rt.tc.monitor_config.conditioning_monitor_config.apply_to(
                    rt=rt,
                    step=step,
                )
            )
            monitor_metrics.update(
                rt.tc.monitor_config.sampling_monitor_config.apply_to(
                    rt=rt,
                    step=step,
                )
            )

    if encoder_was_training:
        encoder.train()
    if decoder_was_training:
        decoder.train()
    if critic_was_training:
        critic.train()

    return monitor_metrics


def integrated_(
    *,
    rt: "MultimodalTrainingRuntime",
) -> dict[str, float]:
    total_steps = rt.tc.integrated_n_steps
    log_every_n_steps = rt.tc.monitor_config.schedule_config.log_every_n_steps_integrated

    latest_metrics: dict[str, float] = {}
    progress = tqdm(
        range(1, total_steps + 1),
        desc="integrated",
    )
    for step in progress:
        train_metrics = rt._integrated_train_step()
        latest_metrics = {f"train_{key}": value for key, value in train_metrics.items()}
        _log_wandb_scalars(
            rt=rt,
            stage="integrated",
            step=step,
            metrics=train_metrics,
        )
        progress.set_postfix(
            critic_loss=f"{train_metrics['critic_loss']:.4f}",
            repair_loss=f"{train_metrics['repair_loss']:.4f}",
            transport_loss=f"{train_metrics['transport_loss']:.4f}",
        )

        if _should_log_monitor(
            step=step,
            total_steps=total_steps,
            every_n_steps=log_every_n_steps,
        ):
            monitor_metrics = rt._integrated_eval_step(step=step)
            latest_metrics.update(
                {f"monitor_{key}": value for key, value in monitor_metrics.items()}
            )
            _log_wandb_scalars(
                rt=rt,
                stage="integrated",
                step=step,
                metrics={
                    f"monitor_{key}": value for key, value in monitor_metrics.items()
                },
            )
            rt.fabric.print(
                f"[integrated] step {step}/{total_steps}: "
                f"train: {_format_metrics_summary(metrics=train_metrics)}; "
                f"monitor: {_format_metrics_summary(metrics=monitor_metrics)}"
            )

    return latest_metrics


__all__ = [
    "integrated_",
    "integrated_constraint_repair_step_",
    "integrated_eval_step_",
    "integrated_train_step_",
    "integrated_transport_step_",
]
