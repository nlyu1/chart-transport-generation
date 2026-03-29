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
from src.experiments.multimodal_gaussian.common import (
    compute_cycle_losses,
    detach_metrics,
    format_metrics_summary,
    log_wandb_scalars,
    optimizer_step_,
    preserve_module_train_states,
    runtime_precision_context,
)
from src.monitoring.utils import MonitorStage

if TYPE_CHECKING:
    from src.experiments.multimodal_gaussian.state import MultimodalTrainingRuntime


_INTEGRATED_TRAIN_METRIC_NAMES = {
    "critic_loss": "critic",
    "repair_loss": "repair",
    "data_cycle_loss": "data_cycle",
    "prior_cycle_loss": "prior_cycle",
    "data_dual": "data_dual",
    "prior_dual": "prior_dual",
    "transport_loss": "transport",
    "encoder_transport_loss": "enc_transport",
    "decoder_transport_loss": "dec_transport",
    "transport_field_norm": "field",
}

_INTEGRATED_MONITOR_METRIC_NAMES = {
    "constraint_reconstruction_mean": "recon_err",
    "constraint_latent_norm_mean": "latent_norm",
    "critic_monitor_snapshot_score_norm_mean": "score",
    "critic_monitor_transport_norm_mean": "field",
    "encoder_conditioning_mean": "enc_cond",
    "decoder_conditioning_mean": "dec_cond",
}


def _select_metrics(
    *,
    metrics: dict[str, float],
    selected_names: dict[str, str],
) -> dict[str, float]:
    missing = [key for key in selected_names if key not in metrics]
    if len(missing) > 0:
        raise KeyError(f"Missing metrics: {missing}")
    return {
        metric_name: metrics[key]
        for key, metric_name in selected_names.items()
    }


def _select_present_metrics(
    *,
    metrics: dict[str, float],
    selected_names: dict[str, str],
) -> dict[str, float]:
    return {
        metric_name: metrics[key]
        for key, metric_name in selected_names.items()
        if key in metrics
    }


def _select_present_monitor_metrics(
    *,
    metrics: dict[str, float],
) -> dict[str, float]:
    selected_metrics = _select_present_metrics(
        metrics=metrics,
        selected_names=_INTEGRATED_MONITOR_METRIC_NAMES,
    )
    selected_metrics.update(
        {
            f"kl_{key.removeprefix('projected_kl_mean_')}": value
            for key, value in metrics.items()
            if key.startswith("projected_kl_mean_")
        }
    )
    return selected_metrics


def _constraint_budget(
    *,
    budget_per_dim: float,
    numel: int,
) -> float:
    return budget_per_dim * (numel**0.5)


def _compute_constraint_repair_losses(
    *,
    rt: "MultimodalTrainingRuntime",
    batch_size: int,
) -> dict[str, torch.Tensor]:
    constraint_method = rt.tc.chart_transport_config.loss_config.constraint_config.constraint_method
    cycle_losses = compute_cycle_losses(
        rt=rt,
        batch_size=batch_size,
    )

    if isinstance(constraint_method, LossConstraintConfig):
        repair_loss = (
            constraint_method.data_loss_weight * cycle_losses.data_cycle_loss
            + constraint_method.prior_loss_weight * cycle_losses.prior_cycle_loss
        )
        return {
            "repair_loss": repair_loss,
            "data_cycle_loss": cycle_losses.data_cycle_loss,
            "prior_cycle_loss": cycle_losses.prior_cycle_loss,
        }

    data_budget = _constraint_budget(
        budget_per_dim=constraint_method.data_constraint_budget_per_dim,
        numel=rt.runtime_data_config.data_numel(),
    )
    prior_budget = _constraint_budget(
        budget_per_dim=constraint_method.prior_constraint_budget_per_dim,
        numel=rt.tc.chart_transport_config.prior_config.latent_numel(),
    )
    repair_loss = cycle_losses.data_cycle_loss + cycle_losses.prior_cycle_loss
    repair_loss = repair_loss + rt.data_dual * (
        cycle_losses.data_cycle_loss - data_budget
    )
    repair_loss = repair_loss + rt.prior_dual * (
        cycle_losses.prior_cycle_loss - prior_budget
    )
    return {
        "repair_loss": repair_loss,
        "data_cycle_loss": cycle_losses.data_cycle_loss,
        "prior_cycle_loss": cycle_losses.prior_cycle_loss,
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

    with runtime_precision_context(rt=rt):
        losses = _compute_constraint_repair_losses(
            rt=rt,
            batch_size=rt.tc.train_batch_size,
        )
    optimizer_step_(
        rt=rt,
        loss=losses["repair_loss"],
    )
    _update_constraint_duals_(
        rt=rt,
        data_cycle_loss=losses["data_cycle_loss"],
        prior_cycle_loss=losses["prior_cycle_loss"],
    )
    metrics = detach_metrics(losses=losses)
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

    with runtime_precision_context(rt=rt):
        losses = _compute_transport_losses(
            rt=rt,
            batch_size=rt.tc.train_batch_size,
        )
    optimizer_step_(
        rt=rt,
        loss=losses["transport_loss"],
    )
    return detach_metrics(losses=losses)


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
    run_constraint_monitor: bool,
    run_critic_monitor: bool,
    run_conditioning_monitor: bool,
    run_sampling_monitor: bool,
) -> dict[str, float]:
    encoder = rt.chart_transport_model.encoder
    decoder = rt.chart_transport_model.decoder
    critic = rt.chart_transport_model.critic

    with preserve_module_train_states(modules=[encoder, decoder, critic]):
        encoder.eval()
        decoder.eval()
        critic.eval()

        with torch.no_grad():
            with runtime_precision_context(rt=rt):
                raw_monitor_metrics: dict[str, float] = {}
                if run_constraint_monitor:
                    raw_monitor_metrics.update(
                        rt.tc.monitor_config.constraint_monitor_config.apply_to(
                            rt=rt,
                            step=step,
                            stage=MonitorStage.INTEGRATED,
                        )
                    )
                if run_critic_monitor:
                    raw_monitor_metrics.update(
                        rt.tc.monitor_config.critic_monitor_config.apply_to(
                            rt=rt,
                            step=step,
                            stage=MonitorStage.INTEGRATED,
                        )
                    )
                if run_conditioning_monitor:
                    raw_monitor_metrics.update(
                        rt.tc.monitor_config.conditioning_monitor_config.apply_to(
                            rt=rt,
                            step=step,
                            stage=MonitorStage.INTEGRATED,
                        )
                    )
                if run_sampling_monitor:
                    raw_monitor_metrics.update(
                        rt.tc.monitor_config.sampling_monitor_config.apply_to(
                            rt=rt,
                            step=step,
                            stage=MonitorStage.INTEGRATED,
                        )
                    )

    return _select_present_monitor_metrics(metrics=raw_monitor_metrics)


def integrated_(
    *,
    rt: "MultimodalTrainingRuntime",
) -> dict[str, float]:
    total_steps = rt.tc.integrated_n_steps
    monitor_config = rt.tc.monitor_config

    latest_metrics: dict[str, float] = {}
    progress = tqdm(
        range(1, total_steps + 1),
        desc="integrated",
    )
    for step in progress:
        raw_train_metrics = rt._integrated_train_step()
        train_metrics = _select_metrics(
            metrics=raw_train_metrics,
            selected_names=_INTEGRATED_TRAIN_METRIC_NAMES,
        )
        latest_metrics = {f"train_{key}": value for key, value in train_metrics.items()}
        log_wandb_scalars(
            rt=rt,
            stage="integrated",
            step=step,
            metrics=train_metrics,
        )
        progress.set_postfix(
            critic=f"{train_metrics['critic']:.4f}",
            repair=f"{train_metrics['repair']:.4f}",
            transport=f"{train_metrics['transport']:.4f}",
        )

        if monitor_config.should_run_stage(
            stage=MonitorStage.INTEGRATED,
            step=step,
            total_steps=total_steps,
        ):
            force_stage = monitor_config.should_force_stage(
                stage=MonitorStage.INTEGRATED,
                step=step,
                total_steps=total_steps,
            )
            monitor_metrics = rt._integrated_eval_step(
                step=step,
                run_constraint_monitor=monitor_config.should_activate_component(
                    component_config=monitor_config.constraint_monitor_config,
                    stage=MonitorStage.INTEGRATED,
                    step=step,
                    total_steps=total_steps,
                    force=force_stage,
                ),
                run_critic_monitor=monitor_config.should_activate_component(
                    component_config=monitor_config.critic_monitor_config,
                    stage=MonitorStage.INTEGRATED,
                    step=step,
                    total_steps=total_steps,
                    force=force_stage,
                ),
                run_conditioning_monitor=monitor_config.should_activate_component(
                    component_config=monitor_config.conditioning_monitor_config,
                    stage=MonitorStage.INTEGRATED,
                    step=step,
                    total_steps=total_steps,
                    force=force_stage,
                ),
                run_sampling_monitor=monitor_config.should_activate_component(
                    component_config=monitor_config.sampling_monitor_config,
                    stage=MonitorStage.INTEGRATED,
                    step=step,
                    total_steps=total_steps,
                    force=force_stage,
                ),
            )
            latest_metrics.update(
                {f"monitor_{key}": value for key, value in monitor_metrics.items()}
            )
            log_wandb_scalars(
                rt=rt,
                stage="integrated",
                step=step,
                metrics={
                    f"monitor_{key}": value for key, value in monitor_metrics.items()
                },
            )
            rt.fabric.print(
                f"[integrated] step {step}/{total_steps}: "
                f"train: {format_metrics_summary(metrics=train_metrics)}; "
                f"monitor: {format_metrics_summary(metrics=monitor_metrics)}"
            )

    return latest_metrics


__all__ = [
    "integrated_",
    "integrated_constraint_repair_step_",
    "integrated_eval_step_",
    "integrated_train_step_",
    "integrated_transport_step_",
]
