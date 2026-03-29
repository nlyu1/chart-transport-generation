from __future__ import annotations

from jaxtyping import Float
import torch
import torch.nn as nn
from torch import Tensor

from src.chart_transport.transport_loss import TransportLossConfig
from src.priors.base import BasePriorConfig


def sample_transport_times(
    *,
    transport_config: TransportLossConfig,
    device: torch.device,
    batch_shape: tuple[int, ...],
) -> Float[Tensor, "..."]:
    t_min, t_max = transport_config.t_range
    return t_min + (t_max - t_min) * torch.rand(
        *batch_shape,
        device=device,
        dtype=torch.float32,
    )


def sample_stratified_transport_times(
    *,
    transport_config: TransportLossConfig,
    device: torch.device,
    batch_size: int,
) -> Float[Tensor, "batch num_time_samples"]:
    bin_edges = torch.linspace(
        transport_config.t_range[0],
        transport_config.t_range[1],
        transport_config.num_time_samples + 1,
        device=device,
        dtype=torch.float32,
    )
    bin_starts = bin_edges[:-1]
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    return bin_starts.unsqueeze(0) + bin_widths.unsqueeze(0) * torch.rand(
        batch_size,
        transport_config.num_time_samples,
        device=device,
        dtype=torch.float32,
    )


def critic_score_from_noise_prediction(
    *,
    predicted_noise: Float[Tensor, "batch ..."],
    t: Float[Tensor, "batch"],
) -> Float[Tensor, "batch ..."]:
    return -predicted_noise / t.unsqueeze(-1)


def estimate_transport_targets(
    *,
    critic: nn.Module,
    prior_config: BasePriorConfig,
    transport_config: TransportLossConfig,
    clean_latents: Float[Tensor, "batch latent_dim"],
) -> tuple[
    Float[Tensor, "batch latent_dim"],
    Float[Tensor, "batch latent_dim"],
    Float[Tensor, "batch 1"],
]:
    transport_source_latents = clean_latents.detach()
    transport_t = sample_stratified_transport_times(
        transport_config=transport_config,
        device=transport_source_latents.device,
        batch_size=transport_source_latents.shape[0],
    )
    transport_source_latents = transport_source_latents.unsqueeze(1).expand(
        -1,
        transport_config.num_time_samples,
        -1,
    )
    transport_eps = torch.randn(
        transport_source_latents.shape[0],
        transport_source_latents.shape[1],
        transport_source_latents.shape[-1],
        device=transport_source_latents.device,
        dtype=transport_source_latents.dtype,
    )

    if transport_config.antipodal_estimate:
        transport_t = torch.cat([transport_t, transport_t], dim=1)
        transport_source_latents = transport_source_latents.repeat(1, 2, 1)
        transport_eps = torch.cat([transport_eps, -transport_eps], dim=1)

    transport_noised_latents = (
        (1.0 - transport_t).unsqueeze(-1) * transport_source_latents
        + transport_t.unsqueeze(-1) * transport_eps
    )
    flat_transport_noised_latents = transport_noised_latents.reshape(
        -1,
        transport_noised_latents.shape[-1],
    )
    flat_transport_t = transport_t.reshape(-1)

    transport_predicted_noise = critic(
        flat_transport_noised_latents,
        flat_transport_t,
    ).reshape_as(transport_noised_latents)
    transport_prior_score = prior_config.analytic_score(
        flat_transport_noised_latents.float(),
        flat_transport_t.float(),
    ).to(dtype=flat_transport_noised_latents.dtype).reshape_as(
        transport_noised_latents
    )
    transport_pullback_weight = (
        transport_config.kl_weight_schedule.pullback_weight(
            flat_transport_t.float(),
        )
        .to(dtype=flat_transport_noised_latents.dtype)
        .reshape_as(transport_t)
    )
    transport_field_terms = transport_pullback_weight.unsqueeze(-1) * (
        transport_prior_score + transport_predicted_noise / transport_t.unsqueeze(-1)
    )

    if transport_config.antipodal_estimate:
        midpoint = transport_config.num_time_samples
        transport_field_terms = 0.5 * (
            transport_field_terms[:, :midpoint]
            + transport_field_terms[:, midpoint:]
        )

    transport_field = transport_field_terms.mean(dim=1)
    transport_field_norm = transport_field.norm(dim=-1, keepdim=True)
    transport_step_size = torch.minimum(
        torch.full_like(
            transport_field_norm,
            transport_config.transport_step_size,
        ),
        torch.full_like(
            transport_field_norm,
            transport_config.transport_step_cap,
        )
        / transport_field_norm.clamp_min(1e-6),
    )
    transport_step = transport_step_size * transport_field
    transported_latents = clean_latents.detach() + transport_step
    return transported_latents, transport_field, transport_field_norm


__all__ = [
    "critic_score_from_noise_prediction",
    "estimate_transport_targets",
    "sample_stratified_transport_times",
    "sample_transport_times",
]
