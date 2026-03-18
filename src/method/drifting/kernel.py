from __future__ import annotations

from typing import Literal

import torch
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict
from torch import Tensor


class GaussianDriftingStatistics(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    data_density: Tensor
    model_density: Tensor
    data_field: Tensor
    model_field: Tensor
    canonical_field: Tensor
    density_ratio: Tensor
    drift: Tensor


def gaussian_kernel_weights(
    *,
    query: Float[Tensor, "query dim"],
    reference: Float[Tensor, "reference dim"],
    bandwidth: float,
) -> Float[Tensor, "query reference"]:
    # Gaussian KDE weights from pairwise squared distances.
    squared_distance = torch.cdist(query, reference).square()
    return torch.exp(-squared_distance / (2.0 * bandwidth * bandwidth))


def compute_normalized_kernel_field(
    *,
    query: Float[Tensor, "query dim"],
    reference: Float[Tensor, "reference dim"],
    bandwidth: float,
    stability_eps: float,
    exclude_diagonal: bool,
) -> tuple[
    Float[Tensor, "query 1"],
    Float[Tensor, "query dim"],
]:
    weights = gaussian_kernel_weights(
        query=query,
        reference=reference,
        bandwidth=bandwidth,
    )
    deltas = reference.unsqueeze(0) - query.unsqueeze(1)

    if exclude_diagonal:
        mask = ~torch.eye(
            query.shape[0],
            device=query.device,
            dtype=torch.bool,
        )
        weights = weights.masked_fill(~mask, 0.0)
        sample_count = mask.sum(dim=-1, keepdim=True).to(dtype=query.dtype)
    else:
        sample_count = torch.full(
            (query.shape[0], 1),
            fill_value=reference.shape[0],
            device=query.device,
            dtype=query.dtype,
        )

    density = weights.sum(dim=-1, keepdim=True) / sample_count
    numerator = (weights.unsqueeze(-1) * deltas).sum(dim=1) / sample_count
    field = numerator / density.clamp_min(stability_eps)
    return density, field


def compute_gaussian_drifting_statistics(
    *,
    model_samples: Float[Tensor, "batch dim"],
    data_samples: Float[Tensor, "data_batch dim"],
    bandwidth: float,
    objective: Literal["reverse_kl", "forward_kl"],
    stability_eps: float,
    exclude_self_interactions: bool,
) -> GaussianDriftingStatistics:
    data_density, data_field = compute_normalized_kernel_field(
        query=model_samples,
        reference=data_samples,
        bandwidth=bandwidth,
        stability_eps=stability_eps,
        exclude_diagonal=False,
    )
    model_density, model_field = compute_normalized_kernel_field(
        query=model_samples,
        reference=model_samples,
        bandwidth=bandwidth,
        stability_eps=stability_eps,
        exclude_diagonal=exclude_self_interactions,
    )
    canonical_field = data_field - model_field
    if objective == "forward_kl":
        density_ratio = data_density / model_density.clamp_min(stability_eps)
    else:
        density_ratio = torch.ones_like(data_density)
    drift = density_ratio * canonical_field
    return GaussianDriftingStatistics(
        data_density=data_density,
        model_density=model_density,
        data_field=data_field,
        model_field=model_field,
        canonical_field=canonical_field,
        density_ratio=density_ratio,
        drift=drift,
    )
