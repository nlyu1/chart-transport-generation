from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float
from pydantic import model_validator
from torch import Tensor

from src.common.training import BaseLoss
from src.config.base import BaseConfig
from src.utils import clip_norm


class GaussianKernelConfig(BaseConfig):
    width: float
    min_density: float = 1e-8

    @model_validator(mode="after")
    def _validate_config(self) -> "GaussianKernelConfig":
        if self.width <= 0.0:
            raise ValueError("width must be positive")
        if self.min_density <= 0.0:
            raise ValueError("min_density must be positive")
        return self

    def evaluate(
        self,
        *,
        query_points: Float[Tensor, "query 2"],
        reference_points: Float[Tensor, "reference 2"],
    ) -> tuple[
        Float[Tensor, "query reference"],
        Float[Tensor, "query reference 2"],
    ]:
        offsets = reference_points.unsqueeze(0) - query_points.unsqueeze(1)
        squared_distances = offsets.square().sum(dim=-1)
        weights = torch.exp(-0.5 * squared_distances / (self.width**2))
        return weights, offsets


class ReverseKLDriftingLossConfig(BaseConfig):
    kernel: GaussianKernelConfig
    transport_step_multiplier: float
    transport_step_cap: float
    weight: float = 1.0

    @dataclass
    class Loss(BaseLoss):
        transport: Float[Tensor, ""]
        weight: float

        def sum(self):
            return self.transport * self.weight

    @dataclass
    class FieldEstimate:
        attraction: Float[Tensor, "query 2"]
        repulsion: Float[Tensor, "query 2"]
        reverse_kl: Float[Tensor, "query 2"]
        data_density: Float[Tensor, "query"]
        model_density: Float[Tensor, "query"]

    @model_validator(mode="after")
    def _validate_config(self) -> "ReverseKLDriftingLossConfig":
        if self.transport_step_multiplier <= 0.0:
            raise ValueError("transport_step_multiplier must be positive")
        if self.transport_step_cap <= 0.0:
            raise ValueError("transport_step_cap must be positive")
        if self.weight <= 0.0:
            raise ValueError("weight must be positive")
        return self

    def _estimate_normalized_field(
        self,
        *,
        query_points: Float[Tensor, "query 2"],
        reference_points: Float[Tensor, "reference 2"],
    ) -> tuple[
        Float[Tensor, "query 2"],
        Float[Tensor, "query"],
    ]:
        weights, offsets = self.kernel.evaluate(
            query_points=query_points,
            reference_points=reference_points,
        )
        density = weights.sum(dim=1).clamp_min(self.kernel.min_density)
        field = (weights.unsqueeze(-1) * offsets).sum(dim=1) / density.unsqueeze(-1)
        return field, density

    def estimate_reverse_kl_field(
        self,
        *,
        query_points: Float[Tensor, "query 2"],
        data_samples: Float[Tensor, "data 2"],
        model_samples: Float[Tensor, "model 2"],
    ) -> FieldEstimate:
        attraction, data_density = self._estimate_normalized_field(
            query_points=query_points,
            reference_points=data_samples,
        )
        repulsion, model_density = self._estimate_normalized_field(
            query_points=query_points,
            reference_points=model_samples,
        )
        reverse_kl = attraction - repulsion
        return self.FieldEstimate(
            attraction=attraction,
            repulsion=repulsion,
            reverse_kl=reverse_kl,
            data_density=data_density,
            model_density=model_density,
        )

    def scale_clip_transport_field(
        self,
        field: Float[Tensor, "batch 2"],
    ) -> Float[Tensor, "batch 2"]:
        return clip_norm(
            field * self.transport_step_multiplier,
            max_norm=self.transport_step_cap,
        )

    def apply(
        self,
        state,
        *,
        data_samples: Float[Tensor, "batch 2"],
        latent: Float[Tensor, "batch 2"],
    ) -> Loss:
        model_samples = state.decode(latent)
        with torch.no_grad():
            field_estimate = self.estimate_reverse_kl_field(
                query_points=model_samples,
                data_samples=data_samples,
                model_samples=model_samples.detach(),
            )
            transported_model_samples = (
                model_samples
                + self.scale_clip_transport_field(field_estimate.reverse_kl)
            ).detach()
        per_sample_squared_error = (model_samples - transported_model_samples).square()
        return self.Loss(
            transport=per_sample_squared_error.sum(dim=-1).mean(),
            weight=self.weight,
        )


__all__ = [
    "GaussianKernelConfig",
    "ReverseKLDriftingLossConfig",
]
