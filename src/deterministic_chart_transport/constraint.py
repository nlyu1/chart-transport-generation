from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from src.common.training import BaseLoss
from src.config.base import BaseConfig

if TYPE_CHECKING:
    from src.deterministic_chart_transport.study import (
        DeterministicChartTransportStudyState,
    )


class ReconstructionConfig(BaseConfig):
    huber_delta: float
    weight: float

    @dataclass
    class Loss(BaseLoss):
        value: Float[Tensor, ""]
        weight: float

        def sum(self):
            return self.value * self.weight

    def apply(
        self,
        *,
        sample: Float[Tensor, "batch ..."],
        reconstructed_sample: Float[Tensor, "batch ..."],
    ) -> Loss:
        return self.Loss(
            value=F.huber_loss(
                reconstructed_sample,
                sample,
                delta=self.huber_delta,
            ),
            weight=self.weight,
        )


class LatentScaleAnchorConfig(BaseConfig):
    latent_norm_weight: float
    latent_zero_mean_weight: float
    target_norm_per_dimension: float

    @dataclass
    class Loss(BaseLoss):
        latent_norm: Float[Tensor, ""]
        latent_zero_mean: Float[Tensor, ""]
        latent_norm_weight: float
        latent_zero_mean_weight: float

        def sum(self):
            return (
                self.latent_norm * self.latent_norm_weight
                + self.latent_zero_mean * self.latent_zero_mean_weight
            )

    def apply(
        self,
        *,
        latent: Float[Tensor, "batch ..."],
    ) -> Loss:
        flat_latent = latent.reshape(latent.shape[0], -1)
        per_dimension_second_moment = flat_latent.pow(2).mean(dim=0)
        return self.Loss(
            latent_norm=(
                per_dimension_second_moment - self.target_norm_per_dimension**2
            )
            .pow(2)
            .mean(),
            latent_zero_mean=flat_latent.mean(dim=0).pow(2).mean(),
            latent_norm_weight=self.latent_norm_weight,
            latent_zero_mean_weight=self.latent_zero_mean_weight,
        )


class LatentNormAnchorConfig(BaseConfig):
    latent_norm_weight: float

    @dataclass
    class Loss(BaseLoss):
        latent_norm: Float[Tensor, ""]
        weight: float

        def sum(self):
            return self.latent_norm * self.weight

    def apply(
        self,
        *,
        latent: Float[Tensor, "batch ..."],
    ) -> Loss:
        return self.Loss(
            latent_norm=latent.pow(2).mean(),
            weight=self.latent_norm_weight,
        )


class ChartPretrainConfig(BaseConfig):
    data_reconstruction: ReconstructionConfig
    prior_roundtrip: ReconstructionConfig
    anchor_config: LatentNormAnchorConfig

    @dataclass
    class Loss(BaseLoss):
        data_reconstruction: ReconstructionConfig.Loss
        prior_roundtrip: ReconstructionConfig.Loss
        anchor: LatentNormAnchorConfig.Loss

        def sum(self):
            return (
                self.data_reconstruction.sum()
                + self.prior_roundtrip.sum()
                + self.anchor.sum()
            )

    def apply(
        self,
        state: DeterministicChartTransportStudyState,
        *,
        data: Float[Tensor, "batch ..."],
        data_latent: Float[Tensor, "batch ..."],
        model_sample: Float[Tensor, "batch ..."],
        prior: Float[Tensor, "batch ..."],
    ) -> Loss:
        reconstructed_data_sample = state.decode(latent=data_latent)
        reconstructed_prior = state.encode(data=model_sample)
        return self.Loss(
            data_reconstruction=self.data_reconstruction.apply(
                sample=data,
                reconstructed_sample=reconstructed_data_sample,
            ),
            prior_roundtrip=self.prior_roundtrip.apply(
                sample=prior,
                reconstructed_sample=reconstructed_prior,
            ),
            anchor=self.anchor_config.apply(latent=data_latent),
        )


class IntegratedChartConstraintConfig(BaseConfig):
    data_reconstruction: ReconstructionConfig
    prior_roundtrip: ReconstructionConfig
    data_latent_anchor: LatentScaleAnchorConfig

    @dataclass
    class Loss(BaseLoss):
        data_reconstruction: ReconstructionConfig.Loss
        prior_roundtrip: ReconstructionConfig.Loss
        data_latent_anchor: LatentScaleAnchorConfig.Loss

        def sum(self):
            return (
                self.data_reconstruction.sum()
                + self.prior_roundtrip.sum()
                + self.data_latent_anchor.sum()
            )

    def apply(
        self,
        state: DeterministicChartTransportStudyState,
        *,
        data: Float[Tensor, "batch ..."],
        data_latent: Float[Tensor, "batch ..."],
        model_sample: Float[Tensor, "batch ..."],
        prior: Float[Tensor, "batch ..."],
    ) -> Loss:
        reconstructed_data_sample = state.decode(latent=data_latent)
        reconstructed_prior = state.encode(data=model_sample)
        return self.Loss(
            data_reconstruction=self.data_reconstruction.apply(
                sample=data,
                reconstructed_sample=reconstructed_data_sample,
            ),
            prior_roundtrip=self.prior_roundtrip.apply(
                sample=prior,
                reconstructed_sample=reconstructed_prior,
            ),
            data_latent_anchor=self.data_latent_anchor.apply(latent=data_latent),
        )


__all__ = [
    "ChartPretrainConfig",
    "IntegratedChartConstraintConfig",
    "LatentNormAnchorConfig",
    "LatentScaleAnchorConfig",
    "ReconstructionConfig",
]
