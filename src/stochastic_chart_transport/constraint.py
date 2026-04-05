from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from src.common.training import BaseLoss
from src.config.base import BaseConfig

if TYPE_CHECKING:
    pass


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
        sample,
        reconstructed_sample,
    ) -> Loss:
        return self.Loss(
            value=F.huber_loss(reconstructed_sample, sample, delta=self.huber_delta),
            weight=self.weight,
        )


class LatentAnchorConfig(BaseConfig):
    """
    Penalizes the energy of a given latent representation
    """

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
            latent_norm=latent.pow(2).mean(), weight=self.latent_norm_weight
        )


class IntegratedChartConstraintConfig(BaseConfig):
    data_reconstruction: ReconstructionConfig
    data_latent_anchor: LatentAnchorConfig
    model_latent_anchor: LatentAnchorConfig

    @dataclass
    class Loss(BaseLoss):
        data_reconstruction: ReconstructionConfig.Loss
        data_latent_anchor: LatentAnchorConfig.Loss
        model_latent_anchor: LatentAnchorConfig.Loss

        def sum(self):
            return (
                self.data_reconstruction.sum()
                + self.data_latent_anchor.sum()
                + self.model_latent_anchor.sum()
            )

    def apply(
        self,
        state: ChartTransportStudyState,
        *,
        data: Float[Tensor, "batch ..."],
        data_latent: Float[Tensor, "batch ..."],
        model_latent: Float[Tensor, "batch ..."],
    ) -> IntegratedChartConstraintConfig.Loss:
        reconstructed_data_sample, _ = state.decode(data_latent)
        # Compute concrete losses
        data_reconstruction = self.data_reconstruction.apply(
            sample=data,
            reconstructed_sample=reconstructed_data_sample,
        )
        model_latent_anchor_loss = self.model_latent_anchor.apply(latent=model_latent)
        data_latent_anchor_loss = self.data_latent_anchor.apply(latent=data_latent)

        return self.Loss(
            data_reconstruction=data_reconstruction,
            model_latent_anchor=model_latent_anchor_loss,
            data_latent_anchor=data_latent_anchor_loss,
        )


class ChartPretrainConfig(BaseConfig):
    """
    During pretrain, we require:
    1. Encoded data fibers are separable. By "fiber" here, we mean the image of a
        single sample under the stochastic encoder.
        - decoder(encoder(data, fiber)) reconstructs data
    2. Data fibers well-span the space
        - decoder(encoder(data, fiber)) weakly reconstructs the fiber
    3. Model samples are reasonable
        - encoder(decoder_with_fiber(prior)) = prior
    4. Latents are anchored about zero
    """

    data_reconstruction: ReconstructionConfig
    data_fiber_reconstruction: ReconstructionConfig
    prior_reconstruction: ReconstructionConfig
    anchor_config: LatentAnchorConfig

    @dataclass
    class Loss(BaseLoss):
        data_reconstruction: ReconstructionConfig.Loss
        data_fiber_reconstruction: ReconstructionConfig.Loss
        prior_reconstruction: ReconstructionConfig.Loss
        anchor: LatentAnchorConfig.Loss

        def sum(self):
            return (
                self.data_fiber_reconstruction.sum()
                + self.data_reconstruction.sum()
                + self.prior_reconstruction.sum()
                + self.anchor.sum()
            )

    def apply(
        self,
        state: ChartTransportStudyState,
        *,
        data: Float[Tensor, "batch ..."],
    ) -> ChartPretrainConfig.Loss:
        batch_size = data.shape[0]
        prior = state.prior_config.sample(batch_size=batch_size).type_as(data)
        data_fiber = state.get_fiber(batch_size=batch_size).type_as(data)

        # Forward passes
        data_latent = state.encode(data=data, fiber=data_fiber)
        combined_sample, combined_fiber = state.decode(torch.cat([data_latent, prior]))
        reconstructed_data_sample, model_sample = combined_sample.chunk(2, dim=0)
        reconstructed_data_fiber, model_fiber = combined_fiber.chunk(2, dim=0)
        reconstructed_prior = state.encode(data=model_sample, fiber=model_fiber)

        # Compute concrete losses
        data_reconstruction = self.data_reconstruction.apply(
            sample=data,
            reconstructed_sample=reconstructed_data_sample,
        )
        data_fiber_reconstruction = self.data_fiber_reconstruction.apply(
            sample=data_fiber, reconstructed_sample=reconstructed_data_fiber
        )
        prior_reconstruction = self.prior_reconstruction.apply(
            sample=prior, reconstructed_sample=reconstructed_prior
        )
        anchor_loss = self.anchor_config.apply(latent=data_latent)

        return self.Loss(
            data_reconstruction=data_reconstruction,
            data_fiber_reconstruction=data_fiber_reconstruction,
            prior_reconstruction=prior_reconstruction,
            anchor=anchor_loss,
        )
