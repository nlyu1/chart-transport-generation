from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from src.common.training import BaseLoss
from src.config.base import BaseConfig

if TYPE_CHECKING:
    pass


class ReconstructionLossConfig(BaseConfig):
    huber_delta: float
    data_reconstruction_weight: float
    stochastic_reconstruction_divider: float
    prior_reconstruction_weight: float

    @dataclass
    class Loss(BaseLoss):
        data: Float[Tensor, ""]
        fiber: Float[Tensor, ""]
        prior: Float[Tensor, ""]

        def sum(self):
            return self.data + self.fiber + self.prior

    def apply(
        self,
        *,
        data,
        fiber,
        prior,
        reconstructed_data,
        reconstructed_fiber,
        reconstructed_prior,
    ) -> Loss:
        data_reconstruction_loss = (
            F.huber_loss(reconstructed_data, data, delta=self.huber_delta)
            * self.data_reconstruction_weight
        )

        fiber_reconstruction_loss = F.huber_loss(
            reconstructed_fiber, fiber, delta=self.huber_delta
        ) * (self.data_reconstruction_weight / self.stochastic_reconstruction_divider)

        prior_reconstruction_loss = (
            F.huber_loss(reconstructed_prior, prior, delta=self.huber_delta)
            * self.prior_reconstruction_weight
        )

        return self.Loss(
            data=data_reconstruction_loss,
            fiber=fiber_reconstruction_loss,
            prior=prior_reconstruction_loss,
        )


class AnchorLossConfig(BaseConfig):
    latent_norm_weight: float
    latent_zero_mean_weight: float

    @dataclass
    class Loss(BaseLoss):
        data_norm: Float[Tensor, ""]
        data_mean: Float[Tensor, ""]

        def sum(self):
            return self.data_norm + self.data_mean

    def apply(
        self,
        *,
        data_latent: Float[Tensor, "batch ..."],
    ) -> Loss:
        return self.Loss(
            data_norm=self.latent_norm_weight * data_latent.pow(2).mean(),
            data_mean=self.latent_zero_mean_weight * data_latent.mean(0).pow(2).mean(),
        )


class ChartPretrainConfig(BaseConfig):
    reconstruction_config: ReconstructionLossConfig
    anchor_config: AnchorLossConfig

    @dataclass
    class Loss(BaseLoss):
        reconstruction_loss: ReconstructionLossConfig.Loss
        anchor_loss: AnchorLossConfig.Loss | None

        def sum(self):
            if self.anchor_loss is None:
                return self.reconstruction_loss.sum()
            return self.reconstruction_loss.sum() + self.anchor_loss.sum()

    def apply(
        self,
        state: ChartTransportStudyState,
        *,
        data: Float[Tensor, "batch ..."],
        data_fiber: Float[Tensor, "batch ..."],
        data_latent: Float[Tensor, "batch ..."],
        model_sample: Float[Tensor, "batch ..."],
        model_fiber: Float[Tensor, "batch ..."],
        prior: Float[Tensor, "batch ..."],
        compute_anchor_loss: bool,
    ) -> ChartPretrainConfig.Loss:
        """
        Model forward-pass computations:
        1. decoder(data latent) -> (reconstructed_data, reconstructed_data_fiber)

        To re-use computation, **it is the caller's responsibility** to ensure:
        1. encoder(data, data_fiber) -> data_latent
        2. decoder(prior) -> (model_sample, model_fiber)

        These computations should be done with full gradients attached
        """
        model = state.model

        reconstructed_data, reconstructed_data_fiber = state.unpack_fiber(
            model.decoder(data_latent)
        )
        reconstructed_prior = model.encoder(state.pack_fiber(model_sample, model_fiber))

        reconstruction_loss = self.reconstruction_config.apply(
            data=data,
            fiber=data_fiber,
            reconstructed_data=reconstructed_data,
            reconstructed_fiber=reconstructed_data_fiber,
            prior=prior,
            reconstructed_prior=reconstructed_prior,
        )

        anchor_loss: AnchorLossConfig.Loss | None = None
        if compute_anchor_loss:
            anchor_loss = self.anchor_config.apply(
                data_latent=data_latent,
            )

        return self.Loss(
            reconstruction_loss=reconstruction_loss,
            anchor_loss=anchor_loss,
        )
