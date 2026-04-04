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
        model_norm: Float[Tensor, ""]
        data_mean: Float[Tensor, ""]
        model_mean: Float[Tensor, ""]

        def sum(self):
            return self.data_norm + self.model_norm + self.data_mean + self.model_mean

    def apply(
        self,
        *,
        data_latents: Float[Tensor, "batch ..."],
        model_latents: Float[Tensor, "batch ..."],
    ) -> Loss:
        return self.Loss(
            data_norm=self.latent_norm_weight * data_latents.pow(2).mean(),
            model_norm=self.latent_norm_weight * model_latents.pow(2).mean(),
            data_mean=self.latent_zero_mean_weight * data_latents.mean(0).pow(2).mean(),
            model_mean=self.latent_zero_mean_weight
            * model_latents.mean(0).pow(2).mean(),
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
        *,
        state: ChartTransportStudyState,
        data: Float[Tensor, "batch ..."],
        compute_anchor_loss: bool,
    ) -> ChartPretrainConfig.Loss:
        batch_size = state.batch_size
        model = state.model
        assert data.shape[0] == batch_size

        data_fiber = state.fiber_packing.get_fiber(batch_size).type_as(data)
        prior = state.config.prior.sample(batch_size=batch_size).type_as(data)

        data_latents = model.encoder(state.fiber_packing.pack(data, data_fiber))

        decoded_data_and_prior = model.decoder(torch.cat([data_latents, prior], dim=0))
        decoded_data, decoded_prior = decoded_data_and_prior.chunk(2, dim=0)
        reconstructed_data, reconstructed_data_fiber = state.fiber_packing.unpack(
            decoded_data
        )
        reconstructed_prior = model.encoder(decoded_prior)

        reconstruction_loss = self.reconstruction_config.apply(
            data=data,
            fiber=data_fiber,
            prior=prior,
            reconstructed_data=reconstructed_data,
            reconstructed_fiber=reconstructed_data_fiber,
            reconstructed_prior=reconstructed_prior,
        )

        anchor_loss: AnchorLossConfig.Loss | None = None
        if compute_anchor_loss:
            model_fiber = state.fiber_packing.get_fiber(batch_size).type_as(data)
            model_samples, _ = state.fiber_packing.unpack(decoded_prior)
            model_latents = model.encoder(
                state.fiber_packing.pack(model_samples, model_fiber)
            )
            anchor_loss = self.anchor_config.apply(
                data_latents=data_latents,
                model_latents=model_latents,
            )

        return self.Loss(
            reconstruction_loss=reconstruction_loss,
            anchor_loss=anchor_loss,
        )
