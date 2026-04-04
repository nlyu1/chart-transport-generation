from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from einops import einsum
from jaxtyping import Float
from torch import Tensor

from src.common.training import BaseLoss
from src.config.base import BaseConfig

if TYPE_CHECKING:
    from src.stochastic_chart_transport.study import StochasticChartTransportStudyState


class CriticLossConfig(BaseConfig):
    """
    Specify for the score matching critic's loss
    """

    huber_delta: float
    weight: float
    t_min: float
    t_max: float

    @dataclass
    class Loss(BaseLoss):
        model: Float[Tensor, ""]
        data: Float[Tensor, ""]
        weight: float

        def sum(self):
            return (self.model + self.data) * self.weight

    def rescale_unit_t(self, *, t: Float[Tensor, "batch"]) -> Float[Tensor, "batch"]:
        return t * (self.t_max - self.t_min) + self.t_min

    def epsilon_like(
        self, *, latent: Float[Tensor, "batch ..."]
    ) -> Float[Tensor, "batch ..."]:
        return torch.randn(
            (latent.shape[0], *latent.shape[1:]),
            device=latent.device,
            dtype=latent.dtype,
        )

    def apply_mixture(
        self,
        *,
        latent: Float[Tensor, "batch ..."],
        epsilon: Float[Tensor, "batch ..."],
        t: Float[Tensor, "batch"],
    ) -> Float[Tensor, "batch ..."]:
        return einsum(t, epsilon, "b, b ... -> b ...") + einsum(
            1.0 - t, latent, "b, b ... -> b ..."
        )

    def apply(
        self,
        *,
        state: StochasticChartTransportStudyState,
        data_latent: Float[Tensor, "batch ..."],
        model_latent: Float[Tensor, "batch ..."],
    ) -> CriticLossConfig.Loss:
        if data_latent.shape != model_latent.shape:
            raise ValueError("data_latent and model_latent must have the same shape")

        batch_size = data_latent.shape[0]
        data_t = self.rescale_unit_t(
            t=torch.rand(
                (batch_size,),
                device=data_latent.device,
                dtype=data_latent.dtype,
            )
        )
        data_epsilon = self.epsilon_like(latent=data_latent)
        data_noised_latent = self.apply_mixture(
            latent=data_latent,
            epsilon=data_epsilon,
            t=data_t,
        )

        model_t = self.rescale_unit_t(
            t=torch.rand(
                (batch_size,),
                device=model_latent.device,
                dtype=model_latent.dtype,
            )
        )
        model_epsilon = self.epsilon_like(latent=model_latent)
        model_noised_latent = self.apply_mixture(
            latent=model_latent,
            epsilon=model_epsilon,
            t=model_t,
        )

        combined_noised_latent = torch.cat(
            [data_noised_latent, model_noised_latent], dim=0
        )
        combined_t = torch.cat([data_t, model_t], dim=0)
        combined_categorical = torch.cat(
            [
                torch.zeros((batch_size,), device=data_latent.device, dtype=torch.long),
                torch.ones((batch_size,), device=model_latent.device, dtype=torch.long),
            ],
            dim=0,
        )
        data_epsilon_preds, model_epsilon_preds = state.model.critic(
            combined_noised_latent,
            t=combined_t,
            categorical=combined_categorical,
        ).chunk(2, dim=0)

        return self.Loss(
            data=F.huber_loss(
                data_epsilon,
                data_epsilon_preds,
                delta=self.huber_delta,
                reduction="mean",
            ),
            model=F.huber_loss(
                model_epsilon,
                model_epsilon_preds,
                delta=self.huber_delta,
                reduction="mean",
            ),
            weight=self.weight,
        )
