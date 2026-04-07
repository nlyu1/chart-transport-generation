from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import einsum
from jaxtyping import Float
from torch import Tensor

from src.common.training import BaseLoss
from src.config.base import BaseConfig


class CriticLossConfig(BaseConfig):
    huber_delta: float
    weight: float
    t_min: float
    t_max: float

    @dataclass
    class Loss(BaseLoss):
        data: Float[Tensor, ""]
        weight: float

        def sum(self):
            return self.data * self.weight

    def rescale_unit_t(
        self,
        *,
        t: Float[Tensor, "batch"],
    ) -> Float[Tensor, "batch"]:
        return t * (self.t_max - self.t_min) + self.t_min

    def epsilon_like(
        self,
        *,
        latent: Float[Tensor, "batch ..."],
    ) -> Float[Tensor, "batch ..."]:
        return torch.randn(
            latent.shape,
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
        state,
        data_latent: Float[Tensor, "batch ..."],
    ) -> Loss:
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
        data_epsilon_preds = state.model.critic(data_noised_latent, t=data_t)
        return self.Loss(
            data=F.huber_loss(
                data_epsilon,
                data_epsilon_preds,
                delta=self.huber_delta,
                reduction="mean",
            ),
            weight=self.weight,
        )


__all__ = ["CriticLossConfig"]
