from __future__ import annotations

import math
from typing import Self

import torch
from jaxtyping import Float
from torch import Tensor

from src.priors.base import BasePriorConfig


class UnimodalGaussianPriorConfig(BasePriorConfig):
    @classmethod
    def initialize(
        cls,
        *,
        latent_shape: list[int],
    ) -> Self:
        return cls(latent_shape=latent_shape)

    def sample(
        self,
        *,
        batch_size: int,
    ) -> Float[Tensor, "batch ..."]:
        return torch.randn(batch_size, *self.latent_shape)

    def log_likelihood(
        self,
        samples: Float[Tensor, "batch ..."],
    ) -> Float[Tensor, "batch"]:
        flattened_samples = samples.reshape(samples.shape[0], -1)
        return -0.5 * (
            flattened_samples.square().sum(dim=1)
            + self.latent_numel() * math.log(2.0 * math.pi)
        )

    def analytic_score(
        self,
        y_t: Float[Tensor, "batch ..."],
        t: Float[Tensor, "batch"],
    ) -> Float[Tensor, "batch ..."]:
        variance = ((1.0 - t).square() + t.square()).reshape(
            t.shape[0],
            *([1] * (y_t.ndim - 1)),
        )
        return -y_t / variance


__all__ = ["UnimodalGaussianPriorConfig"]
