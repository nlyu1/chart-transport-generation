from __future__ import annotations

import math
from typing import Self

import torch
from jaxtyping import Float
from pydantic import model_validator
from torch import Tensor

from src.priors.base import BasePriorConfig


class AnchoredGaussianScaleMixturePriorConfig(BasePriorConfig):
    precision: float

    @classmethod
    def initialize(
        cls,
        *,
        data_shape: list[int],
        precision: float,
    ) -> Self:
        return cls(
            data_shape=data_shape,
            precision=precision,
        )

    @model_validator(mode="after")
    def _validate_config(self) -> Self:
        if self.precision < 1.0:
            raise ValueError("precision must be at least 1.0")
        return self

    @staticmethod
    def _normal_log_density(
        *,
        samples: Float[Tensor, "batch ..."],
        variance: Float[Tensor, "batch ..."],
    ) -> Float[Tensor, "batch ..."]:
        return -0.5 * (
            samples.square() / variance + torch.log(2.0 * math.pi * variance)
        )

    def _noised_component_variances(
        self,
        *,
        t: Float[Tensor, "batch"],
        target_ndim: int,
    ) -> tuple[Float[Tensor, "batch ..."], Float[Tensor, "batch ..."]]:
        one_minus_t = 1.0 - t
        narrow_variance = t.square() + one_minus_t.square() / self.precision
        wide_variance = t.square() + one_minus_t.square() * self.precision
        broadcast_shape = (t.shape[0], *([1] * (target_ndim - 1)))
        return (
            narrow_variance.reshape(broadcast_shape),
            wide_variance.reshape(broadcast_shape),
        )

    def sample(
        self,
        *,
        batch_size: int,
    ) -> Float[Tensor, "batch ..."]:
        sample_shape = (batch_size, *self.data_shape)
        narrow_probability = self.precision / (self.precision + 1.0)
        component_is_narrow = torch.rand(sample_shape) < narrow_probability
        component_variance = torch.where(
            component_is_narrow,
            torch.full(sample_shape, 1.0 / self.precision),
            torch.full(sample_shape, self.precision),
        )
        return torch.randn(sample_shape) * component_variance.sqrt()

    def log_likelihood(
        self,
        samples: Float[Tensor, "batch ..."],
    ) -> Float[Tensor, "batch"]:
        narrow_log_density = self._normal_log_density(
            samples=samples,
            variance=torch.full_like(samples, 1.0 / self.precision),
        )
        wide_log_density = self._normal_log_density(
            samples=samples,
            variance=torch.full_like(samples, self.precision),
        )
        coordinate_log_likelihood = torch.logaddexp(
            math.log(self.precision) + narrow_log_density,
            wide_log_density,
        ) - math.log(self.precision + 1.0)
        return coordinate_log_likelihood.reshape(samples.shape[0], -1).sum(dim=1)

    def analytic_score(
        self,
        y_t: Float[Tensor, "batch ..."],
        t: Float[Tensor, "batch"],
    ) -> Float[Tensor, "batch ..."]:
        narrow_variance, wide_variance = self._noised_component_variances(
            t=t,
            target_ndim=y_t.ndim,
        )
        narrow_log_density = self._normal_log_density(
            samples=y_t,
            variance=narrow_variance,
        )
        wide_log_density = self._normal_log_density(
            samples=y_t,
            variance=wide_variance,
        )
        component_logits = torch.stack(
            (math.log(self.precision) + narrow_log_density, wide_log_density),
            dim=0,
        )
        responsibilities = torch.softmax(component_logits, dim=0)
        return -y_t * (
            responsibilities[0] / narrow_variance + responsibilities[1] / wide_variance
        )
