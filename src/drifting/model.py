from __future__ import annotations

from typing import Self

import torch
import torch.nn as nn
from jaxtyping import Float
from pydantic import model_validator
from torch import Tensor

from src.model.base import ModelConfig


class AffineGaussianTransportModel(nn.Module):
    def __init__(self, *, config: "AffineGaussianTransportModelConfig") -> None:
        super().__init__()
        self.config = config
        self.mean = nn.Parameter(config.init_mean.clone())
        self.linear = nn.Parameter(config.init_linear.clone())

    def forward(
        self,
        latent: Float[Tensor, "batch latent_dim"],
    ) -> Float[Tensor, "batch ambient_dim"]:
        return latent @ self.linear.transpose(0, 1) + self.mean

    def covariance(self) -> Float[Tensor, "ambient_dim ambient_dim"]:
        return self.linear @ self.linear.transpose(0, 1)

    def distribution(self) -> torch.distributions.MultivariateNormal:
        covariance = self.covariance()
        jitter = self.config.distribution_jitter * torch.eye(
            covariance.shape[0],
            device=covariance.device,
            dtype=covariance.dtype,
        )
        return torch.distributions.MultivariateNormal(
            loc=self.mean,
            covariance_matrix=covariance + jitter,
        )


class AffineGaussianTransportModelConfig(ModelConfig):
    init_mean: Float[Tensor, "ambient_dim"]
    init_linear: Float[Tensor, "ambient_dim latent_dim"]
    lr: float
    grad_clip_norm: float
    distribution_jitter: float = 1e-4

    @classmethod
    def initialize(
        cls,
        *,
        init_mean: list[float],
        init_linear: list[list[float]],
        lr: float,
        grad_clip_norm: float,
        distribution_jitter: float = 1e-4,
    ) -> Self:
        return cls(
            init_mean=torch.tensor(init_mean, dtype=torch.float32),
            init_linear=torch.tensor(init_linear, dtype=torch.float32),
            lr=lr,
            grad_clip_norm=grad_clip_norm,
            distribution_jitter=distribution_jitter,
        )

    @model_validator(mode="after")
    def _validate_config(self) -> Self:
        if self.init_mean.ndim != 1:
            raise ValueError("init_mean must be rank-1")
        if self.init_linear.ndim != 2:
            raise ValueError("init_linear must be rank-2")
        if self.init_linear.shape[0] != self.init_mean.shape[0]:
            raise ValueError(
                "init_linear output dimension must match init_mean dimension"
            )
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive")
        if self.distribution_jitter <= 0.0:
            raise ValueError("distribution_jitter must be positive")
        return self

    @property
    def latent_dimension(self) -> int:
        return self.init_linear.shape[1]

    @property
    def ambient_dimension(self) -> int:
        return self.init_mean.shape[0]

    def get_model(self) -> AffineGaussianTransportModel:
        return AffineGaussianTransportModel(config=self)

    def get_optimizer(
        self,
        model: AffineGaussianTransportModel,
    ) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), lr=self.lr)


__all__ = [
    "AffineGaussianTransportModel",
    "AffineGaussianTransportModelConfig",
]
