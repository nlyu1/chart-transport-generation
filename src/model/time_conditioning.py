from __future__ import annotations

import math

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from pydantic import model_validator
from torch import Tensor

from src.model.base import ModelConfig


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(
        self,
        *,
        min_t_lambda: float,
        max_t_lambda: float,
        condition_dim: int,
    ) -> None:
        super().__init__()
        self.min_t_lambda = min_t_lambda
        self.max_t_lambda = max_t_lambda
        self.condition_dim = condition_dim

    def forward(
        self,
        t: Float[Tensor, "batch"],
    ) -> Float[Tensor, "batch condition_dim"]:
        half_dim = self.condition_dim // 2
        if half_dim == 0:
            raise ValueError("condition_dim must be at least 2")
        lambdas = torch.linspace(
            math.log(self.min_t_lambda),
            math.log(self.max_t_lambda),
            half_dim,
            device=t.device,
            dtype=t.dtype,
        ).exp()
        angles = (2.0 * math.pi * t.unsqueeze(-1)) / lambdas.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class TimeConditioning(nn.Module):
    def __init__(self, *, config: "TimeConditioningConfig") -> None:
        super().__init__()
        self.embedding = SinusoidalTimeEmbedding(
            min_t_lambda=config.min_t_lambda,
            max_t_lambda=config.max_t_lambda,
            condition_dim=config.condition_dim,
        )
        self.projection = nn.Sequential(
            nn.Linear(
                in_features=config.condition_dim,
                out_features=config.condition_dim,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=config.condition_dim,
                out_features=config.condition_dim,
                bias=True,
            ),
            nn.SiLU(),
        )

    def forward(
        self,
        t: Float[Tensor, "batch"],
    ) -> Float[Tensor, "batch condition_dim"]:
        return self.projection(self.embedding(t))


class TimeConditioningConfig(ModelConfig):
    """
    Creates a sinusoidal embedding of ``t`` with ``condition_dim`` features,
    then projects it through a ``condition_dim -> condition_dim -> condition_dim`` MLP.
    """

    min_t_lambda: float
    max_t_lambda: float
    condition_dim: int

    @model_validator(mode="after")
    def _validate_config(self) -> "TimeConditioningConfig":
        if self.min_t_lambda <= 0:
            raise ValueError("min_t_lambda must be positive")
        if self.max_t_lambda <= 0:
            raise ValueError("max_t_lambda must be positive")
        if self.min_t_lambda > self.max_t_lambda:
            raise ValueError("min_t_lambda must be <= max_t_lambda")
        if self.condition_dim < 2:
            raise ValueError("condition_dim must be at least 2")
        if self.condition_dim % 2 != 0:
            raise ValueError("condition_dim must be divisible by 2")
        return self

    def get_model(self) -> TimeConditioning:
        return TimeConditioning(config=self)


class CategoricalConditioning(nn.Module):
    def __init__(self, *, config: "CategoricalConditioningConfig") -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.num_classes,
            embedding_dim=config.condition_dim,
        )

    def forward(
        self,
        categorical: Int[Tensor, "batch"],
    ) -> Float[Tensor, "batch condition_dim"]:
        return self.embedding(categorical)


class CategoricalConditioningConfig(ModelConfig):
    """
    The resulting class accepts categorical=[batch, (0...C-1)]
    and outputs [batch, condition_dim]
    """

    num_classes: int
    condition_dim: int

    @model_validator(mode="after")
    def _validate_config(self) -> "CategoricalConditioningConfig":
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.condition_dim <= 0:
            raise ValueError("condition_dim must be positive")
        return self

    def get_model(self) -> CategoricalConditioning:
        return CategoricalConditioning(config=self)
