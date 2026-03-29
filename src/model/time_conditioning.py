from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from pydantic import model_validator
from torch import Tensor

from src.model.base import ModelConfig


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(
        self,
        *,
        min_t_lambda: float,
        max_t_lambda: float,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.min_t_lambda = min_t_lambda
        self.max_t_lambda = max_t_lambda
        self.embedding_dim = embedding_dim

    def forward(
        self,
        t: Float[Tensor, "batch"],
    ) -> Float[Tensor, "batch embedding_dim"]:
        half_dim = self.embedding_dim // 2
        if half_dim == 0:
            raise ValueError("embedding_dim must be at least 2")
        lambdas = torch.linspace(
            math.log(self.min_t_lambda),
            math.log(self.max_t_lambda),
            half_dim,
            device=t.device,
            dtype=t.dtype,
        ).exp()
        angles = (2.0 * math.pi * t.unsqueeze(-1)) / lambdas.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if embedding.shape[-1] == self.embedding_dim:
            return embedding
        return F.pad(embedding, (0, self.embedding_dim - embedding.shape[-1]))


class TimeConditioning(nn.Module):
    def __init__(self, *, config: "TimeConditioningConfig") -> None:
        super().__init__()
        self.embedding = SinusoidalTimeEmbedding(
            min_t_lambda=config.min_t_lambda,
            max_t_lambda=config.max_t_lambda,
            embedding_dim=config.sinusoidal_dim,
        )
        self.projection = nn.Sequential(
            nn.Linear(
                in_features=config.sinusoidal_dim,
                out_features=config.hidden_dim,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=config.hidden_dim,
                out_features=config.output_dim,
                bias=True,
            ),
        )

    def forward(
        self,
        t: Float[Tensor, "batch"],
    ) -> Float[Tensor, "batch output_dim"]:
        return self.projection(self.embedding(t))


class TimeConditioningConfig(ModelConfig):
    min_t_lambda: float
    max_t_lambda: float
    sinusoidal_dim: int
    hidden_dim: int
    output_dim: int

    @model_validator(mode="after")
    def _validate_config(self) -> "TimeConditioningConfig":
        if self.min_t_lambda <= 0.0:
            raise ValueError("min_t_lambda must be positive")
        if self.max_t_lambda <= 0.0:
            raise ValueError("max_t_lambda must be positive")
        if self.min_t_lambda > self.max_t_lambda:
            raise ValueError("min_t_lambda must be <= max_t_lambda")
        if self.sinusoidal_dim < 2:
            raise ValueError("sinusoidal_dim must be at least 2")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        return self

    def get_model(self) -> TimeConditioning:
        return TimeConditioning(config=self)
