from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from pydantic import model_validator
from torch import Tensor

from src.model.base import BaseConfig
from src.model.mlp import MLP, MLPConfig


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, *, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(
        self,
        t: Float[Tensor, "batch 1"],
    ) -> Float[Tensor, "batch embedding_dim"]:
        half_dim = self.embedding_dim // 2
        if half_dim == 0:
            raise ValueError("embedding_dim must be at least 2")
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / max(half_dim - 1, 1)
        )
        angles = t * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if embedding.shape[-1] == self.embedding_dim:
            return embedding
        return F.pad(embedding, (0, self.embedding_dim - embedding.shape[-1]))


class TimeConditioning(nn.Module):
    def __init__(self, *, config: "TimeConditioningConfig") -> None:
        super().__init__()
        self.config = config
        self.embedding = SinusoidalTimeEmbedding(
            embedding_dim=config.embedding_dim,
        )
        self.mlp = MLP(config=config.conditioning_mlp_config)

    def forward(
        self,
        t: Float[Tensor, "batch 1"],
    ) -> Float[Tensor, "batch output_dim"]:
        embedding = self.embedding(t)
        return self.mlp(embedding)


class TimeConditioningConfig(BaseConfig):
    embedding_dim: int
    conditioning_mlp_config: MLPConfig

    @model_validator(mode="after")
    def _validate_config(self) -> "TimeConditioningConfig":
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.conditioning_mlp_config.input_dim != self.embedding_dim:
            raise ValueError(
                "conditioning_mlp_config.input_dim must match embedding_dim"
            )
        return self

    @property
    def output_dim(self) -> int:
        return self.conditioning_mlp_config.output_dim

    def get_module(self) -> TimeConditioning:
        return TimeConditioning(config=self)
