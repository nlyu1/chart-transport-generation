from __future__ import annotations

import math
from typing import Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from pydantic import model_validator
from torch import Tensor

from src.model.base import EndConfig, ModelConfig
from src.model.mlp import MLP, MLPConfig, build_stacked_mlp_configs


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


class TimeConditioningConfig(ModelConfig):
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

    def get_model(self) -> TimeConditioning:
        return TimeConditioning(config=self)

    def get_module(self) -> TimeConditioning:
        return self.get_model()


class TimeConditionedEndConfig(EndConfig):
    time_conditioning_block_config: TimeConditioningConfig


class TimeConditionedResidualMLP(nn.Module):
    def __init__(
        self,
        *,
        mlp_config: MLPConfig,
        time_feature_dim: int,
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(mlp_config.input_dim)
        self.time_projection = nn.Linear(
            in_features=time_feature_dim,
            out_features=mlp_config.input_dim,
            bias=True,
        )
        self.mlp = MLP(config=mlp_config)
        if mlp_config.input_dim == mlp_config.output_dim:
            self.residual_projection = nn.Identity()
        else:
            self.residual_projection = nn.Linear(
                in_features=mlp_config.input_dim,
                out_features=mlp_config.output_dim,
                bias=True,
            )

    def forward(
        self,
        x: Float[Tensor, "... input_dim"],
        time_features: Float[Tensor, "... time_feature_dim"],
    ) -> Float[Tensor, "... output_dim"]:
        residual = self.residual_projection(x)
        hidden = self.layer_norm(x)
        hidden = hidden + self.time_projection(F.silu(time_features))
        hidden = self.mlp(hidden)
        return residual + hidden


class StackedTimeConditionedMLP(nn.Module):
    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        config: "StackedTimeConditionedMLPConfig",
    ) -> None:
        super().__init__()
        self.shape = shape
        self.sample_dim = math.prod(shape)
        self.time_conditioning = config.time_conditioning_block_config.get_model()
        self.blocks = nn.ModuleList(
            TimeConditionedResidualMLP(
                mlp_config=block_config,
                time_feature_dim=config.time_conditioning_block_config.output_dim,
            )
            for block_config in config.blocks_configs
        )

    def forward(
        self,
        x: Float[Tensor, "batch *shape"],
        t: Float[Tensor, "batch 1"],
    ) -> Float[Tensor, "batch *shape"]:
        if tuple(x.shape[1:]) != self.shape:
            raise ValueError(
                f"expected trailing shape {self.shape}, got {tuple(x.shape[1:])}"
            )
        time_features = self.time_conditioning(t)
        hidden = x.reshape(x.shape[0], self.sample_dim)
        for block in self.blocks:
            hidden = block(hidden, time_features)
        return hidden.reshape(x.shape[0], *self.shape)


class StackedTimeConditionedMLPConfig(TimeConditionedEndConfig):
    blocks_configs: list[MLPConfig]

    @classmethod
    def initialize(
        cls,
        *,
        shape: tuple[int, ...],
        dims: list[int],
        time_conditioning_block_config: TimeConditioningConfig,
    ) -> Self:
        sample_dim = math.prod(shape)
        return cls(
            shape=shape,
            time_conditioning_block_config=time_conditioning_block_config,
            blocks_configs=build_stacked_mlp_configs(
                input_dim=sample_dim,
                dims=dims,
                output_dim=sample_dim,
            ),
        )

    @model_validator(mode="after")
    def _validate_stacked_config(self) -> "StackedTimeConditionedMLPConfig":
        if not self.blocks_configs:
            raise ValueError("blocks_configs must be non-empty")
        if self.blocks_configs[0].input_dim != self.sample_dim:
            raise ValueError("first block input_dim must match the flattened sample_dim")
        for previous_config, next_config in zip(
            self.blocks_configs[:-1],
            self.blocks_configs[1:],
            strict=True,
        ):
            if previous_config.output_dim != next_config.input_dim:
                raise ValueError("adjacent block dims must match")
        if self.blocks_configs[-1].output_dim != self.sample_dim:
            raise ValueError("last block output_dim must match the flattened sample_dim")
        return self

    def get_model(self) -> StackedTimeConditionedMLP:
        return StackedTimeConditionedMLP(
            shape=self.shape,
            config=self,
        )
