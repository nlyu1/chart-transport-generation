from __future__ import annotations

import math
from typing import Self

import torch
import torch.nn as nn
from jaxtyping import Float
from pydantic import model_validator
from torch import Tensor

from src.model.base import EndConfig, ModelConfig


class MLP(nn.Module):
    def __init__(self, *, config: "MLPConfig") -> None:
        super().__init__()
        self.config = config
        layer_dims = config.layer_dims
        self.layers = nn.ModuleList(
            nn.Linear(
                in_features=in_dim,
                out_features=out_dim,
                bias=True,
            )
            for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:], strict=True)
        )
        self.activation = nn.SiLU()

    def forward(
        self,
        x: Float[Tensor, "batch input_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        hidden = x
        last_layer_index = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index != last_layer_index:
                hidden = self.activation(hidden)
        return hidden


class MLPConfig(ModelConfig):
    input_dim: int
    hidden_dims: list[int]
    output_dim: int

    @model_validator(mode="after")
    def _validate_config(self) -> "MLPConfig":
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if any(hidden_dim <= 0 for hidden_dim in self.hidden_dims):
            raise ValueError("hidden_dims must be positive")
        return self

    @property
    def layer_dims(self) -> tuple[int, ...]:
        return (self.input_dim, *self.hidden_dims, self.output_dim)

    def get_model(self) -> MLP:
        return MLP(config=self)


def build_stacked_mlp_configs(
    *,
    input_dim: int,
    dims: list[int],
    output_dim: int,
) -> list[MLPConfig]:
    block_dims = [input_dim, *dims, output_dim]
    return [
        MLPConfig(
            input_dim=in_dim,
            hidden_dims=[2 * max(in_dim, out_dim)],
            output_dim=out_dim,
        )
        for in_dim, out_dim in zip(block_dims[:-1], block_dims[1:], strict=True)
    ]


class ResidualMLP(nn.Module):
    def __init__(self, *, mlp_config: MLPConfig) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(mlp_config.input_dim)
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
    ) -> Float[Tensor, "... output_dim"]:
        residual = self.residual_projection(x)
        hidden = self.mlp(self.layer_norm(x))
        return residual + hidden


class StackedMLP(nn.Module):
    def __init__(self, *, shape: tuple[int, ...], config: "StackedMLPConfig") -> None:
        super().__init__()
        self.shape = shape
        self.sample_dim = math.prod(shape)
        self.blocks = nn.ModuleList(
            ResidualMLP(mlp_config=block_config)
            for block_config in config.blocks_configs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[1:]) != self.shape:
            raise ValueError(
                f"expected trailing shape {self.shape}, got {tuple(x.shape[1:])}"
            )
        hidden = x.reshape(x.shape[0], self.sample_dim)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden.reshape(x.shape[0], *self.shape)


class StackedMLPConfig(EndConfig):
    blocks_configs: list[MLPConfig]

    @classmethod
    def initialize(
        cls,
        *,
        shape: tuple[int, ...],
        dims: list[int],
    ) -> Self:
        sample_dim = math.prod(shape)
        return cls(
            shape=shape,
            blocks_configs=build_stacked_mlp_configs(
                input_dim=sample_dim,
                dims=dims,
                output_dim=sample_dim,
            ),
        )

    @model_validator(mode="after")
    def _validate_stacked_config(self) -> "StackedMLPConfig":
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

    def get_model(self) -> StackedMLP:
        return StackedMLP(
            shape=self.shape,
            config=self,
        )


class MLPEnd(nn.Module):
    def __init__(self, *, shape: tuple[int, ...], mlp_config: MLPConfig) -> None:
        super().__init__()
        self.shape = shape
        self.sample_dim = math.prod(shape)
        if mlp_config.input_dim != self.sample_dim:
            raise ValueError("mlp_config.input_dim must match the flattened sample_dim")
        if mlp_config.output_dim != self.sample_dim:
            raise ValueError("mlp_config.output_dim must match the flattened sample_dim")
        self.mlp = MLP(config=mlp_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[1:]) != self.shape:
            raise ValueError(
                f"expected trailing shape {self.shape}, got {tuple(x.shape[1:])}"
            )
        hidden = x.reshape(x.shape[0], self.sample_dim)
        output = self.mlp(hidden)
        return output.reshape(x.shape[0], *self.shape)


class MLPEndConfig(EndConfig):
    mlp_config: MLPConfig

    @model_validator(mode="after")
    def _validate_end_config(self) -> "MLPEndConfig":
        if self.mlp_config.input_dim != self.sample_dim:
            raise ValueError("mlp_config.input_dim must match the flattened sample_dim")
        if self.mlp_config.output_dim != self.sample_dim:
            raise ValueError("mlp_config.output_dim must match the flattened sample_dim")
        return self

    def get_model(self) -> MLPEnd:
        return MLPEnd(
            shape=self.shape,
            mlp_config=self.mlp_config,
        )
