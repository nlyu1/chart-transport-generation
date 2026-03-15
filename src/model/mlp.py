from __future__ import annotations

import math

import torch
import torch.nn as nn
from jaxtyping import Float
from pydantic import model_validator
from torch import Tensor

from src.model.base import BaseConfig, EndConfig


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


class MLPConfig(BaseConfig):
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
