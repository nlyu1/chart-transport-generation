from __future__ import annotations

from typing import Self

import torch.nn as nn
from jaxtyping import Float
from pydantic import model_validator
from torch import Tensor

from src.model.base import ModelConfig
from src.model.time_conditioning import TimeConditioningConfig


class MLP(nn.Module):
    def __init__(self, *, config: "MLPConfig") -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            nn.Linear(
                in_features=in_dim,
                out_features=out_dim,
                bias=True,
            )
            for in_dim, out_dim in zip(
                config.layer_dims[:-1],
                config.layer_dims[1:],
                strict=True,
            )
        )
        self.activation = nn.SiLU()

    def forward(
        self,
        x: Float[Tensor, "... input_dim"],
    ) -> Float[Tensor, "... output_dim"]:
        hidden = x
        last_layer_index = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index != last_layer_index:
                hidden = self.activation(hidden)
        return hidden


class MLPConfig(ModelConfig):
    layer_dims: list[int]

    @model_validator(mode="after")
    def _validate_config(self) -> "MLPConfig":
        if len(self.layer_dims) < 2:
            raise ValueError("layer_dims must have at least two entries")
        return self

    @property
    def input_dim(self) -> int:
        return self.layer_dims[0]

    @property
    def output_dim(self) -> int:
        return self.layer_dims[-1]

    def get_model(self) -> MLP:
        return MLP(config=self)


class ResidualMLPConfig(ModelConfig):
    input_dim: int
    hidden_dim: int
    output_dim: int
    add_embedding: bool

    @classmethod
    def initialize(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        add_embedding: bool,
    ) -> Self:
        return cls(
            input_dim=input_dim,
            hidden_dim=2 * max(input_dim, output_dim),
            output_dim=output_dim,
            add_embedding=add_embedding,
        )

    @property
    def mlp_config(self) -> MLPConfig:
        return MLPConfig(
            layer_dims=[self.input_dim, self.hidden_dim, self.output_dim],
        )

    def get_model(self) -> "ResidualMLP":
        return ResidualMLP(config=self)


class ResidualMLP(nn.Module):
    def __init__(self, *, config: ResidualMLPConfig) -> None:
        super().__init__()
        self.config = config
        self.layer_norm = nn.LayerNorm(config.input_dim)
        self.mlp = config.mlp_config.get_model()
        if config.input_dim == config.output_dim:
            self.residual_projection = nn.Identity()
        else:
            self.residual_projection = nn.Linear(
                in_features=config.input_dim,
                out_features=config.output_dim,
                bias=True,
            )

    def forward(
        self,
        x: Float[Tensor, "... input_dim"],
        embedding: Float[Tensor, "... input_dim"] | None = None,
    ) -> Float[Tensor, "... output_dim"]:
        hidden = self.layer_norm(x)
        if self.config.add_embedding:
            if embedding is None:
                raise ValueError("embedding is required when add_embedding=True")
            hidden = hidden + embedding
        residual = self.residual_projection(x)
        return residual + self.mlp(hidden)


class StackedResidualMLPConfig(ModelConfig):
    layer_dims: list[int]
    blocks_configs: list[ResidualMLPConfig]
    time_conditioning_config: TimeConditioningConfig | None = None

    @classmethod
    def initialize(
        cls,
        *,
        layer_dims: list[int],
        time_conditioning_config: TimeConditioningConfig | None = None,
    ) -> Self:
        return cls(
            layer_dims=layer_dims,
            blocks_configs=[
                ResidualMLPConfig.initialize(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    add_embedding=time_conditioning_config is not None and index > 0,
                )
                for index, (input_dim, output_dim) in enumerate(
                    zip(layer_dims[:-1], layer_dims[1:], strict=True)
                )
            ],
            time_conditioning_config=time_conditioning_config,
        )

    @model_validator(mode="after")
    def _validate_config(self) -> "StackedResidualMLPConfig":
        if len(self.layer_dims) < 2:
            raise ValueError("layer_dims must have at least two entries")
        if self.time_conditioning_config is not None:
            for hidden_dim in self.layer_dims[1:-1]:
                if hidden_dim != self.time_conditioning_config.embedding_dim:
                    raise ValueError(
                        "time_conditioning_config.embedding_dim must match all intermediate layer_dims"
                    )
        return self

    def get_model(self) -> "StackedResidualMLP":
        return StackedResidualMLP(config=self)


class StackedResidualMLP(nn.Module):
    def __init__(self, *, config: StackedResidualMLPConfig) -> None:
        super().__init__()
        self.config = config
        self.time_conditioning = None
        if config.time_conditioning_config is not None:
            self.time_conditioning = config.time_conditioning_config.get_model()
        self.blocks = nn.ModuleList(
            block_config.get_model() for block_config in config.blocks_configs
        )

    def forward(
        self,
        x: Float[Tensor, "batch input_dim"],
        t: Float[Tensor, "batch 1"] | None = None,
    ) -> Float[Tensor, "batch output_dim"]:
        embedding = None
        if self.time_conditioning is not None:
            if t is None:
                raise ValueError("t is required when time_conditioning_config is set")
            embedding = self.time_conditioning(t)
        hidden = x
        for block in self.blocks:
            if block.config.add_embedding:
                hidden = block(hidden, embedding=embedding)
            else:
                hidden = block(hidden)
        return hidden
