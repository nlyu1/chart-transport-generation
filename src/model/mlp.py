from __future__ import annotations

from typing import Self

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from pydantic import model_validator
from torch import Tensor

from src.model.base import ModelConfig
from src.model.time_conditioning import (
    CategoricalConditioningConfig,
    TimeConditioningConfig,
)


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
    condition_dim: int

    @classmethod
    def initialize(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        condition_dim: int,
    ) -> Self:
        return cls(
            input_dim=input_dim,
            hidden_dim=2 * max(input_dim, output_dim),
            output_dim=output_dim,
            condition_dim=condition_dim,
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
        self.conditioning_projection = None
        if config.condition_dim > 0:
            self.conditioning_projection = nn.Linear(
                in_features=config.condition_dim,
                out_features=config.input_dim,
                bias=True,
            )
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
        conditioning: Float[Tensor, "... condition_dim"] | None = None,
    ) -> Float[Tensor, "... output_dim"]:
        hidden = self.layer_norm(x)
        if self.conditioning_projection is not None:
            if conditioning is None:
                raise ValueError("conditioning is required when condition_dim > 0")
            hidden = hidden + self.conditioning_projection(conditioning)
        residual = self.residual_projection(x)
        return residual + self.mlp(hidden)


class StackedResidualMLPConfig(ModelConfig):
    layer_dims: list[int]
    blocks_configs: list[ResidualMLPConfig]
    cat_conditioning_config: CategoricalConditioningConfig | None = None
    time_conditioning_config: TimeConditioningConfig | None = None

    @classmethod
    def initialize(
        cls,
        *,
        layer_dims: list[int],
        cat_conditioning_config: CategoricalConditioningConfig | None = None,
        time_conditioning_config: TimeConditioningConfig | None = None,
    ) -> Self:
        """
        The high-level module is responsible
        for producing a single conditioning vector by concatenating all active
        conditioning sources. Lower-level MLPs then define their own projection
        of this unified conditioning input.
        """
        condition_dim = 0
        if time_conditioning_config is not None:
            condition_dim += time_conditioning_config.condition_dim
        if cat_conditioning_config is not None:
            condition_dim += cat_conditioning_config.condition_dim
        return cls(
            layer_dims=layer_dims,
            blocks_configs=[
                ResidualMLPConfig.initialize(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    condition_dim=condition_dim,
                )
                for input_dim, output_dim in zip(
                    layer_dims[:-1],
                    layer_dims[1:],
                    strict=True,
                )
            ],
            cat_conditioning_config=cat_conditioning_config,
            time_conditioning_config=time_conditioning_config,
        )

    @model_validator(mode="after")
    def _validate_config(self) -> "StackedResidualMLPConfig":
        if len(self.layer_dims) < 2:
            raise ValueError("layer_dims must have at least two entries")
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
        self.cat_conditioning = None
        if config.cat_conditioning_config is not None:
            self.cat_conditioning = config.cat_conditioning_config.get_model()
        self.blocks = nn.ModuleList(
            block_config.get_model() for block_config in config.blocks_configs
        )

    def forward(
        self,
        x: Float[Tensor, "batch input_dim"],
        t: Float[Tensor, "batch"] | None = None,
        categorical: Int[Tensor, "batch"] | None = None,
    ) -> Float[Tensor, "batch output_dim"]:
        conditionings: list[Float[Tensor, "batch condition_dim"]] = []
        if self.time_conditioning is not None:
            if t is None:
                raise ValueError("t is required when time_conditioning_config is set")
            conditionings.append(self.time_conditioning(t))
        if self.cat_conditioning is not None:
            if categorical is None:
                raise ValueError(
                    "categorical is required when cat_conditioning_config is set"
                )
            conditionings.append(self.cat_conditioning(categorical))
        conditioning = None
        if conditionings:
            conditioning = torch.cat(conditionings, dim=-1)
        hidden = x
        for block in self.blocks:
            hidden = block(hidden, conditioning=conditioning)
        return hidden
