from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.models.activations import get_activation
from pydantic import model_validator

from .base import DiffusersTimeConditioning, TimeConditionedEndConfig


class MLPEnd(nn.Module):
    """Flatten-MLP endomorphism over a fixed sample shape."""

    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        hidden_dim: int,
        depth: int,
        act_fn: str,
        bias: bool,
        dropout: float,
        time_conditioning: DiffusersTimeConditioning | None = None,
    ) -> None:
        super().__init__()
        self.shape = shape
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.time_conditioning = time_conditioning

        sample_dim = 1
        for dim in shape:
            sample_dim *= dim
        self.sample_dim = sample_dim

        in_features = [self.sample_dim, *([hidden_dim] * max(depth - 1, 0))]
        self.layers = nn.ModuleList(
            nn.Linear(in_features=in_dim, out_features=hidden_dim, bias=bias)
            for in_dim in in_features
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(depth))
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation(act_fn)
        self.output = nn.Linear(hidden_dim, self.sample_dim, bias=bias)

        if time_conditioning is None:
            self.time_projections = None
        else:
            projection_out_dim = (
                hidden_dim
                if time_conditioning.conditioning_mode == "default"
                else 2 * hidden_dim
            )
            self.time_projections = nn.ModuleList(
                nn.Linear(time_conditioning.time_embed_dim, projection_out_dim, bias=False)
                for _ in range(depth)
            )

    def forward(
        self,
        x: torch.Tensor,
        *,
        timestep: torch.Tensor | float | int | None = None,
        timestep_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tuple(x.shape[1:]) != self.shape:
            raise ValueError(
                f"expected trailing shape {self.shape}, got {tuple(x.shape[1:])}"
            )

        hidden = x.reshape(x.shape[0], self.sample_dim)
        time_embedding = self._get_time_embedding(
            sample=hidden,
            timestep=timestep,
            timestep_cond=timestep_cond,
        )

        for index, (layer, norm) in enumerate(zip(self.layers, self.norms, strict=True)):
            hidden = norm(layer(hidden))
            if time_embedding is not None and self.time_projections is not None:
                projected = self.time_projections[index](time_embedding)
                if self.time_conditioning.conditioning_mode == "default":
                    hidden = hidden + projected
                else:
                    scale, shift = projected.chunk(2, dim=-1)
                    hidden = hidden * (1.0 + scale) + shift
            hidden = self.activation(hidden)
            hidden = self.dropout(hidden)

        output = self.output(hidden)
        return output.reshape(x.shape[0], *self.shape)

    def _get_time_embedding(
        self,
        *,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int | None,
        timestep_cond: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.time_conditioning is None:
            if timestep is not None or timestep_cond is not None:
                raise ValueError(
                    "timestep inputs were provided but this model is not time-conditioned"
                )
            return None
        if timestep is None:
            raise ValueError("timestep must be provided for time-conditioned models")
        return self.time_conditioning(
            sample=sample,
            timestep=timestep,
            timestep_cond=timestep_cond,
        )


class MLPEndConfig(TimeConditionedEndConfig):
    """Config for a shape-preserving MLP over flattened image samples."""

    hidden_dim: int
    depth: int = 2
    act_fn: str = "silu"
    bias: bool = True
    dropout: float = 0.0

    @model_validator(mode="after")
    def _validate_mlp(self) -> "MLPEndConfig":
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        return self

    def get_model(self) -> MLPEnd:
        time_conditioning = None
        if self.time_conditioning_config is not None:
            time_conditioning = self.time_conditioning_config.get_module(
                hidden_dim=self.hidden_dim
            )

        return MLPEnd(
            shape=self.shape,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            act_fn=self.act_fn,
            bias=self.bias,
            dropout=self.dropout,
            time_conditioning=time_conditioning,
        )
