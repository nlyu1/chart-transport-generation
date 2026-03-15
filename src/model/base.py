from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn
from diffusers.models.activations import get_activation
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    TimestepEmbedding,
    Timesteps,
)
from pydantic import BaseModel, ConfigDict, model_validator

TimeEmbeddingType = Literal["positional", "fourier"]
TimeConditioningMode = Literal["default", "scale_shift"]


class BaseConfig(BaseModel):
    """Immutable config surface in the same style as Prometheus configs."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )


class EndConfig(BaseConfig, ABC):
    """Config for shape-preserving tensor -> tensor maps."""

    shape: tuple[int, ...]

    @model_validator(mode="after")
    def _validate_shape(self) -> "EndConfig":
        if len(self.shape) == 0:
            raise ValueError("shape must have at least one dimension")
        if any(dim <= 0 for dim in self.shape):
            raise ValueError(f"shape must be positive in every dimension, got {self.shape}")
        return self

    @property
    def sample_dim(self) -> int:
        return math.prod(self.shape)

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Instantiate the configured model."""
        raise NotImplementedError


class TimeConditioningConfig(BaseConfig):
    """
    Diffusers-style timestep embedding config.

    Defaults mirror current diffusers UNet practice:
    positional timestep embeddings, SiLU projection MLP, and scale-shift
    modulation as the stronger conditioning path.
    """

    embedding_type: TimeEmbeddingType = "positional"
    conditioning_mode: TimeConditioningMode = "scale_shift"
    time_embed_dim: int | None = None
    flip_sin_to_cos: bool = True
    freq_shift: float = 0.0
    act_fn: str = "silu"
    post_act_fn: str | None = None
    time_embedding_act_fn: str | None = None
    cond_proj_dim: int | None = None

    @model_validator(mode="after")
    def _validate_config(self) -> "TimeConditioningConfig":
        if self.time_embed_dim is not None and self.time_embed_dim <= 0:
            raise ValueError("time_embed_dim must be positive when provided")
        if self.embedding_type == "fourier":
            resolved_dim = self.time_embed_dim
            if resolved_dim is not None and resolved_dim % 2 != 0:
                raise ValueError("fourier time_embed_dim must be divisible by 2")
        if self.cond_proj_dim is not None and self.cond_proj_dim <= 0:
            raise ValueError("cond_proj_dim must be positive when provided")
        return self

    def resolve_time_embed_dim(self, *, hidden_dim: int) -> int:
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if self.time_embed_dim is not None:
            return self.time_embed_dim
        if self.embedding_type == "fourier":
            return hidden_dim * 2
        return hidden_dim * 4

    def get_module(self, *, hidden_dim: int) -> "DiffusersTimeConditioning":
        return DiffusersTimeConditioning(config=self, hidden_dim=hidden_dim)


class TimeConditionedEndConfig(EndConfig, ABC):
    """Base config for endomorphisms with optional timestep conditioning."""

    time_conditioning_config: TimeConditioningConfig | None = None


class DiffusersTimeConditioning(nn.Module):
    """Adapter around diffusers' `Timesteps` + `TimestepEmbedding` stack."""

    def __init__(self, *, config: TimeConditioningConfig, hidden_dim: int) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.time_embed_dim = config.resolve_time_embed_dim(hidden_dim=hidden_dim)
        self.conditioning_mode = config.conditioning_mode

        if config.embedding_type == "fourier":
            if self.time_embed_dim % 2 != 0:
                raise ValueError("fourier time_embed_dim must be divisible by 2")
            self.time_proj: nn.Module = GaussianFourierProjection(
                self.time_embed_dim // 2,
                set_W_to_weight=False,
                log=False,
                flip_sin_to_cos=config.flip_sin_to_cos,
            )
            timestep_input_dim = self.time_embed_dim
        else:
            self.time_proj = Timesteps(
                hidden_dim,
                config.flip_sin_to_cos,
                config.freq_shift,
            )
            timestep_input_dim = hidden_dim

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            self.time_embed_dim,
            act_fn=config.act_fn,
            post_act_fn=config.post_act_fn,
            cond_proj_dim=config.cond_proj_dim,
        )
        self.time_embed_act = (
            None
            if config.time_embedding_act_fn is None
            else get_activation(config.time_embedding_act_fn)
        )

    def forward(
        self,
        *,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        timestep_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            dtype = torch.float64 if isinstance(timestep, float) else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        else:
            timesteps = timesteps.to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])
        projected = self.time_proj(timesteps).to(dtype=sample.dtype)

        if timestep_cond is not None:
            if self.config.cond_proj_dim is None:
                raise ValueError(
                    "timestep_cond was provided but cond_proj_dim is not configured"
                )
            timestep_cond = timestep_cond.to(device=sample.device, dtype=sample.dtype)

        embedding = self.time_embedding(projected, timestep_cond)
        if self.time_embed_act is not None:
            embedding = self.time_embed_act(embedding)
        return embedding
