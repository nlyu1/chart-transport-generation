from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch.nn as nn
from pydantic import ConfigDict, model_validator

from src.config.base import BaseConfig


class ModelConfig(BaseConfig, ABC):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        extra="forbid",
        frozen=True,
    )

    @abstractmethod
    def get_model(self) -> nn.Module:
        raise NotImplementedError


class EndConfig(ModelConfig):
    shape: tuple[int, ...]

    @model_validator(mode="after")
    def _validate_shape(self) -> "EndConfig":
        if len(self.shape) == 0:
            raise ValueError("shape must have at least one dimension")
        if any(dim <= 0 for dim in self.shape):
            raise ValueError(
                f"shape must be positive in every dimension, got {self.shape}"
            )
        return self

    @property
    def sample_dim(self) -> int:
        return math.prod(self.shape)
