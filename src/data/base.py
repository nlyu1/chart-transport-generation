from __future__ import annotations

import math
from abc import ABC, abstractmethod

from jaxtyping import Float
from pydantic import ConfigDict
from torch import Tensor

from src.config.base import BaseConfig


class BaseDataConfig(BaseConfig, ABC):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        extra="forbid",
        frozen=True,
    )
    num_classes: int
    data_shape: list[int]

    def data_numel(self) -> int:
        return math.prod(self.data_shape)

    @abstractmethod
    def sample_class(
        self,
        *,
        mode_id: int,
        batch_size: int,
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample_unconditional(
        self,
        *,
        batch_size: int,
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def log_likelihood(
        self,
        samples: Float[Tensor, "batch data_dim"],
    ) -> Float[Tensor, "batch"]:
        raise NotImplementedError


__all__ = ["BaseDataConfig"]
