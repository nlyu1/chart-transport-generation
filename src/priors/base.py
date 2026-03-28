from __future__ import annotations

import math
from abc import ABC, abstractmethod

from jaxtyping import Float
from pydantic import ConfigDict
from torch import Tensor

from src.config.base import BaseConfig


class BasePriorConfig(BaseConfig, ABC):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        extra="forbid",
        frozen=True,
    )

    data_shape: list[int]

    def data_numel(self) -> int:
        return math.prod(self.data_shape)

    @abstractmethod
    def sample(
        self,
        *,
        batch_size: int,
    ) -> Float[Tensor, "batch ..."]:
        raise NotImplementedError

    @abstractmethod
    def log_likelihood(
        self,
        samples: Float[Tensor, "batch ..."],
    ) -> Float[Tensor, "batch"]:
        raise NotImplementedError

    @abstractmethod
    def analytic_score(
        self,
        y_t: Float[Tensor, "batch ..."],
        t: Float[Tensor, "batch"],
    ) -> Float[Tensor, "batch ..."]:
        raise NotImplementedError


__all__ = ["BasePriorConfig"]
