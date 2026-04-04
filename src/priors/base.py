from __future__ import annotations

import math
from abc import ABC, abstractmethod

from jaxtyping import Float
from torch import Tensor

from src.config.base import BaseConfig


class BasePriorConfig(BaseConfig, ABC):
    latent_shape: list[int]

    def latent_numel(self) -> int:
        return math.prod(self.latent_shape)

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
        *,
        y_t: Float[Tensor, "batch ..."],
        t: Float[Tensor, "batch"],
    ) -> Float[Tensor, "batch ..."]:
        raise NotImplementedError


__all__ = ["BasePriorConfig"]
