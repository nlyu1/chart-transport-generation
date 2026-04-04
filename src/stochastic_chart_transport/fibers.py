from abc import ABC, abstractmethod

import torch
from jaxtyping import Float
from torch import Tensor

from src.config.base import BaseConfig


class FiberPacking(BaseConfig, ABC):
    @abstractmethod
    def pack(
        data: Float[Tensor, "batch ..."], fiber: Float[Tensor, "batch ..."]
    ) -> Float[Tensor, "batch ..."]:
        pass

    @abstractmethod
    def unpack(
        data_with_fiber: Float[Tensor, "batch ..."],
    ) -> tuple[Float[Tensor, "batch ..."], Float[Tensor, "batch ..."]]:
        pass

    @abstractmethod
    def get_fiber(self, batch_size: int) -> Float[Tensor, "batch ..."]:
        pass


class FlatFiberPacking(FiberPacking):
    fiber_ndims: int

    def pack(self, data, fiber):
        return torch.cat([data, fiber], dim=-1)

    def unpack(self, data_with_fiber):
        return data_with_fiber[..., : -self.fiber_ndims], data_with_fiber[
            ..., -self.fiber_ndims :
        ]

    def get_fiber(self, batch_size: int) -> Float[Tensor, "batch ..."]:
        return torch.randn((batch_size, self.fiber_ndims))
