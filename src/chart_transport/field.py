from __future__ import annotations

from abc import ABC, abstractmethod

from jaxtyping import Float
from torch import Tensor

from src.config.base import BaseConfig


class BaseKLWeightSchedule(BaseConfig, ABC):
    """
    Each class specifies a bulk-KL weighting.
    Bulk-KL weighting translates to score-matching weight along the noise spectrum.
    """

    @abstractmethod
    def bulk_kl_weight(
        self,
        t: Float[Tensor, "batch"],
    ) -> Float[Tensor, "batch"]:
        """Return the de Bruijn weighting of the score integral."""
        raise NotImplementedError

    def pullback_weight(
        self,
        t: Float[Tensor, "batch"],
    ) -> Float[Tensor, "batch"]:
        """
        Pullbacks are multiplied with the (1 - t) interpolation weight.
        Used to aggregate the transport field.
        """
        return self.bulk_kl_weight(t) * (1.0 - t)


class UniformVelocityMatchingSchedule(BaseKLWeightSchedule):
    """
    Uniform velocity matching in flow-matching corresponds to a KL integral
    with weight 1 / (1 - t) ** 2.
    """

    def bulk_kl_weight(
        self,
        t: Float[Tensor, "batch"],
    ) -> Float[Tensor, "batch"]:
        return (1.0 - t).pow(-2.0)


__all__ = ["BaseKLWeightSchedule", "UniformVelocityMatchingSchedule"]
