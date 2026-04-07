from __future__ import annotations

import math
from typing import Self

import torch
from jaxtyping import Float
from torch import Tensor

from src.data.base import BaseDataConfig


class MultimodalGaussianDataConfig(BaseDataConfig):
    num_modes: int
    ambient_dimension: int
    mode_std: float
    offset: float
    scale: float
    plane_basis: Float[Tensor, "2 ambient_dimension"]
    mode_centers: Float[Tensor, "num_modes ambient_dimension"]

    @classmethod
    def initialize(
        cls,
        *,
        num_modes: int,
        mode_std: float,
        offset: float,
        ambient_dimension: int,
        scale: float,
    ) -> Self:
        if num_modes <= 0:
            raise ValueError("num_modes must be positive")
        if ambient_dimension < 2:
            raise ValueError("ambient_dimension must be at least 2")
        if mode_std < 0.0:
            raise ValueError("mode_std must be non-negative")
        if scale <= 0.0:
            raise ValueError("scale must be positive")

        random_plane = torch.randn(ambient_dimension, 2)
        plane_basis, _ = torch.linalg.qr(random_plane, mode="reduced")
        plane_basis = plane_basis.transpose(0, 1)

        angles = torch.linspace(0.0, 2.0 * math.pi, num_modes + 1)[:-1]
        centers_2d = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        mode_centers = scale * (centers_2d @ plane_basis)

        return cls(
            num_classes=num_modes,
            data_shape=[ambient_dimension],
            num_modes=num_modes,
            ambient_dimension=ambient_dimension,
            mode_std=mode_std,
            offset=offset,
            scale=scale,
            plane_basis=plane_basis,
            mode_centers=mode_centers,
        )

    def _offset_vector(self) -> Float[Tensor, "ambient_dimension"]:
        offset_vector = torch.zeros(
            self.ambient_dimension,
            device=self.mode_centers.device,
            dtype=self.mode_centers.dtype,
        )
        offset_vector[0] = self.offset
        return offset_vector

    def _sample_mode_ids(
        self,
        *,
        mode_ids: Tensor,
    ) -> Float[Tensor, "batch ambient_dimension"]:
        mode_ids = mode_ids.to(device=self.mode_centers.device, dtype=torch.long)
        noise_2d = (self.scale * self.mode_std) * torch.randn(
            mode_ids.shape[0],
            2,
            device=self.mode_centers.device,
            dtype=self.mode_centers.dtype,
        )
        noise = noise_2d @ self.plane_basis
        return self.mode_centers[mode_ids] + noise + self._offset_vector()

    def sample_class(
        self,
        *,
        mode_id: int,
        batch_size: int,
    ) -> Float[Tensor, "batch ambient_dimension"]:
        if mode_id < 0 or mode_id >= self.num_classes:
            raise ValueError(f"mode_id must be in [0, {self.num_classes})")
        mode_ids = torch.full(
            (batch_size,),
            fill_value=mode_id,
            device=self.mode_centers.device,
            dtype=torch.long,
        )
        return self._sample_mode_ids(mode_ids=mode_ids)

    def sample_unconditional(
        self,
        *,
        batch_size: int,
    ) -> Float[Tensor, "batch ambient_dimension"]:
        mode_ids = torch.randint(
            self.num_classes,
            size=(batch_size,),
            device=self.mode_centers.device,
        )
        return self._sample_mode_ids(mode_ids=mode_ids)


__all__ = ["MultimodalGaussianDataConfig"]
