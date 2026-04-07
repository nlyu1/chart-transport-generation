from __future__ import annotations

import math
from typing import Self

import torch
from jaxtyping import Float
from torch import Tensor

from src.data.base import BaseDataConfig


class MultimodalGaussianDataConfig(BaseDataConfig):
    num_modes: int
    """Number of Gaussian modes, arranged as roots of unity in a 2D plane."""
    ambient_dimension: int
    """A random isometry embeds the 2D plane into this ambient dimension."""
    mode_std: float
    """Shared isotropic standard deviation for every mode in the 2D plane."""
    data_scale: float
    """Global scale applied to both the mode radius and in-plane noise scale."""
    isometry: Float[Tensor, "2 ambient_dimension"]
    """Linear map from plane coordinates to ambient coordinates."""
    projection: Float[Tensor, "ambient_dimension 2"]
    """Left inverse of the isometry, used to project ambient points back to the plane."""
    offset: float
    """All outputs are shifted by this amount along ambient coordinate 0 **in the 2-dim space**"""

    @classmethod
    def initialize(
        cls,
        *,
        num_modes: int,
        mode_std: float,
        offset: float,
        ambient_dimension: int,
        data_scale: float,
    ) -> Self:
        if num_modes <= 0:
            raise ValueError("num_modes must be positive")
        if ambient_dimension < 2:
            raise ValueError("ambient_dimension must be at least 2")
        if mode_std < 0.0:
            raise ValueError("mode_std must be non-negative")
        if data_scale <= 0.0:
            raise ValueError("data_scale must be positive")
        raw = torch.randn(ambient_dimension, 2)
        projection, _ = torch.linalg.qr(raw, mode="reduced")
        projection = projection[:, :2]
        isometry = projection.transpose(0, 1)
        return cls(
            num_classes=num_modes,
            data_shape=[ambient_dimension],
            num_modes=num_modes,
            ambient_dimension=ambient_dimension,
            mode_std=mode_std,
            data_scale=data_scale,
            isometry=isometry,
            projection=projection,
            offset=offset,
        )

    def offset_vector(self) -> Float[Tensor, "ambient_dimension"]:
        offset_vector = torch.zeros(
            self.ambient_dimension,
            device=self.isometry.device,
            dtype=self.isometry.dtype,
        )
        offset_vector[0] = self.offset
        return offset_vector

    def to(
        self,
        *,
        device: torch.device,
    ) -> Self:
        return self.replace(
            path="isometry",
            replacement=self.isometry.to(device=device),
        ).replace(
            path="projection",
            replacement=self.projection.to(device=device),
        )

    def mode_centers_2d(self) -> Float[Tensor, "num_modes 2"]:
        angles = torch.linspace(
            0.0,
            2.0 * math.pi,
            self.num_modes + 1,
            device=self.isometry.device,
            dtype=self.isometry.dtype,
        )[:-1]
        unit_circle_centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        return self.data_scale * unit_circle_centers

    def embed(
        self,
        x_2d: Float[Tensor, "batch 2"],
    ) -> Float[Tensor, "batch ambient_dimension"]:
        x_2d = x_2d.to(device=self.isometry.device, dtype=self.isometry.dtype)
        return x_2d @ self.isometry + self.offset_vector()

    def project(
        self,
        x_hd: Float[Tensor, "batch ambient_dimension"],
    ) -> Float[Tensor, "batch 2"]:
        x_hd = x_hd.to(device=self.projection.device, dtype=self.projection.dtype)
        return (x_hd - self.offset_vector()) @ self.projection

    def decompose_projection(
        self,
        x_hd: Float[Tensor, "batch ambient_dimension"],
    ) -> tuple[
        Float[Tensor, "batch 2"],
        Float[Tensor, "batch ambient_dimension"],
        Float[Tensor, "batch ambient_dimension"],
    ]:
        projected_2d = self.project(x_hd)
        in_plane = self.embed(projected_2d)
        off_plane = x_hd.to(device=in_plane.device, dtype=in_plane.dtype) - in_plane
        return projected_2d, in_plane, off_plane

    def mode_centers(self) -> Float[Tensor, "num_modes ambient_dimension"]:
        return self.embed(self.mode_centers_2d())

    def _sample_mode_ids(
        self,
        *,
        mode_ids: Tensor,
    ) -> Float[Tensor, "batch ambient_dimension"]:
        centers_2d = self.mode_centers_2d()
        mode_ids = mode_ids.to(device=centers_2d.device, dtype=torch.long)
        noise = (self.data_scale * self.mode_std) * torch.randn(
            mode_ids.shape[0],
            2,
            device=centers_2d.device,
            dtype=centers_2d.dtype,
        )
        return self.embed(centers_2d[mode_ids] + noise)

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
            device=self.isometry.device,
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
            device=self.isometry.device,
        )
        return self._sample_mode_ids(mode_ids=mode_ids)


__all__ = ["MultimodalGaussianDataConfig"]
