from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import einsum
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import Dataset

from src.data.base import GenerativeBatch
from src.data.unity_gaussian_mixture.utils import (
    make_displacement,
    make_random_isometry,
    make_mode_centers,
    truncate_mode_samples_to_ring,
)


@dataclass(kw_only=True, frozen=True)
class UnityGaussianMixtureBatch(GenerativeBatch):
    x_hd: Float[Tensor, "batch ambient_dim"]
    x_2d: Float[Tensor, "batch plane_dim"]
    mode_ids: Int[Tensor, "batch"]

    def data(self) -> Float[Tensor, "batch ambient_dim"]:
        return self.x_hd


def _make_generator(*, seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def collate_unity_gaussian_mixture_batch(
    samples: list[dict[str, Tensor]],
) -> UnityGaussianMixtureBatch:
    return UnityGaussianMixtureBatch(
        x_hd=torch.stack([sample["x_hd"] for sample in samples], dim=0),
        x_2d=torch.stack([sample["x_2d"] for sample in samples], dim=0),
        mode_ids=torch.stack([sample["mode_ids"] for sample in samples], dim=0),
    )


class EmbeddedUnityGaussianMixtureDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        *,
        ambient_dim: int,
        num_modes: int,
        mode_std: float,
        ring_radius_in_zscore: float,
        scale: float,
        offset_norm: float,
        embed_seed: int,
        sample_seed: int,
        size: int,
    ) -> None:
        ring_radius = mode_std * ring_radius_in_zscore
        self._basis = make_random_isometry(ambient_dim=ambient_dim, seed=embed_seed)
        self._displacement = make_displacement(
            ambient_dim=ambient_dim,
            seed=embed_seed + 1,
            offset_norm=offset_norm,
        )
        self._centers = make_mode_centers(
            num_modes=num_modes,
        )
        generator = _make_generator(seed=sample_seed)
        self._mode_ids = torch.randint(
            low=0,
            high=num_modes,
            size=(size,),
            generator=generator,
        )
        noise = torch.randn(size, 2, generator=generator) * mode_std
        noise = truncate_mode_samples_to_ring(
            noise=noise,
            ring_radius=ring_radius,
        )
        self._x_2d = self._centers[self._mode_ids] + noise
        self._x_hd = einsum(
            self._x_2d,
            self._basis,
            "batch plane_dim, ambient_dim plane_dim -> batch ambient_dim",
        )
        self._x_hd = scale * self._x_hd + self._displacement

    def __len__(self) -> int:
        return int(self._mode_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "x_hd": self._x_hd[index],
            "x_2d": self._x_2d[index],
            "mode_ids": self._mode_ids[index],
        }
