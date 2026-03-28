from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass
from torch import Tensor

from src.data.base import DatasetConfig
from src.data.unity_gaussian_mixture.dataset import (
    EmbeddedUnityGaussianMixtureDataset,
    UnityGaussianMixtureBatch,
    collate_unity_gaussian_mixture_batch,
)
from src.data.unity_gaussian_mixture.utils import (
    make_displacement,
    make_mode_centers,
    make_random_isometry,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go

TRAIN_SPLIT_SEED_OFFSET = 0
VAL_SPLIT_SEED_OFFSET = 10_000


@pydantic_dataclass(
    kw_only=True,
    config=ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    ),
)
class UnityGaussianMixtureDatasetConfig(DatasetConfig):
    num_modes: int
    ambient_dim: int
    mode_std: float
    ring_radius_in_zscore: float
    scale: float
    offset_norm: float
    embed_seed: int
    train_size: int
    val_size: int

    def __post_init__(self) -> None:
        if self.ambient_dim < 2:
            raise ValueError("ambient_dim must be at least 2")
        if self.num_modes < 2:
            raise ValueError("num_modes must be at least 2")
        if self.mode_std <= 0.0:
            raise ValueError("mode_std must be positive")
        if self.ring_radius_in_zscore <= 0.0:
            raise ValueError("ring_radius_in_zscore must be positive")
        if self.scale <= 0.0:
            raise ValueError("scale must be positive")
        if self.offset_norm < 0.0:
            raise ValueError("offset_norm must be nonnegative")

    def get_ring_radius(self) -> float:
        return self.mode_std * self.ring_radius_in_zscore

    def get_dataset(
        self,
        *,
        split: Literal["train", "val"],
        seed: int,
    ) -> EmbeddedUnityGaussianMixtureDataset:
        sample_seed = seed + (
            TRAIN_SPLIT_SEED_OFFSET if split == "train" else VAL_SPLIT_SEED_OFFSET
        )
        size = self.train_size if split == "train" else self.val_size
        return EmbeddedUnityGaussianMixtureDataset(
            ambient_dim=self.ambient_dim,
            num_modes=self.num_modes,
            mode_std=self.mode_std,
            ring_radius_in_zscore=self.ring_radius_in_zscore,
            scale=self.scale,
            offset_norm=self.offset_norm,
            embed_seed=self.embed_seed,
            sample_seed=sample_seed,
            size=size,
        )

    def collate_batch(
        self,
        samples: list[dict[str, Tensor]],
    ) -> UnityGaussianMixtureBatch:
        return collate_unity_gaussian_mixture_batch(samples)

    def get_basis(self) -> "Float[Tensor, 'ambient_dim plane_dim']":
        return make_random_isometry(
            ambient_dim=self.ambient_dim,
            seed=self.embed_seed,
        )

    def get_displacement(self) -> "Float[Tensor, 'ambient_dim']":
        return make_displacement(
            ambient_dim=self.ambient_dim,
            seed=self.embed_seed + 1,
            offset_norm=self.offset_norm,
        )

    def get_mode_centers(self) -> "Float[Tensor, 'num_modes plane_dim']":
        return make_mode_centers(
            num_modes=self.num_modes,
        )

    def get_mode_centers_hd(self) -> "Float[Tensor, 'num_modes ambient_dim']":
        basis = self.get_basis()
        displacement = self.get_displacement().to(dtype=basis.dtype)
        mode_centers = self.get_mode_centers().to(dtype=basis.dtype)
        return self.scale * (mode_centers @ basis.transpose(0, 1)) + displacement

    def visualize(
        self,
        *,
        points_by_class: dict[str, Tensor],
        velocities_by_class: dict[str, Tensor] | None = None,
        title: str | None = None,
        alpha: float = 0.4,
    ) -> "go.Figure":
        from src.data.unity_gaussian_mixture.visualization import (
            visualize_unity_gaussian_mixture,
        )

        return visualize_unity_gaussian_mixture(
            config=self,
            points_by_class=points_by_class,
            velocities_by_class=velocities_by_class,
            title=title,
            alpha=alpha,
        )
