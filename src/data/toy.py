from __future__ import annotations

import math
from typing import Literal

import torch
from einops import einsum
from jaxtyping import Float, Int
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .base import DataConfig

TRAIN_SPLIT_SEED_OFFSET = 0
VAL_SPLIT_SEED_OFFSET = 10_000


@pydantic_dataclass(
    kw_only=True,
    frozen=True,
    config=ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    ),
)
class ToyGaussianBatch:
    x_hd: Float[Tensor, "batch ambient_dim"]
    x_2d: Float[Tensor, "batch plane_dim"]
    mode_ids: Int[Tensor, "batch"]


def _make_generator(*, seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _collate_toy_gaussian_batch(
    samples: list[dict[str, Tensor]],
) -> ToyGaussianBatch:
    return ToyGaussianBatch(
        x_hd=torch.stack([sample["x_hd"] for sample in samples], dim=0),
        x_2d=torch.stack([sample["x_2d"] for sample in samples], dim=0),
        mode_ids=torch.stack([sample["mode_ids"] for sample in samples], dim=0),
    )


class EmbeddedToyGaussianDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        *,
        ambient_dim: int,
        num_modes: int,
        mode_std: float,
        ring_radius: float,
        embed_seed: int,
        sample_seed: int,
        size: int,
    ) -> None:
        self._basis = make_random_isometry(ambient_dim=ambient_dim, seed=embed_seed)
        self._centers = make_mode_centers(
            num_modes=num_modes,
            ring_radius=ring_radius,
        )
        generator = _make_generator(seed=sample_seed)
        self._mode_ids = torch.randint(
            low=0,
            high=num_modes,
            size=(size,),
            generator=generator,
        )
        noise = torch.randn(size, 2, generator=generator) * mode_std
        self._x_2d = self._centers[self._mode_ids] + noise
        self._x_hd = einsum(
            self._x_2d,
            self._basis,
            "batch plane_dim, ambient_dim plane_dim -> batch ambient_dim",
        )

    def __len__(self) -> int:
        return int(self._mode_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "x_hd": self._x_hd[index],
            "x_2d": self._x_2d[index],
            "mode_ids": self._mode_ids[index],
        }


def make_random_isometry(
    *,
    ambient_dim: int,
    seed: int,
) -> Float[Tensor, "ambient_dim plane_dim"]:
    generator = _make_generator(seed=seed)
    raw = torch.randn(ambient_dim, 2, generator=generator)
    basis, _ = torch.linalg.qr(raw, mode="reduced")
    return basis[:, :2]


def make_mode_centers(
    *,
    num_modes: int,
    ring_radius: float,
) -> Float[Tensor, "num_modes plane_dim"]:
    angles = torch.linspace(0.0, 2.0 * math.pi, num_modes + 1)[:-1]
    centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    return ring_radius * centers


@pydantic_dataclass(
    kw_only=True,
    config=ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    ),
)
class ToyGaussianDataConfig(DataConfig):
    num_modes: int
    ambient_dim: int
    mode_std: float
    ring_radius: float
    plane_limit: float
    embed_seed: int
    batch_size: int
    eval_size: int
    train_size: int
    val_size: int
    num_workers: int
    pin_memory: bool
    drop_last_train: bool

    def __post_init__(self) -> None:
        if self.ambient_dim < 2:
            raise ValueError("ambient_dim must be at least 2")
        if self.num_modes < 2:
            raise ValueError("num_modes must be at least 2")

    def get_trainloader(self) -> DataLoader[ToyGaussianBatch]:
        dataset = self._build_dataset(split="train")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=_make_generator(seed=self.seed + TRAIN_SPLIT_SEED_OFFSET),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last_train,
            collate_fn=_collate_toy_gaussian_batch,
        )

    def get_valloader(self) -> DataLoader[ToyGaussianBatch]:
        dataset = self._build_dataset(split="val")
        return DataLoader(
            dataset,
            batch_size=self.eval_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=_collate_toy_gaussian_batch,
        )

    def get_basis(self) -> Float[Tensor, "ambient_dim plane_dim"]:
        return make_random_isometry(
            ambient_dim=self.ambient_dim,
            seed=self.embed_seed,
        )

    def get_mode_centers(self) -> Float[Tensor, "num_modes plane_dim"]:
        return make_mode_centers(
            num_modes=self.num_modes,
            ring_radius=self.ring_radius,
        )

    def _build_dataset(
        self,
        *,
        split: Literal["train", "val"],
    ) -> EmbeddedToyGaussianDataset:
        sample_seed = self.seed + (
            TRAIN_SPLIT_SEED_OFFSET if split == "train" else VAL_SPLIT_SEED_OFFSET
        )
        size = self.train_size if split == "train" else self.val_size
        return EmbeddedToyGaussianDataset(
            ambient_dim=self.ambient_dim,
            num_modes=self.num_modes,
            mode_std=self.mode_std,
            ring_radius=self.ring_radius,
            embed_seed=self.embed_seed,
            sample_seed=sample_seed,
            size=size,
        )
