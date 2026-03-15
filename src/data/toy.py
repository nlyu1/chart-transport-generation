from __future__ import annotations

import math
from typing import Literal

import numpy as np
import plotly.graph_objects as go
import torch
from einops import einsum
from jaxtyping import Float, Int
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass
from torch import Tensor
from torch.utils.data import Dataset

from .base import DatasetConfig

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


def truncate_mode_samples_to_ring(
    *,
    noise: Float[Tensor, "batch plane_dim"],
    mode_std: float,
    ring_radius: float,
) -> Float[Tensor, "batch plane_dim"]:
    noise_radii = torch.linalg.vector_norm(noise, dim=-1, keepdim=True)
    safe_noise_radii = torch.where(
        noise_radii > 0.0,
        noise_radii,
        torch.ones_like(noise_radii),
    )
    ring_noise = noise * (ring_radius / safe_noise_radii)
    return torch.where(noise_radii > 2.0 * mode_std, ring_noise, noise)


def compute_mode_ring_residual(
    *,
    points_2d: Float[Tensor, "batch plane_dim"],
    mode_centers: Float[Tensor, "num_modes plane_dim"],
    ring_radius: float,
) -> Float[Tensor, "batch"]:
    distances_to_centers = torch.cdist(points_2d, mode_centers)
    return (distances_to_centers - ring_radius).abs().min(dim=-1).values


def project_batch_to_plane(
    *,
    batch: np.ndarray,
    basis: Float[Tensor, "ambient_dim plane_dim"],
) -> Float[Tensor, "batch plane_dim"]:
    batch_tensor = torch.from_numpy(np.asarray(batch)).to(dtype=basis.dtype)
    if batch_tensor.ndim != 2:
        raise ValueError("batch must have shape [batch, ambient_dim] or [batch, 2]")
    if batch_tensor.shape[-1] == basis.shape[0]:
        return batch_tensor @ basis
    if batch_tensor.shape[-1] == basis.shape[1]:
        return batch_tensor
    raise ValueError(
        "batch must have last dimension equal to ambient_dim or plane_dim"
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
        noise = truncate_mode_samples_to_ring(
            noise=noise,
            mode_std=mode_std,
            ring_radius=ring_radius,
        )
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
class ToyGaussianDatasetConfig(DatasetConfig):
    num_modes: int
    ambient_dim: int
    mode_std: float
    ring_radius: float
    plane_limit: float
    embed_seed: int
    train_size: int
    val_size: int

    def __post_init__(self) -> None:
        if self.ambient_dim < 2:
            raise ValueError("ambient_dim must be at least 2")
        if self.num_modes < 2:
            raise ValueError("num_modes must be at least 2")

    def get_dataset(
        self,
        *,
        split: Literal["train", "val"],
        seed: int,
    ) -> EmbeddedToyGaussianDataset:
        sample_seed = seed + (
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

    def collate_batch(self, samples: list[dict[str, Tensor]]) -> ToyGaussianBatch:
        return _collate_toy_gaussian_batch(samples)

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

    def visualize(self, *, batch: np.ndarray) -> go.Figure:
        basis = self.get_basis()
        mode_centers = self.get_mode_centers().to(dtype=torch.float32, device="cpu")
        projected_batch = project_batch_to_plane(batch=batch, basis=basis)
        projected_batch = projected_batch.detach().to(dtype=torch.float32, device="cpu")

        off_manifold_residual = compute_mode_ring_residual(
            points_2d=projected_batch,
            mode_centers=mode_centers,
            ring_radius=self.ring_radius,
        )
        off_manifold_score = torch.clamp(
            off_manifold_residual / (2.0 * self.mode_std),
            min=0.0,
            max=1.0,
        )

        figure = go.Figure()
        for center_x, center_y in mode_centers.tolist():
            figure.add_shape(
                type="circle",
                xref="x",
                yref="y",
                x0=center_x - self.ring_radius,
                x1=center_x + self.ring_radius,
                y0=center_y - self.ring_radius,
                y1=center_y + self.ring_radius,
                line=dict(color="rgba(45, 111, 142, 0.35)", width=1.5),
            )

        figure.add_trace(
            go.Scatter(
                x=(mode_centers[:, 0] + self.ring_radius).tolist(),
                y=mode_centers[:, 1].tolist(),
                mode="text",
                text=[f"r={self.ring_radius:.2f}"] * self.num_modes,
                textfont=dict(color="rgba(45, 111, 142, 0.85)", size=11),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=mode_centers[:, 0].tolist(),
                y=mode_centers[:, 1].tolist(),
                mode="markers+text",
                text=[f"mode {mode_index}" for mode_index in range(self.num_modes)],
                textposition="top center",
                name="mode centers",
                marker=dict(
                    size=10,
                    color="#2D708E",
                    line=dict(color="#FFFFFF", width=1.0),
                ),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=projected_batch[:, 0].tolist(),
                y=projected_batch[:, 1].tolist(),
                mode="markers",
                name="batch",
                customdata=torch.stack(
                    [off_manifold_residual, off_manifold_score],
                    dim=-1,
                ).tolist(),
                marker=dict(
                    size=8,
                    color=off_manifold_score.tolist(),
                    cmin=0.0,
                    cmax=1.0,
                    colorscale="Viridis",
                    line=dict(color="rgba(255, 255, 255, 0.25)", width=0.5),
                    colorbar=dict(
                        title="off-manifold",
                        tickvals=[0.0, 1.0],
                        ticktext=["on", "off"],
                    ),
                ),
                hovertemplate=(
                    "x=%{x:.3f}<br>"
                    "y=%{y:.3f}<br>"
                    "ring residual=%{customdata[0]:.3f}<br>"
                    "off-manifold score=%{customdata[1]:.3f}<extra></extra>"
                ),
            )
        )
        figure.update_layout(
            template="plotly_white",
            title="Toy Gaussian Mode Rings",
            xaxis=dict(
                title="x",
                range=[-self.plane_limit, self.plane_limit],
                zeroline=False,
            ),
            yaxis=dict(
                title="y",
                range=[-self.plane_limit, self.plane_limit],
                scaleanchor="x",
                scaleratio=1,
                zeroline=False,
            ),
        )
        return figure
