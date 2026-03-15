from __future__ import annotations

import math

import torch
from jaxtyping import Float
from torch import Tensor


def truncate_mode_samples_to_ring(
    *,
    noise: Float[Tensor, "batch plane_dim"],
    ring_radius: float,
) -> Float[Tensor, "batch plane_dim"]:
    noise_radii = torch.linalg.vector_norm(noise, dim=-1, keepdim=True)
    safe_noise_radii = torch.where(
        noise_radii > 0.0,
        noise_radii,
        torch.ones_like(noise_radii),
    )
    ring_noise = noise * (ring_radius / safe_noise_radii)
    return torch.where(noise_radii > ring_radius, ring_noise, noise)


def compute_mode_ring_residual(
    *,
    points_2d: Float[Tensor, "batch plane_dim"],
    mode_centers: Float[Tensor, "num_modes plane_dim"],
    ring_radius: float,
) -> Float[Tensor, "batch"]:
    distances_to_centers = torch.cdist(points_2d, mode_centers)
    return (distances_to_centers - ring_radius).abs().min(dim=-1).values


def make_random_isometry(
    *,
    ambient_dim: int,
    seed: int,
) -> Float[Tensor, "ambient_dim plane_dim"]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    raw = torch.randn(ambient_dim, 2, generator=generator)
    basis, _ = torch.linalg.qr(raw, mode="reduced")
    return basis[:, :2]


def make_displacement(
    *,
    ambient_dim: int,
    seed: int,
    offset_norm: float,
) -> Float[Tensor, "ambient_dim"]:
    if offset_norm == 0.0:
        return torch.zeros(ambient_dim)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    raw = torch.randn(ambient_dim, generator=generator)
    raw_norm = torch.linalg.vector_norm(raw)
    if raw_norm == 0.0:
        raise ValueError("failed to construct nonzero displacement")
    return raw * (offset_norm / raw_norm)


def make_mode_centers(
    *,
    num_modes: int,
) -> Float[Tensor, "num_modes plane_dim"]:
    angles = torch.linspace(0.0, 2.0 * math.pi, num_modes + 1)[:-1]
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
