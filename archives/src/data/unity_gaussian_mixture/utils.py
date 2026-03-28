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


def project_points_and_compute_manifold_deviation(
    *,
    points: Float[Tensor, "batch dim"],
    basis: Float[Tensor, "ambient_dim plane_dim"],
    scale: float,
    displacement: Float[Tensor, "ambient_dim"],
    mode_centers: Float[Tensor, "num_modes plane_dim"],
    ring_radius: float,
) -> tuple[Float[Tensor, "batch plane_dim"], Float[Tensor, "batch"]]:
    if points.ndim != 2:
        raise ValueError("points must have shape [batch, ambient_dim] or [batch, 2]")
    points = points.detach().to(device="cpu", dtype=basis.dtype)

    if points.shape[-1] == basis.shape[0]:
        displacement = displacement.to(dtype=points.dtype, device=points.device)
        projected_points = (points - displacement) @ basis / scale
        lifted_points = (
            scale * (projected_points @ basis.transpose(0, 1)) + displacement
        )
        off_plane_deviation = torch.linalg.vector_norm(points - lifted_points, dim=-1)
    elif points.shape[-1] == basis.shape[1]:
        projected_points = points
        off_plane_deviation = torch.zeros(
            points.shape[0],
            dtype=points.dtype,
            device=points.device,
        )
    else:
        raise ValueError(
            "points must have last dimension equal to ambient_dim or plane_dim"
        )

    ring_residual = compute_mode_ring_residual(
        points_2d=projected_points,
        mode_centers=mode_centers.to(
            dtype=projected_points.dtype, device=projected_points.device
        ),
        ring_radius=ring_radius,
    )
    manifold_deviation = torch.sqrt(
        (scale * ring_residual).square() + off_plane_deviation.square()
    )
    return projected_points, manifold_deviation


def project_velocities_and_compute_off_manifold_magnitude(
    *,
    velocities: Float[Tensor, "batch ambient_dim"],
    basis: Float[Tensor, "ambient_dim plane_dim"],
    scale: float,
) -> tuple[Float[Tensor, "batch plane_dim"], Float[Tensor, "batch"]]:
    if velocities.ndim != 2:
        raise ValueError("velocities must have shape [batch, ambient_dim]")
    if velocities.shape[-1] != basis.shape[0]:
        raise ValueError("velocities must have last dimension equal to ambient_dim")

    velocities = velocities.detach().to(device="cpu", dtype=basis.dtype)
    projected_velocities = velocities @ basis / scale
    lifted_velocities = scale * (projected_velocities @ basis.transpose(0, 1))
    off_manifold_magnitude = torch.linalg.vector_norm(
        velocities - lifted_velocities,
        dim=-1,
    )
    return projected_velocities, off_manifold_magnitude


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
