from __future__ import annotations

from enum import Enum
from pathlib import Path

from jaxtyping import Float, Int
import plotly.graph_objects as go
import polars as pl
import torch
from torch import Tensor


COLOR_BANK = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
)


class MonitorStage(str, Enum):
    CHART = "chart"
    CRITIC = "critic"
    INTEGRATED = "integrated"


def marker_color(
    *,
    group_id: int,
) -> str:
    return COLOR_BANK[group_id % len(COLOR_BANK)]


def step_folder(
    *,
    run_folder: Path,
    stage: MonitorStage,
    step: int,
) -> Path:
    folder = run_folder / f"{stage.value}_{step}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def write_figure(
    *,
    figure: go.Figure,
    path_stem: Path,
) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path_stem.with_suffix(".html"))
    figure.write_image(path_stem.with_suffix(".png"))


def write_mode_value_parquet(
    *,
    path: Path,
    mode_ids: Int[Tensor, "batch"],
    value_column_name: str,
    values: Float[Tensor, "batch"],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        {
            "mode_id": mode_ids.detach().cpu().long().tolist(),
            value_column_name: values.detach().cpu().float().tolist(),
        },
        schema={
            "mode_id": pl.Int64,
            value_column_name: pl.Float32,
        },
    ).sort(
        by=["mode_id", value_column_name],
    )
    frame.write_parquet(path)


def flatten_latents(
    points: Float[Tensor, "batch ..."],
) -> Float[Tensor, "batch latent_dim"]:
    return points.reshape(points.shape[0], -1).to(dtype=torch.float32)


def sample_mode_batch(
    *,
    data_config,
    device: torch.device,
    batch_size_per_mode: int,
) -> tuple[Tensor, Int[Tensor, "batch"]]:
    samples = []
    mode_ids = []
    for mode_id in range(data_config.num_classes):
        samples.append(
            data_config.sample_class(
                mode_id=mode_id,
                batch_size=batch_size_per_mode,
            )
        )
        mode_ids.append(
            torch.full(
                (batch_size_per_mode,),
                fill_value=mode_id,
                device=device,
                dtype=torch.long,
            )
        )
    return torch.cat(samples, dim=0), torch.cat(mode_ids, dim=0)


def orient_projection_basis(
    basis: Float[Tensor, "latent_dim projection_dim"],
) -> Float[Tensor, "latent_dim projection_dim"]:
    oriented_basis = basis.clone()
    for component_index in range(oriented_basis.shape[1]):
        component = oriented_basis[:, component_index]
        dominant_coordinate = int(component.abs().argmax().item())
        if component[dominant_coordinate] < 0.0:
            oriented_basis[:, component_index] = -component
    return oriented_basis


def fit_latent_pca_projection(
    *,
    reference_points: Float[Tensor, "batch ..."],
    projection_dim: int,
) -> tuple[
    Float[Tensor, "1 latent_dim"],
    Float[Tensor, "latent_dim projection_dim"],
]:
    if projection_dim <= 0:
        raise ValueError("projection_dim must be positive")

    flat_reference_points = flatten_latents(reference_points)
    projection_center = flat_reference_points.mean(dim=0, keepdim=True)
    centered_points = flat_reference_points - projection_center

    if flat_reference_points.shape[-1] == 1:
        projection_basis = torch.zeros(
            (1, projection_dim),
            device=flat_reference_points.device,
            dtype=flat_reference_points.dtype,
        )
        projection_basis[0, 0] = 1.0
        return projection_center, projection_basis

    _, _, right_singular_vectors = torch.linalg.svd(
        centered_points,
        full_matrices=False,
    )
    projection_basis = right_singular_vectors[:projection_dim].transpose(0, 1).contiguous()
    if projection_basis.shape[1] < projection_dim:
        padding = torch.zeros(
            projection_basis.shape[0],
            projection_dim - projection_basis.shape[1],
            device=projection_basis.device,
            dtype=projection_basis.dtype,
        )
        projection_basis = torch.cat([projection_basis, padding], dim=1)
    return projection_center, orient_projection_basis(projection_basis)


def project_latents_to_pca_space(
    *,
    reference_points: Float[Tensor, "batch ..."],
    points: Float[Tensor, "batch ..."],
    projection_dim: int,
) -> tuple[
    Float[Tensor, "batch projection_dim"],
    Float[Tensor, "batch"],
]:
    projection_center, projection_basis = fit_latent_pca_projection(
        reference_points=reference_points,
        projection_dim=projection_dim,
    )
    flat_points = flatten_latents(points)
    centered_points = flat_points - projection_center
    projected_points = centered_points @ projection_basis
    reconstructed_points = (
        projection_center + projected_points @ projection_basis.transpose(0, 1)
    )
    off_plane_norm = (flat_points - reconstructed_points).norm(dim=-1)
    return projected_points, off_plane_norm


def project_latent_vectors_to_pca_space(
    *,
    reference_points: Float[Tensor, "batch ..."],
    vectors: Float[Tensor, "batch ..."],
    projection_dim: int,
) -> Float[Tensor, "batch projection_dim"]:
    _, projection_basis = fit_latent_pca_projection(
        reference_points=reference_points,
        projection_dim=projection_dim,
    )
    return flatten_latents(vectors) @ projection_basis


def latent_square_limits(
    points: Float[Tensor, "batch projection_dim"],
    *,
    padding: float,
) -> tuple[float, float, float, float]:
    mins = points.min(dim=0).values
    maxs = points.max(dim=0).values
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float((maxs - mins).max().item())
    radius = max(radius * (1.0 + padding), 1.0)
    return (
        float(center[0] - radius),
        float(center[0] + radius),
        float(center[1] - radius),
        float(center[1] + radius),
    )


def build_latent_grid(
    *,
    reference_points: Float[Tensor, "batch ..."],
    resolution: int,
) -> tuple[
    Float[Tensor, "grid latent_dim"],
    Float[Tensor, "resolution"],
    Float[Tensor, "resolution"],
]:
    projected_points, _ = project_latents_to_pca_space(
        reference_points=reference_points,
        points=reference_points,
        projection_dim=2,
    )
    x_min, x_max, y_min, y_max = latent_square_limits(
        projected_points,
        padding=0.18,
    )
    xs = torch.linspace(
        x_min,
        x_max,
        resolution,
        device=projected_points.device,
        dtype=projected_points.dtype,
    )
    ys = torch.linspace(
        y_min,
        y_max,
        resolution,
        device=projected_points.device,
        dtype=projected_points.dtype,
    )
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    projected_grid_points = torch.stack(
        [grid_x.reshape(-1), grid_y.reshape(-1)],
        dim=-1,
    )
    projection_center, projection_basis = fit_latent_pca_projection(
        reference_points=reference_points,
        projection_dim=2,
    )
    grid_points = (
        projection_center + projected_grid_points @ projection_basis.transpose(0, 1)
    )
    return grid_points.to(device=reference_points.device), xs, ys


def vector_display_length(
    points: Float[Tensor, "batch projection_dim"],
    *,
    fraction: float,
) -> float:
    point_span = points.max(dim=0).values - points.min(dim=0).values
    return fraction * max(float(point_span.max().item()), 1.0)


def normalize_vectors(
    *,
    vectors: Float[Tensor, "batch projection_dim"],
    display_length: float,
) -> Float[Tensor, "batch projection_dim"]:
    magnitudes = vectors.norm(dim=-1, keepdim=True)
    return display_length * vectors / magnitudes.clamp_min(1e-6)


def critic_score_from_noise_prediction(
    *,
    predicted_noise: Float[Tensor, "batch latent_dim"],
    t: Float[Tensor, "batch"],
) -> Float[Tensor, "batch latent_dim"]:
    return -predicted_noise / t.unsqueeze(-1)


__all__ = [
    "COLOR_BANK",
    "build_latent_grid",
    "critic_score_from_noise_prediction",
    "fit_latent_pca_projection",
    "flatten_latents",
    "latent_square_limits",
    "marker_color",
    "normalize_vectors",
    "orient_projection_basis",
    "project_latents_to_pca_space",
    "project_latent_vectors_to_pca_space",
    "sample_mode_batch",
    "step_folder",
    "vector_display_length",
    "write_figure",
    "write_mode_value_parquet",
]
