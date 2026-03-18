from __future__ import annotations

import plotly.graph_objects as go
import torch
from jaxtyping import Float
from torch import Tensor

from src.data.unity_gaussian_mixture.config import UnityGaussianMixtureDatasetConfig
from src.data.unity_gaussian_mixture.utils import compute_mode_ring_residual

POINT_MARKER_SYMBOLS = ("circle", "x", "star")


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


def visualize_unity_gaussian_mixture(
    *,
    config: UnityGaussianMixtureDatasetConfig,
    points_by_class: dict[str, Float[Tensor, "batch dim"]],
    title: str | None = None,
    alpha: float = 0.4,
) -> go.Figure:
    if not points_by_class:
        raise ValueError("points_by_class must not be empty")
    if len(points_by_class) > len(POINT_MARKER_SYMBOLS):
        raise ValueError(
            f"points_by_class supports at most {len(POINT_MARKER_SYMBOLS)} groups"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0.0, 1.0]")

    basis = config.get_basis()
    ring_radius = config.get_ring_radius()
    displacement = config.get_displacement().to(dtype=basis.dtype)
    mode_centers = config.get_mode_centers().to(dtype=torch.float32, device="cpu")
    projected_points_and_deviation_by_class = {
        class_name: project_points_and_compute_manifold_deviation(
            points=points,
            basis=basis,
            scale=config.scale,
            displacement=displacement,
            mode_centers=mode_centers,
            ring_radius=ring_radius,
        )
        for class_name, points in points_by_class.items()
    }
    max_manifold_deviation = max(
        deviation.max().item()
        for _, deviation in projected_points_and_deviation_by_class.values()
    )
    max_manifold_deviation = max(max_manifold_deviation, 1e-6)

    figure = go.Figure()
    for center_x, center_y in mode_centers.tolist():
        figure.add_shape(
            type="circle",
            xref="x",
            yref="y",
            x0=center_x - ring_radius,
            x1=center_x + ring_radius,
            y0=center_y - ring_radius,
            y1=center_y + ring_radius,
            line=dict(color="rgba(45, 111, 142, 0.35)", width=1.5),
        )

    figure.add_trace(
        go.Scatter(
            x=mode_centers[:, 0].tolist(),
            y=mode_centers[:, 1].tolist(),
            mode="markers",
            name="center",
            showlegend=True,
            hoverinfo="skip",
            marker=dict(
                size=10,
                color="#2D708E",
                line=dict(color="#FFFFFF", width=1.0),
            ),
        )
    )

    for class_index, (class_name, point_data) in enumerate(
        projected_points_and_deviation_by_class.items()
    ):
        projected_points, manifold_deviation = point_data
        projected_points = projected_points.to(dtype=torch.float32, device="cpu")
        manifold_deviation = manifold_deviation.to(
            dtype=torch.float32,
            device="cpu",
        )
        figure.add_trace(
            go.Scatter(
                x=projected_points[:, 0].tolist(),
                y=projected_points[:, 1].tolist(),
                mode="markers",
                name=class_name,
                showlegend=True,
                opacity=alpha,
                customdata=manifold_deviation.unsqueeze(-1).tolist(),
                marker=dict(
                    size=7,
                    symbol=POINT_MARKER_SYMBOLS[class_index],
                    color=manifold_deviation.tolist(),
                    colorscale="Viridis",
                    cmin=0.0,
                    cmax=max_manifold_deviation,
                    showscale=class_index == 0,
                    colorbar=dict(
                        title="manifold deviation",
                    ),
                    line=dict(width=0.0),
                ),
                hovertemplate=(
                    f"class={class_name}<br>"
                    "x=%{x:.3f}<br>"
                    "y=%{y:.3f}<br>"
                    "manifold deviation=%{customdata[0]:.3f}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        template="plotly_white",
        title=title,
        legend=dict(
            x=0.0,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.7)",
        ),
        xaxis=dict(title=""),
        yaxis=dict(
            title="",
            scaleanchor="x",
        ),
        dragmode="pan",
    )
    return figure
