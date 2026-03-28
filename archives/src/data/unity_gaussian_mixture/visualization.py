from __future__ import annotations

import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import torch
from jaxtyping import Float
from torch import Tensor

from src.data.unity_gaussian_mixture.config import UnityGaussianMixtureDatasetConfig
from src.data.unity_gaussian_mixture.utils import (
    project_points_and_compute_manifold_deviation,
    project_velocities_and_compute_off_manifold_magnitude,
)

POINT_MARKER_SYMBOLS = ("circle", "x", "star")
VELOCITY_COLOR_SCALE = "Bluered"
VELOCITY_ARROW_LENGTH_SCALE = 0.75
VELOCITY_ARROW_HEAD_SCALE = 0.2


def _add_velocity_arrows(
    *,
    figure: go.Figure,
    projected_points: Float[Tensor, "batch plane_dim"],
    projected_velocities: Float[Tensor, "batch plane_dim"],
    off_manifold_magnitude: Float[Tensor, "batch"],
    class_name: str,
    arrow_length: float,
    show_colorbar: bool,
    max_off_manifold_magnitude: float,
    max_projected_velocity_norm: float,
    velocity_scale: float,
) -> None:
    norms = torch.linalg.vector_norm(projected_velocities, dim=-1)
    nonzero_mask = norms > 0.0
    if not bool(nonzero_mask.any()):
        return

    projected_points = projected_points[nonzero_mask].to(dtype=torch.float32, device="cpu")
    projected_velocities = projected_velocities[nonzero_mask].to(
        dtype=torch.float32,
        device="cpu",
    )
    norms = norms[nonzero_mask].to(dtype=torch.float32, device="cpu")
    off_manifold_magnitude = off_manifold_magnitude[nonzero_mask].to(
        dtype=torch.float32,
        device="cpu",
    )
    arrow_delta = arrow_length * projected_velocities / max_projected_velocity_norm
    arrow_heads = projected_points + arrow_delta
    total_velocity_norm = torch.sqrt(
        (velocity_scale * norms).square() + off_manifold_magnitude.square()
    )

    normalized_colors = off_manifold_magnitude / max_off_manifold_magnitude
    colors = sample_colorscale(
        VELOCITY_COLOR_SCALE,
        normalized_colors.tolist(),
    )
    for index, color in enumerate(colors):
        tail_x = float(projected_points[index, 0].item())
        tail_y = float(projected_points[index, 1].item())
        head_x = float(arrow_heads[index, 0].item())
        head_y = float(arrow_heads[index, 1].item())
        delta_x = head_x - tail_x
        delta_y = head_y - tail_y
        left_x = head_x - VELOCITY_ARROW_HEAD_SCALE * (delta_x + delta_y)
        left_y = head_y - VELOCITY_ARROW_HEAD_SCALE * (delta_y - delta_x)
        right_x = head_x - VELOCITY_ARROW_HEAD_SCALE * (delta_x - delta_y)
        right_y = head_y - VELOCITY_ARROW_HEAD_SCALE * (delta_y + delta_x)

        figure.add_trace(
            go.Scatter(
                x=[tail_x, head_x, None, left_x, head_x, right_x],
                y=[tail_y, head_y, None, left_y, head_y, right_y],
                mode="lines",
                name=f"{class_name} velocity",
                showlegend=False,
                hoverinfo="skip",
                line=dict(color=color, width=2.0),
            )
        )

    figure.add_trace(
        go.Scatter(
            x=projected_points[:, 0].tolist(),
            y=projected_points[:, 1].tolist(),
            mode="markers",
            name=f"{class_name} velocity",
            showlegend=False,
            hovertemplate=(
                f"class={class_name}<br>"
                "projected velocity norm=%{customdata[0]:.3f}<br>"
                "velocity off-manifold=%{customdata[1]:.3f}<br>"
                "total velocity norm=%{customdata[2]:.3f}<extra></extra>"
            ),
            customdata=torch.stack(
                [
                    norms,
                    off_manifold_magnitude,
                    total_velocity_norm,
                ],
                dim=-1,
            ).tolist(),
            marker=dict(
                size=8,
                opacity=0.0,
                color=off_manifold_magnitude.tolist(),
                colorscale=VELOCITY_COLOR_SCALE,
                cmin=0.0,
                cmax=max_off_manifold_magnitude,
                showscale=show_colorbar,
                colorbar=dict(title="velocity off-manifold"),
                line=dict(width=0.0),
            ),
        )
    )


def visualize_unity_gaussian_mixture(
    *,
    config: UnityGaussianMixtureDatasetConfig,
    points_by_class: dict[str, Float[Tensor, "batch dim"]],
    velocities_by_class: dict[str, Float[Tensor, "batch ambient_dim"]] | None = None,
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
    if velocities_by_class is None:
        velocities_by_class = {}

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
    unknown_velocity_classes = sorted(
        set(velocities_by_class).difference(points_by_class)
    )
    if unknown_velocity_classes:
        raise ValueError(
            "velocities_by_class keys must be a subset of points_by_class; got "
            + ", ".join(unknown_velocity_classes)
        )
    projected_velocities_and_magnitude_by_class = {}
    for class_name, velocities in velocities_by_class.items():
        if int(velocities.shape[0]) != int(points_by_class[class_name].shape[0]):
            raise ValueError(
                f"velocities for class {class_name!r} must match point batch size"
            )
        projected_velocities_and_magnitude_by_class[class_name] = (
            project_velocities_and_compute_off_manifold_magnitude(
                velocities=velocities,
                basis=basis,
                scale=config.scale,
            )
        )
    max_manifold_deviation = max(
        deviation.max().item()
        for _, deviation in projected_points_and_deviation_by_class.values()
    )
    max_manifold_deviation = max(max_manifold_deviation, 1e-6)
    max_velocity_off_manifold_magnitude = max(
        (
            magnitude.max().item()
            for _, magnitude in projected_velocities_and_magnitude_by_class.values()
        ),
        default=0.0,
    )
    max_velocity_off_manifold_magnitude = max(
        max_velocity_off_manifold_magnitude,
        1e-6,
    )
    max_projected_velocity_norm = max(
        (
            torch.linalg.vector_norm(projected_velocities, dim=-1).max().item()
            for projected_velocities, _ in projected_velocities_and_magnitude_by_class.values()
        ),
        default=0.0,
    )
    max_projected_velocity_norm = max(max_projected_velocity_norm, 1e-6)
    velocity_arrow_length = VELOCITY_ARROW_LENGTH_SCALE * ring_radius

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
        velocity_data = projected_velocities_and_magnitude_by_class.get(class_name)
        if velocity_data is not None:
            projected_velocities, off_manifold_magnitude = velocity_data
            _add_velocity_arrows(
                figure=figure,
                projected_points=projected_points,
                projected_velocities=projected_velocities,
                off_manifold_magnitude=off_manifold_magnitude,
                class_name=class_name,
                arrow_length=velocity_arrow_length,
                show_colorbar=class_index == 0,
                max_off_manifold_magnitude=max_velocity_off_manifold_magnitude,
                max_projected_velocity_norm=max_projected_velocity_norm,
                velocity_scale=config.scale,
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
