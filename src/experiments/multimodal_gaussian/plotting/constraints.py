from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import torch
from jaxtyping import Float, Int
from torch import Tensor

MODE_COLORS = (
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
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
)


def _flatten_batch(
    samples: Float[Tensor, "batch ..."],
) -> Float[Tensor, "batch features"]:
    return samples.reshape(samples.shape[0], -1).to(dtype=torch.float32)


def _orient_projected_axes(
    *,
    projected: Float[Tensor, "batch 2"],
    basis: Float[Tensor, "features components"],
) -> Float[Tensor, "batch 2"]:
    oriented_projection = projected.clone()
    for component_index in range(basis.shape[1]):
        component = basis[:, component_index]
        dominant_coordinate = int(component.abs().argmax().item())
        if component[dominant_coordinate] < 0.0:
            oriented_projection[:, component_index] = -oriented_projection[
                :, component_index
            ]
    return oriented_projection


def _project_points(
    samples: Float[Tensor, "batch ..."],
) -> Float[Tensor, "batch 2"]:
    flat_samples = _flatten_batch(samples)
    if flat_samples.shape[1] == 2:
        return flat_samples
    if flat_samples.shape[1] == 1:
        zeros = torch.zeros_like(flat_samples)
        return torch.cat([flat_samples, zeros], dim=1)

    centered_samples = flat_samples - flat_samples.mean(dim=0, keepdim=True)
    _, _, right_singular_vectors = torch.linalg.svd(
        centered_samples, full_matrices=False
    )
    basis = right_singular_vectors[:2].transpose(0, 1)
    projected = centered_samples @ basis
    if projected.shape[1] == 1:
        projected = torch.cat([projected, torch.zeros_like(projected)], dim=1)
    return _orient_projected_axes(projected=projected, basis=basis)


def _label_values(
    labels: Int[Tensor, "batch"],
) -> list[int]:
    return [int(value) for value in torch.unique(labels, sorted=True).tolist()]


def _mode_color(
    *,
    label: int,
) -> str:
    return MODE_COLORS[label % len(MODE_COLORS)]


def _validate_shared_batch(
    *,
    expected_batch_size: int,
    tensor: Tensor,
    name: str,
) -> None:
    if tensor.shape[0] != expected_batch_size:
        raise ValueError(
            f"{name} must have batch size {expected_batch_size}, got {tensor.shape[0]}",
        )


def plot_sample_pairs(
    samples: Float[Tensor, "batch ..."],
    pairs: Float[Tensor, "batch ..."],
    manifold_deviation: Float[Tensor, "batch"],
    labels: Int[Tensor, "batch"],
    alpha: float = 0.4,
) -> go.Figure:
    """
    Plot aligned sample/pair points in a shared 2D projection.
    Hovering the sample or pair shows the associated manifold deviation.
    """
    batch_size = samples.shape[0]
    _validate_shared_batch(expected_batch_size=batch_size, tensor=pairs, name="pairs")
    _validate_shared_batch(
        expected_batch_size=batch_size,
        tensor=manifold_deviation,
        name="manifold_deviation",
    )
    _validate_shared_batch(expected_batch_size=batch_size, tensor=labels, name="labels")

    projected_points = _project_points(torch.cat([samples, pairs], dim=0))
    sample_points = projected_points[:batch_size]
    pair_points = projected_points[batch_size:]

    figure = go.Figure()
    for label_value in _label_values(labels):
        label_mask = labels == label_value
        label_color = _mode_color(label=label_value)

        sample_x = sample_points[label_mask, 0].numpy()
        sample_y = sample_points[label_mask, 1].numpy()
        pair_x = pair_points[label_mask, 0].numpy()
        pair_y = pair_points[label_mask, 1].numpy()
        label_deviation = manifold_deviation[label_mask].numpy()

        segment_count = sample_x.shape[0]
        segment_x = np.full(segment_count * 3, np.nan, dtype=np.float32)
        segment_y = np.full(segment_count * 3, np.nan, dtype=np.float32)
        segment_x[0::3] = sample_x
        segment_x[1::3] = pair_x
        segment_y[0::3] = sample_y
        segment_y[1::3] = pair_y

        figure.add_trace(
            go.Scatter(
                x=segment_x,
                y=segment_y,
                mode="lines",
                line={"width": 1.0, "color": label_color},
                opacity=max(0.15, alpha * 0.7),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=f"label-{label_value}",
                name=f"label {label_value}",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=sample_x,
                y=sample_y,
                mode="markers",
                marker={
                    "size": 7,
                    "symbol": "circle",
                    "color": label_color,
                    "opacity": alpha,
                },
                customdata=np.column_stack(
                    [
                        np.full(segment_count, label_value, dtype=np.int64),
                        label_deviation,
                    ]
                ),
                hovertemplate=(
                    "role=sample<br>"
                    "label=%{customdata[0]}<br>"
                    "x=%{x:.3f}<br>"
                    "y=%{y:.3f}<br>"
                    "manifold_deviation=%{customdata[1]:.4f}<extra></extra>"
                ),
                legendgroup=f"label-{label_value}",
                name=f"label {label_value}",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=pair_x,
                y=pair_y,
                mode="markers",
                marker={
                    "size": 8,
                    "symbol": "x",
                    "color": label_color,
                    "opacity": alpha,
                },
                customdata=np.column_stack(
                    [
                        np.full(segment_count, label_value, dtype=np.int64),
                        label_deviation,
                    ]
                ),
                hovertemplate=(
                    "role=pair<br>"
                    "label=%{customdata[0]}<br>"
                    "x=%{x:.3f}<br>"
                    "y=%{y:.3f}<br>"
                    "manifold_deviation=%{customdata[1]:.4f}<extra></extra>"
                ),
                showlegend=False,
                legendgroup=f"label-{label_value}",
                name=f"label {label_value} pair",
            )
        )

    figure.update_layout(
        title="Sample pairs",
        template="plotly_white",
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1.0)
    return figure


def plot_latents(
    latents: Float[Tensor, "batch ..."],
    off_manifold_norm: Float[Tensor, "batch"],
    labels: Int[Tensor, "batch"],
    alpha: float = 0.4,
) -> go.Figure:
    """
    Plot latent points in a 2-axis PCA projection.
    Hovering a point shows both latent norm and sample off-manifold norm.
    """
    batch_size = latents.shape[0]
    _validate_shared_batch(
        expected_batch_size=batch_size,
        tensor=off_manifold_norm,
        name="off_manifold_norm",
    )
    _validate_shared_batch(expected_batch_size=batch_size, tensor=labels, name="labels")

    projected_latents = _project_points(latents)
    latent_norm = torch.linalg.vector_norm(_flatten_batch(latents), dim=-1)

    figure = go.Figure()
    for label_value in _label_values(labels):
        label_mask = labels == label_value
        label_color = _mode_color(label=label_value)
        label_count = int(label_mask.sum().item())
        figure.add_trace(
            go.Scatter(
                x=projected_latents[label_mask, 0].numpy(),
                y=projected_latents[label_mask, 1].numpy(),
                mode="markers",
                marker={
                    "size": 7,
                    "symbol": "circle",
                    "color": label_color,
                    "opacity": alpha,
                },
                customdata=np.column_stack(
                    [
                        np.full(label_count, label_value, dtype=np.int64),
                        latent_norm[label_mask].numpy(),
                        off_manifold_norm[label_mask].numpy(),
                    ]
                ),
                hovertemplate=(
                    "label=%{customdata[0]}<br>"
                    "pc1=%{x:.3f}<br>"
                    "pc2=%{y:.3f}<br>"
                    "|latent|=%{customdata[1]:.4f}<br>"
                    "off_manifold_norm=%{customdata[2]:.4f}<extra></extra>"
                ),
                legendgroup=f"label-{label_value}",
                name=f"label {label_value}",
            )
        )

    figure.update_layout(
        title="Latents by mode",
        template="plotly_white",
        xaxis_title="pc1",
        yaxis_title="pc2",
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1.0)
    return figure


__all__ = ["MODE_COLORS", "plot_latents", "plot_sample_pairs"]
