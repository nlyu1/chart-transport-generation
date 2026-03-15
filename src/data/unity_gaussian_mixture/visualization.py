from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import torch
from jaxtyping import Float
from torch import Tensor

from src.data.unity_gaussian_mixture.definition import (
    UnityGaussianMixtureDatasetConfig,
    compute_mode_ring_residual,
)


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


def visualize_unity_gaussian_mixture(
    *,
    config: UnityGaussianMixtureDatasetConfig,
    batch: np.ndarray,
) -> go.Figure:
    basis = config.get_basis()
    mode_centers = config.get_mode_centers().to(dtype=torch.float32, device="cpu")
    projected_batch = project_batch_to_plane(batch=batch, basis=basis)
    projected_batch = projected_batch.detach().to(dtype=torch.float32, device="cpu")

    off_manifold_residual = compute_mode_ring_residual(
        points_2d=projected_batch,
        mode_centers=mode_centers,
        ring_radius=config.ring_radius,
    )
    off_manifold_score = torch.clamp(
        off_manifold_residual / (2.0 * config.mode_std),
        min=0.0,
        max=1.0,
    )

    figure = go.Figure()
    for center_x, center_y in mode_centers.tolist():
        figure.add_shape(
            type="circle",
            xref="x",
            yref="y",
            x0=center_x - config.ring_radius,
            x1=center_x + config.ring_radius,
            y0=center_y - config.ring_radius,
            y1=center_y + config.ring_radius,
            line=dict(color="rgba(45, 111, 142, 0.35)", width=1.5),
        )

    figure.add_trace(
        go.Scatter(
            x=(mode_centers[:, 0] + config.ring_radius).tolist(),
            y=mode_centers[:, 1].tolist(),
            mode="text",
            text=[f"r={config.ring_radius:.2f}"] * config.num_modes,
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
            text=[f"mode {mode_index}" for mode_index in range(config.num_modes)],
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
        title="Unity Gaussian Mixture Mode Rings",
        xaxis=dict(
            title="x",
            range=[-config.plane_limit, config.plane_limit],
            zeroline=False,
        ),
        yaxis=dict(
            title="y",
            range=[-config.plane_limit, config.plane_limit],
            scaleanchor="x",
            scaleratio=1,
            zeroline=False,
        ),
    )
    return figure
