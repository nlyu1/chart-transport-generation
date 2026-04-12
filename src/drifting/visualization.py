from __future__ import annotations

import plotly.figure_factory as ff
import plotly.graph_objects as go
import torch
from jaxtyping import Float
from pydantic import model_validator
from torch import Tensor

from src.config.base import BaseConfig


class RegularGridConfig(BaseConfig):
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    num_points_per_axis: int
    quiver_stride: int
    model_quiver_stride: int
    padding_fraction: float = 0.15
    min_half_span: float = 1.0

    @model_validator(mode="after")
    def _validate_config(self) -> "RegularGridConfig":
        if self.x_range[0] >= self.x_range[1]:
            raise ValueError("x_range must be increasing")
        if self.y_range[0] >= self.y_range[1]:
            raise ValueError("y_range must be increasing")
        if self.num_points_per_axis <= 1:
            raise ValueError("num_points_per_axis must exceed 1")
        if self.quiver_stride <= 0:
            raise ValueError("quiver_stride must be positive")
        if self.model_quiver_stride <= 0:
            raise ValueError("model_quiver_stride must be positive")
        if self.padding_fraction < 0.0:
            raise ValueError("padding_fraction must be non-negative")
        if self.min_half_span <= 0.0:
            raise ValueError("min_half_span must be positive")
        return self


def infer_square_axis_ranges(
    *,
    samples: Float[Tensor, "batch 2"],
    padding_fraction: float,
    min_half_span: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    min_xy = samples.min(dim=0).values
    max_xy = samples.max(dim=0).values
    center = 0.5 * (min_xy + max_xy)
    half_span = 0.5 * (max_xy - min_xy).max()
    padded_half_span = max(
        min_half_span,
        float(half_span.item()) * (1.0 + padding_fraction),
    )
    x_range = (
        float(center[0].item()) - padded_half_span,
        float(center[0].item()) + padded_half_span,
    )
    y_range = (
        float(center[1].item()) - padded_half_span,
        float(center[1].item()) + padded_half_span,
    )
    return x_range, y_range


def make_regular_grid(
    *,
    config: RegularGridConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    Float[Tensor, "grid 2"],
    Float[Tensor, "grid_x grid_y"],
    Float[Tensor, "grid_x grid_y"],
]:
    x_coordinates = torch.linspace(
        config.x_range[0],
        config.x_range[1],
        config.num_points_per_axis,
        device=device,
        dtype=dtype,
    )
    y_coordinates = torch.linspace(
        config.y_range[0],
        config.y_range[1],
        config.num_points_per_axis,
        device=device,
        dtype=dtype,
    )
    grid_x, grid_y = torch.meshgrid(
        x_coordinates,
        y_coordinates,
        indexing="xy",
    )
    grid_points = torch.stack(
        [grid_x.reshape(-1), grid_y.reshape(-1)],
        dim=-1,
    )
    return grid_points, grid_x, grid_y


def make_drifting_figure(
    *,
    data_samples: Float[Tensor, "data 2"],
    model_samples: Float[Tensor, "model 2"],
    representative_model_samples: Float[Tensor, "rep 2"],
    grid_x: Float[Tensor, "grid_x grid_y"],
    grid_y: Float[Tensor, "grid_x grid_y"],
    model_density: Float[Tensor, "grid_x grid_y"],
    grid_vector_field: Float[Tensor, "grid_x grid_y 2"],
    representative_model_field: Float[Tensor, "rep 2"],
    quiver_stride: int,
    grid_arrow_scale_multiplier: float = 2.5,
    sample_arrow_scale_multiplier: float = 1.0,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=grid_x[:, 0].detach().cpu(),
            y=grid_y[0, :].detach().cpu(),
            z=model_density.detach().cpu().transpose(0, 1),
            name="model contour",
            contours=dict(coloring="lines"),
            line=dict(color="royalblue", width=2),
            showscale=False,
            opacity=0.9,
            hoverinfo="skip",
        )
    )
    quiver_x = grid_x[::quiver_stride, ::quiver_stride].reshape(-1).detach().cpu()
    quiver_y = grid_y[::quiver_stride, ::quiver_stride].reshape(-1).detach().cpu()
    quiver_u = (
        grid_vector_field[::quiver_stride, ::quiver_stride, 0]
        .reshape(-1)
        .detach()
        .cpu()
        * grid_arrow_scale_multiplier
    )
    quiver_v = (
        grid_vector_field[::quiver_stride, ::quiver_stride, 1]
        .reshape(-1)
        .detach()
        .cpu()
        * grid_arrow_scale_multiplier
    )
    quiver = ff.create_quiver(
        x=quiver_x,
        y=quiver_y,
        u=quiver_u,
        v=quiver_v,
        scale=0.35,
        arrow_scale=0.45,
        line=dict(color="rgba(46, 139, 87, 0.85)", width=1.2),
        name="grid drift",
    )
    for trace in quiver.data:
        trace.showlegend = False
        trace.hoverinfo = "skip"
        fig.add_trace(trace)
    sample_quiver = ff.create_quiver(
        x=representative_model_samples[:, 0].detach().cpu(),
        y=representative_model_samples[:, 1].detach().cpu(),
        u=representative_model_field[:, 0].detach().cpu() * sample_arrow_scale_multiplier,
        v=representative_model_field[:, 1].detach().cpu() * sample_arrow_scale_multiplier,
        scale=0.25,
        arrow_scale=0.3,
        line=dict(color="rgba(17, 24, 39, 0.65)", width=1.0),
        name="model drift",
    )
    for trace in sample_quiver.data:
        trace.showlegend = False
        trace.hoverinfo = "skip"
        fig.add_trace(trace)
    fig.add_trace(
        go.Scatter(
            x=data_samples[:, 0].detach().cpu(),
            y=data_samples[:, 1].detach().cpu(),
            mode="markers",
            name="data",
            marker=dict(size=5, color="rgba(208, 51, 58, 0.35)"),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=model_samples[:, 0].detach().cpu(),
            y=model_samples[:, 1].detach().cpu(),
            mode="markers",
            name="model",
            marker=dict(size=5, color="rgba(17, 24, 39, 0.45)"),
            hoverinfo="skip",
        )
    )
    x_range = (
        float(grid_x.min().item()),
        float(grid_x.max().item()),
    )
    y_range = (
        float(grid_y.min().item()),
        float(grid_y.max().item()),
    )
    fig.update_layout(
        template="plotly_white",
        width=900,
        height=900,
        showlegend=False,
        xaxis=dict(
            visible=False,
            showgrid=False,
            zeroline=False,
            range=list(x_range),
            scaleanchor="y",
        ),
        yaxis=dict(
            visible=False,
            showgrid=False,
            zeroline=False,
            range=list(y_range),
        ),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


__all__ = [
    "infer_square_axis_ranges",
    "RegularGridConfig",
    "make_drifting_figure",
    "make_regular_grid",
]
