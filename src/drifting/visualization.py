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
        return self


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
    grid_x: Float[Tensor, "grid_x grid_y"],
    grid_y: Float[Tensor, "grid_x grid_y"],
    model_density: Float[Tensor, "grid_x grid_y"],
    vector_field: Float[Tensor, "grid_x grid_y 2"],
    quiver_stride: int,
    title: str,
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
        )
    )
    quiver_x = grid_x[::quiver_stride, ::quiver_stride].reshape(-1).detach().cpu()
    quiver_y = grid_y[::quiver_stride, ::quiver_stride].reshape(-1).detach().cpu()
    quiver_u = (
        vector_field[::quiver_stride, ::quiver_stride, 0].reshape(-1).detach().cpu()
    )
    quiver_v = (
        vector_field[::quiver_stride, ::quiver_stride, 1].reshape(-1).detach().cpu()
    )
    quiver = ff.create_quiver(
        x=quiver_x,
        y=quiver_y,
        u=quiver_u,
        v=quiver_v,
        scale=0.25,
        arrow_scale=0.3,
        line=dict(color="seagreen", width=1),
        name="reverse-KL drift",
    )
    for trace in quiver.data:
        trace.showlegend = False
        fig.add_trace(trace)
    fig.add_trace(
        go.Scatter(
            x=data_samples[:, 0].detach().cpu(),
            y=data_samples[:, 1].detach().cpu(),
            mode="markers",
            name="data",
            marker=dict(size=5, color="rgba(208, 51, 58, 0.35)"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=model_samples[:, 0].detach().cpu(),
            y=model_samples[:, 1].detach().cpu(),
            mode="markers",
            name="model",
            marker=dict(size=5, color="rgba(17, 24, 39, 0.45)"),
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=900,
        height=900,
        xaxis=dict(title="x", scaleanchor="y"),
        yaxis=dict(title="y"),
    )
    return fig


__all__ = [
    "RegularGridConfig",
    "make_drifting_figure",
    "make_regular_grid",
]
