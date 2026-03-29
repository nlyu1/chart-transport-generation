from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from jaxtyping import Float, Int
from torch import Tensor

from src.monitoring.utils import marker_color


def _symmetric_stack_positions(
    *,
    count: int,
    device: torch.device,
) -> Tensor:
    if count <= 0:
        return torch.zeros(0, device=device, dtype=torch.float32)
    positions = torch.zeros(count, device=device, dtype=torch.float32)
    if count == 1:
        return positions
    levels = torch.arange(1, count, device=device, dtype=torch.float32)
    magnitudes = torch.div(levels + 1, 2, rounding_mode="floor")
    signs = torch.where(levels % 2 == 1, 1.0, -1.0)
    positions[1:] = magnitudes * signs
    return positions


def beeswarm_offsets(
    *,
    values: Float[Tensor, "count"],
    max_span: float,
    num_bins: int,
) -> Tensor:
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    count = int(values.shape[0])
    if count <= 1:
        return torch.zeros_like(values, dtype=torch.float32)
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")

    min_value = values.min()
    max_value = values.max()
    if torch.isclose(min_value, max_value):
        quantized = torch.zeros(count, device=values.device, dtype=torch.long)
    else:
        scaled = (values - min_value) / (max_value - min_value)
        quantized = torch.round((num_bins - 1) * scaled).long()

    offsets = torch.zeros(count, device=values.device, dtype=torch.float32)
    max_abs_offset = 0.0
    for bin_id in torch.unique(quantized, sorted=True):
        bucket_indices = torch.nonzero(quantized == bin_id, as_tuple=False).squeeze(-1)
        stack_positions = _symmetric_stack_positions(
            count=int(bucket_indices.shape[0]),
            device=values.device,
        )
        offsets[bucket_indices] = stack_positions
        if stack_positions.numel() > 0:
            max_abs_offset = max(
                max_abs_offset,
                float(stack_positions.abs().max().item()),
            )
    if max_abs_offset == 0.0:
        return offsets
    return offsets * (max_span / max_abs_offset)


def add_mode_beeswarm_panel_traces(
    *,
    figure: go.Figure,
    values: Float[Tensor, "batch"],
    labels: Int[Tensor, "batch"],
    col: int,
    value_label: str,
) -> None:
    values_cpu = values.detach().cpu().float()
    labels_cpu = labels.detach().cpu().long()
    num_modes = int(labels_cpu.max().item()) + 1
    for mode_id in range(num_modes):
        mask = labels_cpu == mode_id
        if not mask.any():
            continue
        mode_values = values_cpu[mask]
        sorted_indices = torch.argsort(mode_values)
        sorted_values = mode_values[sorted_indices]
        offsets = beeswarm_offsets(
            values=sorted_values,
            max_span=0.32,
            num_bins=100,
        )
        y_values = torch.full_like(sorted_values, float(mode_id)) + offsets
        figure.add_trace(
            go.Scatter(
                x=sorted_values.tolist(),
                y=y_values.tolist(),
                mode="markers",
                marker={
                    "size": 7,
                    "color": marker_color(group_id=mode_id),
                    "opacity": 0.72,
                    "line": {"width": 0.5, "color": "rgba(0, 0, 0, 0.18)"},
                },
                name=f"mode {mode_id}",
                customdata=[[mode_id]] * int(sorted_values.shape[0]),
                hovertemplate=(
                    "class=%{customdata[0]}"
                    + f"<br>{value_label}=%{{x:.4f}}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=col,
        )


def mode_beeswarm_figure(
    *,
    panel_values: list[Float[Tensor, "batch"]],
    labels: Int[Tensor, "batch"],
    panel_titles: list[str],
    xaxis_title: str,
    title: str,
    value_label: str,
) -> go.Figure:
    if len(panel_values) == 0:
        raise ValueError("panel_values must be non-empty")
    if len(panel_values) != len(panel_titles):
        raise ValueError("panel_values and panel_titles must have matching lengths")

    figure = make_subplots(
        rows=1,
        cols=len(panel_values),
        shared_yaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=tuple(panel_titles),
    )
    for col, values in enumerate(panel_values, start=1):
        add_mode_beeswarm_panel_traces(
            figure=figure,
            values=values,
            labels=labels,
            col=col,
            value_label=value_label,
        )

    labels_cpu = labels.detach().cpu().long()
    num_modes = int(labels_cpu.max().item()) + 1
    tick_values = list(range(num_modes))
    tick_text = [str(mode_id) for mode_id in tick_values]

    for col in range(1, len(panel_values) + 1):
        figure.update_xaxes(title=xaxis_title, row=1, col=col)
        figure.update_yaxes(
            tickmode="array",
            tickvals=tick_values,
            ticktext=tick_text,
            autorange="reversed",
            row=1,
            col=col,
        )
    figure.update_yaxes(
        title="Class",
        row=1,
        col=1,
    )
    figure.update_layout(
        template="plotly_white",
        width=max(1100, 360 * len(panel_values)),
        height=max(480, 120 * num_modes),
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
        title=title,
    )
    return figure


__all__ = [
    "add_mode_beeswarm_panel_traces",
    "beeswarm_offsets",
    "mode_beeswarm_figure",
]
