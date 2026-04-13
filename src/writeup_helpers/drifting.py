from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import torch

from src.drifting.visualization import make_drifting_figure


def load_drifting_snapshots(artifact_dir: Path) -> list[dict]:
    """Load all numeric .pt snapshot files from artifact_dir, sorted by step."""
    paths = sorted(
        (p for p in artifact_dir.glob("*.pt") if p.stem.isdigit()),
        key=lambda p: int(p.stem),
    )
    return [
        torch.load(p, weights_only=False, map_location="cpu") for p in paths
    ]


def build_drifting_slider_figure(
    snapshots: list[dict],
    *,
    quiver_stride: int = 4,
    scatter_subsample: int = 4,
    width: int = 700,
    height: int = 750,
) -> go.Figure:
    """Build a single plotly figure with a slider that cycles through snapshots.

    Uses visibility toggling: all traces for all steps are added to the figure,
    and the slider controls which step's traces are visible.
    """
    traces_per_step: int | None = None
    all_traces: list[go.BaseTraceType] = []

    for snapshot in snapshots:
        fig = make_drifting_figure(
            data_samples=snapshot["data_samples"][::scatter_subsample],
            model_samples=snapshot["model_samples"][::scatter_subsample],
            representative_model_samples=snapshot["representative_model_samples"],
            representative_model_field=snapshot["representative_model_field"],
            grid_x=snapshot["grid_x"],
            grid_y=snapshot["grid_y"],
            model_density=snapshot["model_density"],
            grid_vector_field=snapshot["scaled_grid_field"],
            quiver_stride=quiver_stride,
        )
        # Keep traces: 0=contour, 1=grid drift, 3=data scatter, 4=model scatter
        # Drop trace 2 (model drift arrows on individual samples)
        kept = [t for i, t in enumerate(fig.data) if i != 2]
        if traces_per_step is None:
            traces_per_step = len(kept)
        for trace in kept:
            trace.visible = False
            all_traces.append(trace)

    assert traces_per_step is not None

    # Legend: 0=contour (no legend), 1=grid drift, 2=data scatter, 3=model scatter
    _legend_config = {
        1: "Drifting field",
        2: "Data distribution",
        3: "Model samples",
    }
    for step_idx in range(len(snapshots)):
        base = step_idx * traces_per_step
        for trace_offset, label in _legend_config.items():
            trace = all_traces[base + trace_offset]
            trace.legendgroup = label
            trace.showlegend = True
            trace.name = label

    for i in range(traces_per_step):
        all_traces[i].visible = True

    total_traces = len(all_traces)
    slider_steps = []
    for idx, snapshot in enumerate(snapshots):
        visible = [False] * total_traces
        for j in range(traces_per_step):
            visible[idx * traces_per_step + j] = True
        x_range = [
            float(snapshot["grid_x"].min()),
            float(snapshot["grid_x"].max()),
        ]
        y_range = [
            float(snapshot["grid_y"].min()),
            float(snapshot["grid_y"].max()),
        ]
        slider_steps.append(
            dict(
                method="update",
                args=[
                    {"visible": visible},
                    {"xaxis.range": x_range, "yaxis.range": y_range},
                ],
                label=str(snapshot["step"]),
            )
        )

    result = go.Figure(data=all_traces)
    result.update_layout(
        template="plotly_white",
        width=width,
        height=height,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="center",
            x=0.5,
        ),
        xaxis=dict(visible=False, showgrid=False, zeroline=False, scaleanchor="y"),
        yaxis=dict(visible=False, showgrid=False, zeroline=False),
        margin=dict(l=10, r=10, t=40, b=60),
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Step: "),
                pad=dict(t=30),
                steps=slider_steps,
            )
        ],
    )
    return result


def save_drifting_slider_html(
    *,
    artifact_dir: Path,
    output_path: Path,
    quiver_stride: int = 4,
    scatter_subsample: int = 4,
    width: int = 700,
    height: int = 750,
) -> Path:
    """Load drifting snapshots and save a self-contained HTML slider visualization."""
    snapshots = load_drifting_snapshots(artifact_dir)
    fig = build_drifting_slider_figure(
        snapshots,
        quiver_stride=quiver_stride,
        scatter_subsample=scatter_subsample,
        width=width,
        height=height,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)
    return output_path
