from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from plotly.subplots import make_subplots
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torch import Tensor

from src.method.drifting.kernel import compute_gaussian_drifting_statistics
from src.method.drifting.kernel import compute_gaussian_drifting_statistics_at_query
from src.toy import (
    DataConfig,
    DecoderMLP,
    add_quiver_traces,
    build_gaussian_contour_points,
    mode_centers_tensor,
    normalize_vectors_for_display,
    sample_bimodal_gaussian,
    save_plotly_figure,
)


@dataclass(
    kw_only=True,
    config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"),
)
class DriftingLossConfig:
    objective: Literal["reverse_kl", "forward_kl"]
    bandwidth: float
    drift_scale: float
    exclude_self_interactions: bool
    stability_eps: float

    def __post_init__(self) -> None:
        if self.bandwidth <= 0.0:
            raise ValueError("bandwidth must be positive")
        if self.drift_scale <= 0.0:
            raise ValueError("drift_scale must be positive")
        if self.stability_eps <= 0.0:
            raise ValueError("stability_eps must be positive")


@dataclass(
    kw_only=True,
    config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"),
)
class TrainConfig:
    steps: int
    lr: float
    weight_decay: float
    grad_clip: float
    log_every_steps: int

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be nonnegative")
        if self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be positive")
        if self.log_every_steps < 1:
            raise ValueError("log_every_steps must be positive")


@dataclass(
    kw_only=True,
    config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"),
)
class PlotConfig:
    eval_size: int
    contour_levels: tuple[float, float, float, float, float, float]
    contour_points_per_level: int
    snapshot_every_steps: int
    arrow_stride: int
    arrow_display_length: float

    def __post_init__(self) -> None:
        if self.eval_size < 1:
            raise ValueError("eval_size must be positive")
        if self.contour_points_per_level < 1:
            raise ValueError("contour_points_per_level must be positive")
        if self.snapshot_every_steps < 1:
            raise ValueError("snapshot_every_steps must be positive")
        if self.arrow_stride < 1:
            raise ValueError("arrow_stride must be positive")
        if self.arrow_display_length <= 0.0:
            raise ValueError("arrow_display_length must be positive")


@dataclass(
    kw_only=True,
    config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"),
)
class ExperimentConfig:
    data: DataConfig
    loss: DriftingLossConfig
    train: TrainConfig
    plot: PlotConfig


class BimodalGaussianDriftingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = DecoderMLP()

    def decode(
        self,
        *,
        z: Float[Tensor, "batch 2"],
    ) -> Float[Tensor, "batch 2"]:
        return self.decoder(z)

    def forward(
        self,
        z: Float[Tensor, "batch 2"],
    ) -> Float[Tensor, "batch 2"]:
        return self.decode(z=z)


def make_run_catalog() -> str:
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_hash = hashlib.sha1(
        datetime.now().isoformat(timespec="microseconds").encode()
    ).hexdigest()[:12]
    return f"{run_timestamp}-{run_hash}"


def make_run_root() -> Path:
    run_root = Path("/tmp") / make_run_catalog()
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def assign_mode_ids(
    *,
    points: Float[Tensor, "batch 2"],
    mode_centers: Float[Tensor, "2 2"],
) -> Tensor:
    return torch.cdist(points, mode_centers).argmin(dim=-1)


def compute_mode_residual(
    *,
    points: Float[Tensor, "batch 2"],
    mode_centers: Float[Tensor, "2 2"],
) -> Tensor:
    return torch.cdist(points, mode_centers).min(dim=-1).values


def compute_drifting_losses(
    *,
    model: BimodalGaussianDriftingModel,
    x_data: Float[Tensor, "batch 2"],
    config: ExperimentConfig,
) -> dict[str, Tensor]:
    batch_size = int(x_data.shape[0])
    z = torch.randn(batch_size, 2, device=x_data.device, dtype=x_data.dtype)
    x_model = model.decode(z=z)
    statistics = compute_gaussian_drifting_statistics(
        model_samples=x_model.detach().float(),
        data_samples=x_data.detach().float(),
        bandwidth=config.loss.bandwidth,
        objective=config.loss.objective,
        stability_eps=config.loss.stability_eps,
        exclude_self_interactions=config.loss.exclude_self_interactions,
    )
    drift = statistics.drift.to(device=x_model.device, dtype=x_model.dtype)
    x_target = (x_model.detach() + config.loss.drift_scale * drift).detach()
    drift_matching_loss = F.mse_loss(x_model.float(), x_target.float())
    return {
        "total_loss": drift_matching_loss,
        "drift_matching_loss": drift_matching_loss.detach(),
        "drift_norm": statistics.drift.norm(dim=-1).mean().detach(),
        "data_density": statistics.data_density.mean().detach(),
        "model_density": statistics.model_density.mean().detach(),
        "density_ratio": statistics.density_ratio.mean().detach(),
    }


def initialize_history() -> dict[str, list[float]]:
    return {
        "step": [],
        "total_loss": [],
        "drift_matching_loss": [],
        "drift_norm": [],
        "data_density": [],
        "model_density": [],
        "density_ratio": [],
        "mode_residual": [],
        "mode_balance_error": [],
        "mean_error": [],
    }


def append_history_entry(
    *,
    history: dict[str, list[float]],
    step: int,
    losses: dict[str, Tensor],
    metrics: dict[str, float],
) -> None:
    history["step"].append(step)
    history["total_loss"].append(float(losses["total_loss"].item()))
    history["drift_matching_loss"].append(float(losses["drift_matching_loss"].item()))
    history["drift_norm"].append(float(losses["drift_norm"].item()))
    history["data_density"].append(float(losses["data_density"].item()))
    history["model_density"].append(float(losses["model_density"].item()))
    history["density_ratio"].append(float(losses["density_ratio"].item()))
    history["mode_residual"].append(float(metrics["mode_residual"]))
    history["mode_balance_error"].append(float(metrics["mode_balance_error"]))
    history["mean_error"].append(float(metrics["mean_error"]))


def evaluate_model(
    *,
    model: BimodalGaussianDriftingModel,
    fixed_eval_x: Float[Tensor, "batch 2"],
    fixed_eval_mode_ids: Tensor,
    fixed_prior_points: Float[Tensor, "prior_batch 2"],
    fixed_contour_ids: Tensor,
    mode_centers: Float[Tensor, "2 2"],
    config: ExperimentConfig,
    step: int,
) -> tuple[dict[str, object], dict[str, float]]:
    model.eval()
    with torch.no_grad():
        z_eval = torch.randn(
            config.plot.eval_size,
            2,
            device=fixed_eval_x.device,
            dtype=fixed_eval_x.dtype,
        )
        x_model = model.decode(z=z_eval).float()
        x_contours = model.decode(z=fixed_prior_points).float()
        contour_statistics = compute_gaussian_drifting_statistics_at_query(
            query_samples=x_contours,
            model_reference_samples=x_model,
            data_samples=fixed_eval_x.float(),
            bandwidth=config.loss.bandwidth,
            objective=config.loss.objective,
            stability_eps=config.loss.stability_eps,
            exclude_model_reference_diagonal=False,
        )
        display_drift, drift_magnitudes = normalize_vectors_for_display(
            vectors=contour_statistics.drift,
            display_length=config.plot.arrow_display_length,
        )
        model_mode_ids = assign_mode_ids(
            points=x_model,
            mode_centers=mode_centers,
        )
        mode_residual = compute_mode_residual(
            points=x_model,
            mode_centers=mode_centers,
        )
        mode_frequencies = torch.bincount(model_mode_ids, minlength=2).float()
        mode_frequencies = mode_frequencies / mode_frequencies.sum().clamp_min(1.0)
        target_mean = torch.tensor(
            config.data.offset,
            device=x_model.device,
            dtype=x_model.dtype,
        )
        metrics = {
            "mode_residual": float(mode_residual.mean().item()),
            "mode_balance_error": float(
                (mode_frequencies - 0.5).abs().max().item()
            ),
            "mean_error": float(
                torch.linalg.vector_norm(x_model.mean(dim=0) - target_mean).item()
            ),
        }
    model.train()

    arrow_indices = torch.arange(
        0,
        x_contours.shape[0],
        config.plot.arrow_stride,
        device=x_contours.device,
    )
    return (
        {
            "step": step,
            "x_data": fixed_eval_x.detach().cpu(),
            "data_mode_ids": fixed_eval_mode_ids.detach().cpu(),
            "x_model": x_model.detach().cpu(),
            "model_mode_ids": model_mode_ids.detach().cpu(),
            "x_contours": x_contours.detach().cpu(),
            "contour_ids": fixed_contour_ids.detach().cpu(),
            "mode_centers": mode_centers.detach().cpu(),
            "arrow_points": x_contours[arrow_indices].detach().cpu(),
            "arrow_vectors": display_drift[arrow_indices].detach().cpu(),
            "arrow_magnitudes": drift_magnitudes[arrow_indices].detach().cpu(),
        },
        metrics,
    )


def plot_drifting_training_history(
    *,
    history: dict[str, list[float]],
    step: int,
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"optimization at step {step}",
            f"distribution diagnostics at step {step}",
        ],
        horizontal_spacing=0.10,
    )
    for metric_name, trace_name in [
        ("total_loss", "total"),
        ("drift_matching_loss", "drift match"),
        ("drift_norm", "drift norm"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=history["step"],
                y=history[metric_name],
                mode="lines",
                name=trace_name,
                line={"width": 2.5},
            ),
            row=1,
            col=1,
        )
    for metric_name, trace_name in [
        ("density_ratio", "density ratio"),
        ("mode_residual", "mode residual"),
        ("mode_balance_error", "balance error"),
        ("mean_error", "mean error"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=history["step"],
                y=history[metric_name],
                mode="lines",
                name=trace_name,
                line={"width": 2.5},
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text="step", row=1, col=1)
    fig.update_xaxes(title_text="step", row=1, col=2)
    fig.update_layout(
        height=420,
        width=1250,
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
    )
    return fig


def _contour_palette(
    *,
    contour_levels: Sequence[float],
) -> list[str]:
    palette_tensor = torch.tensor(
        [
            [0.282623, 0.140926, 0.457517, 1.0],
            [0.253935, 0.265254, 0.529983, 1.0],
            [0.206756, 0.371758, 0.553117, 1.0],
            [0.163625, 0.471133, 0.558148, 1.0],
            [0.127568, 0.566949, 0.550556, 1.0],
            [0.266941, 0.748751, 0.440573, 1.0],
            [0.741388, 0.873449, 0.149561, 1.0],
        ]
    )
    color_indices = torch.linspace(0, 6, len(contour_levels)).round().long()
    return [
        f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"
        for r, g, b, _ in palette_tensor[color_indices].tolist()
    ]


def plot_bimodal_drifting_snapshot(
    *,
    data_points: Float[Tensor, "batch 2"],
    data_mode_ids: Tensor,
    model_points: Float[Tensor, "batch 2"],
    model_mode_ids: Tensor,
    contour_points: Float[Tensor, "prior_batch 2"],
    contour_ids: Tensor,
    contour_levels: Sequence[float],
    mode_centers: Float[Tensor, "2 2"],
    arrow_points: Float[Tensor, "arrow_batch 2"],
    arrow_vectors: Float[Tensor, "arrow_batch 2"],
    arrow_magnitudes: Float[Tensor, "arrow_batch"],
    step: int,
) -> go.Figure:
    data_palette = ["#1f77b4", "#ff7f0e"]
    contour_palette = _contour_palette(contour_levels=contour_levels)

    all_points = torch.cat(
        [data_points, model_points, contour_points, mode_centers],
        dim=0,
    )
    x_min = float(all_points[:, 0].min().item())
    x_max = float(all_points[:, 0].max().item())
    y_min = float(all_points[:, 1].min().item())
    y_max = float(all_points[:, 1].max().item())
    x_pad = 0.08 * max(x_max - x_min, 1.0)
    y_pad = 0.08 * max(y_max - y_min, 1.0)
    x_range = [x_min - x_pad, x_max + x_pad]
    y_range = [y_min - y_pad, y_max + y_pad]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            f"data + model at step {step}",
            f"prior contour pushforward at step {step}",
            f"Gaussian-kernel drift at step {step}",
        ],
        horizontal_spacing=0.05,
    )

    data_points_np = data_points.detach().cpu().float().numpy()
    model_points_np = model_points.detach().cpu().float().numpy()
    contour_points_np = contour_points.detach().cpu().float().numpy()
    mode_centers_np = mode_centers.detach().cpu().float().numpy()
    data_mode_ids_np = data_mode_ids.detach().cpu().numpy()
    model_mode_ids_np = model_mode_ids.detach().cpu().numpy()
    contour_ids_np = contour_ids.detach().cpu().numpy()

    for mode_index, color in enumerate(data_palette):
        data_mask = data_mode_ids_np == mode_index
        model_mask = model_mode_ids_np == mode_index
        fig.add_trace(
            go.Scatter(
                x=data_points_np[data_mask, 0],
                y=data_points_np[data_mask, 1],
                mode="markers",
                marker={"size": 4, "color": color, "opacity": 0.25},
                name=f"data mode {mode_index}",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=model_points_np[model_mask, 0],
                y=model_points_np[model_mask, 1],
                mode="markers",
                marker={"size": 5, "color": color, "opacity": 0.60},
                name=f"model mode {mode_index}",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    for col in [2, 3]:
        fig.add_trace(
            go.Scatter(
                x=data_points_np[:, 0],
                y=data_points_np[:, 1],
                mode="markers",
                marker={"size": 4, "color": "#202020", "opacity": 0.10},
                name="data backdrop",
                showlegend=col == 2,
            ),
            row=1,
            col=col,
        )

    for contour_index, contour_level in enumerate(contour_levels):
        mask = contour_ids_np == contour_index
        contour_name = f"prior contour r={contour_level:.2f}"
        for col in [2, 3]:
            fig.add_trace(
                go.Scatter(
                    x=contour_points_np[mask, 0],
                    y=contour_points_np[mask, 1],
                    mode="markers",
                    marker={
                        "size": 5,
                        "color": contour_palette[contour_index],
                        "opacity": 0.75 if col == 2 else 0.35,
                    },
                    name=contour_name,
                    showlegend=col == 2,
                ),
                row=1,
                col=col,
            )

    for col in [1, 2, 3]:
        fig.add_trace(
            go.Scatter(
                x=mode_centers_np[:, 0],
                y=mode_centers_np[:, 1],
                mode="markers",
                marker={
                    "size": 12,
                    "color": "#111111",
                    "symbol": "x",
                    "line": {"width": 2},
                },
                name="mode centers",
                showlegend=col == 1,
            ),
            row=1,
            col=col,
        )

    add_quiver_traces(
        figure=fig,
        x=arrow_points[:, 0],
        y=arrow_points[:, 1],
        u=arrow_vectors[:, 0],
        v=arrow_vectors[:, 1],
        color="#d62728",
        name="drift",
        row=1,
        col=3,
        magnitudes=arrow_magnitudes,
    )

    for col, y_axis_name in [(1, "y"), (2, "y2"), (3, "y3")]:
        fig.update_xaxes(
            range=x_range,
            scaleanchor=y_axis_name,
            scaleratio=1,
            row=1,
            col=col,
        )
        fig.update_yaxes(range=y_range, row=1, col=col)

    fig.update_layout(
        height=540,
        width=1650,
        margin={"l": 30, "r": 30, "t": 70, "b": 30},
    )
    return fig


def build_snapshot_figures(
    *,
    history: dict[str, list[float]],
    snapshot: dict[str, object],
    config: ExperimentConfig,
) -> dict[str, go.Figure]:
    return {
        "losses": plot_drifting_training_history(
            history=history,
            step=int(snapshot["step"]),
        ),
        "snapshot": plot_bimodal_drifting_snapshot(
            data_points=snapshot["x_data"],
            data_mode_ids=snapshot["data_mode_ids"],
            model_points=snapshot["x_model"],
            model_mode_ids=snapshot["model_mode_ids"],
            contour_points=snapshot["x_contours"],
            contour_ids=snapshot["contour_ids"],
            contour_levels=config.plot.contour_levels,
            mode_centers=snapshot["mode_centers"],
            arrow_points=snapshot["arrow_points"],
            arrow_vectors=snapshot["arrow_vectors"],
            arrow_magnitudes=snapshot["arrow_magnitudes"],
            step=int(snapshot["step"]),
        ),
    }


def save_snapshot_artifacts(
    *,
    figures: dict[str, go.Figure],
    run_root: Path,
    step: int,
) -> Path:
    step_dir = run_root / f"step-{step:07d}"
    save_plotly_figure(
        figure=figures["losses"],
        path=step_dir / "losses.html",
    )
    save_plotly_figure(
        figure=figures["snapshot"],
        path=step_dir / "snapshot.html",
    )
    return step_dir


__all__ = [
    "BimodalGaussianDriftingModel",
    "DriftingLossConfig",
    "ExperimentConfig",
    "PlotConfig",
    "TrainConfig",
    "append_history_entry",
    "build_gaussian_contour_points",
    "build_snapshot_figures",
    "compute_drifting_losses",
    "evaluate_model",
    "initialize_history",
    "make_run_root",
    "mode_centers_tensor",
    "sample_bimodal_gaussian",
    "save_snapshot_artifacts",
]
