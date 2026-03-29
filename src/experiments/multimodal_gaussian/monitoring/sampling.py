from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jaxtyping import Float, Int
import plotly.graph_objects as go
import polars as pl
import torch
from torch import Tensor

from src.monitoring.configs import SamplingMonitorConfig
from src.monitoring.utils import (
    MonitorStage,
    latent_square_limits,
    marker_color,
    sample_mode_batch,
    step_folder,
    write_figure,
)

if TYPE_CHECKING:
    from src.experiments.multimodal_gaussian.state import MultimodalTrainingRuntime


def _generated_samples_figure(
    *,
    data_samples_2d: Float[Tensor, "batch 2"],
    data_labels: Int[Tensor, "batch"],
    generated_samples_2d: Float[Tensor, "batch 2"],
) -> go.Figure:
    figure = go.Figure()
    data_labels_cpu = data_labels.cpu()
    data_samples_cpu = data_samples_2d.cpu()
    generated_samples_cpu = generated_samples_2d.cpu()

    for mode_id in range(int(data_labels_cpu.max().item()) + 1):
        mask = data_labels_cpu == mode_id
        if not mask.any():
            continue
        figure.add_trace(
            go.Scatter(
                x=data_samples_cpu[mask, 0].tolist(),
                y=data_samples_cpu[mask, 1].tolist(),
                mode="markers",
                name=f"mode {mode_id}",
                marker={
                    "size": 5,
                    "opacity": 0.3,
                    "color": marker_color(group_id=mode_id),
                    "line": {"width": 0.0},
                },
                hovertemplate=(
                    "mode="
                    + str(mode_id)
                    + "<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>"
                ),
            )
        )

    figure.add_trace(
        go.Scatter(
            x=generated_samples_cpu[:, 0].tolist(),
            y=generated_samples_cpu[:, 1].tolist(),
            mode="markers",
            name="generated",
            marker={
                "size": 6,
                "opacity": 0.55,
                "color": "rgba(20, 20, 20, 0.8)",
                "line": {"width": 0.0},
            },
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        )
    )
    all_points = torch.cat([data_samples_2d, generated_samples_2d], dim=0)
    x_min, x_max, y_min, y_max = latent_square_limits(
        all_points,
        padding=0.18,
    )
    figure.update_layout(
        template="plotly_white",
        width=900,
        height=900,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        xaxis={"range": [x_min, x_max]},
        yaxis={"range": [y_min, y_max], "scaleanchor": "x", "scaleratio": 1.0},
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        title="Generated samples vs data",
    )
    return figure


def _kde_scale_label(
    *,
    kde_scale: float,
) -> str:
    return f"{kde_scale:g}".replace(".", "p").replace("-", "m")


def _projected_kl_figure(
    *,
    approximate_kl: Float[Tensor, "kde_scales mode_id"],
    kde_scales: list[float],
) -> go.Figure:
    figure = go.Figure()
    approximate_kl_cpu = approximate_kl.cpu()
    mode_ids = list(range(approximate_kl_cpu.shape[1]))

    for scale_index, kde_scale in enumerate(kde_scales):
        figure.add_trace(
            go.Bar(
                x=mode_ids,
                y=approximate_kl_cpu[scale_index].tolist(),
                name=f"scale_per_dim={kde_scale:g}",
                hovertemplate=(
                    "mode=%{x}<br>projected_kl=%{y:.4f}<br>"
                    f"kde_scale_per_dimension={kde_scale:g}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        template="plotly_white",
        width=1000,
        height=600,
        barmode="group",
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        title="Projected KL by mode and KDE scale",
        xaxis={"title": "mode_id"},
        yaxis={"title": "projected KL"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    return figure


def _write_projected_kl_parquet(
    *,
    step_root: Path,
    approximate_kl: Float[Tensor, "kde_scales mode_id"],
    kde_scales: list[float],
) -> None:
    path = step_root / "numbers" / "projected_kl.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    approximate_kl_cpu = approximate_kl.detach().cpu().float()
    mode_ids = []
    kde_scale_per_dimension = []
    projected_kl = []
    for scale_index, scale_value in enumerate(kde_scales):
        for mode_id in range(approximate_kl_cpu.shape[1]):
            mode_ids.append(mode_id)
            kde_scale_per_dimension.append(float(scale_value))
            projected_kl.append(float(approximate_kl_cpu[scale_index, mode_id].item()))
    pl.DataFrame(
        {
            "mode_id": mode_ids,
            "kde_scale_per_dimension": kde_scale_per_dimension,
            "projected_kl": projected_kl,
        },
        schema={
            "mode_id": pl.Int64,
            "kde_scale_per_dimension": pl.Float32,
            "projected_kl": pl.Float32,
        },
    ).sort(
        by=["kde_scale_per_dimension", "mode_id"],
    ).write_parquet(path)


class GaussianSamplingMonitorConfig(SamplingMonitorConfig):
    kde_scales: list[float]
    kl_num_samples: int
    avg_kl_num_batches: int

    def apply_to(
        self,
        *,
        rt: "MultimodalTrainingRuntime",
        step: int,
        stage: MonitorStage,
    ) -> dict[str, float]:
        with torch.no_grad():
            generated_prior = rt.tc.chart_transport_config.prior_config.sample(
                batch_size=self.n_generated_samples,
            ).to(device=rt.device, dtype=torch.float32)
            generated_samples = rt.chart_transport_model.decoder(generated_prior).float()
            approximate_kl = rt.runtime_data_config.approximate_kl(
                data_samples=generated_samples,
                kde_scales=self.kde_scales,
                kl_num_samples=self.kl_num_samples,
                avg_kl_num_batches=self.avg_kl_num_batches,
            ).float()

            data_samples, data_labels = sample_mode_batch(
                data_config=rt.runtime_data_config,
                device=rt.device,
                batch_size_per_mode=self.n_data_samples_per_mode,
            )
            data_samples_2d, _, _ = rt.runtime_data_config.decompose_projection(
                data_samples.float()
            )
            generated_samples_2d, _, _ = rt.runtime_data_config.decompose_projection(
                generated_samples
            )

        folder = step_folder(
            run_folder=rt.tc.folder,
            stage=stage,
            step=step,
        )
        write_figure(
            figure=_generated_samples_figure(
                data_samples_2d=data_samples_2d,
                data_labels=data_labels,
                generated_samples_2d=generated_samples_2d,
            ),
            path_stem=folder / "generated_samples",
        )
        write_figure(
            figure=_projected_kl_figure(
                approximate_kl=approximate_kl,
                kde_scales=self.kde_scales,
            ),
            path_stem=folder / "projected_kl",
        )
        _write_projected_kl_parquet(
            step_root=folder,
            approximate_kl=approximate_kl,
            kde_scales=self.kde_scales,
        )
        return {
            f"projected_kl_mean_{_kde_scale_label(kde_scale=kde_scale)}": approximate_kl[
                scale_index
            ]
            .mean()
            .item()
            for scale_index, kde_scale in enumerate(self.kde_scales)
        }


__all__ = ["GaussianSamplingMonitorConfig"]
