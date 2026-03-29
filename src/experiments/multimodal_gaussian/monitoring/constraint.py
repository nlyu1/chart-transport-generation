from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jaxtyping import Float, Int
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch import Tensor

from src.monitoring.configs import ConstraintMonitorConfig
from src.monitoring.utils import (
    marker_color,
    sample_mode_batch,
    step_folder,
    write_figure,
    write_mode_value_parquet,
)

if TYPE_CHECKING:
    from src.experiments.multimodal_gaussian.state import MultimodalTrainingRuntime


def _mode_mask(
    *,
    labels: Int[Tensor, "batch"],
    mode_id: int,
) -> Tensor:
    return labels == mode_id


def _marker_color(
    *,
    mode_id: int,
) -> str:
    return marker_color(group_id=mode_id)


def _reconstruction_figure(
    *,
    samples_2d: Float[Tensor, "batch 2"],
    reconstructions_2d: Float[Tensor, "batch 2"],
    reconstruction_error: Float[Tensor, "batch"],
    labels: Int[Tensor, "batch"],
) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=3,
        horizontal_spacing=0.08,
        column_widths=[0.34, 0.34, 0.32],
    )
    labels_cpu = labels.cpu()
    samples_cpu = samples_2d.cpu()
    reconstructions_cpu = reconstructions_2d.cpu()
    error_cpu = reconstruction_error.cpu()
    for mode_id in range(int(labels_cpu.max().item()) + 1):
        mask = _mode_mask(labels=labels_cpu, mode_id=mode_id)
        if not mask.any():
            continue
        color = _marker_color(mode_id=mode_id)
        customdata = error_cpu[mask].unsqueeze(-1).tolist()
        sample_x = samples_cpu[mask, 0].tolist()
        sample_y = samples_cpu[mask, 1].tolist()
        recon_x = reconstructions_cpu[mask, 0].tolist()
        recon_y = reconstructions_cpu[mask, 1].tolist()
        figure.add_trace(
            go.Scatter(
                x=sample_x,
                y=sample_y,
                mode="markers",
                name=str(mode_id),
                legendgroup=str(mode_id),
                marker={"size": 6, "color": color, "opacity": 0.8},
                customdata=customdata,
                hovertemplate=(
                    "mode="
                    + str(mode_id)
                    + "<br>x=%{x:.3f}<br>y=%{y:.3f}<br>"
                    + "recon_error=%{customdata[0]:.4f}<extra></extra>"
                ),
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=recon_x,
                y=recon_y,
                mode="markers",
                name=str(mode_id),
                legendgroup=str(mode_id),
                marker={
                    "size": 6,
                    "color": color,
                    "opacity": 0.8,
                    "symbol": "x",
                    "line": {"width": 1.2},
                },
                customdata=customdata,
                hovertemplate=(
                    "mode="
                    + str(mode_id)
                    + "<br>x=%{x:.3f}<br>y=%{y:.3f}<br>"
                    + "recon_error=%{customdata[0]:.4f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        figure.add_trace(
            go.Histogram(
                x=error_cpu[mask].tolist(),
                name=str(mode_id),
                legendgroup=str(mode_id),
                marker={"color": color},
                opacity=0.65,
                showlegend=False,
                hovertemplate=(
                    "mode="
                    + str(mode_id)
                    + "<br>recon_error=%{x:.4f}<br>count=%{y}<extra></extra>"
                ),
            ),
            row=1,
            col=3,
        )
    figure.update_yaxes(scaleanchor="x", scaleratio=1.0, row=1, col=1)
    figure.update_yaxes(scaleanchor="x2", scaleratio=1.0, row=1, col=2)
    figure.update_layout(
        template="plotly_white",
        width=1500,
        height=520,
        barmode="overlay",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
    )
    return figure


def _write_reconstruction_error_parquet(
    *,
    step_root: Path,
    labels: Int[Tensor, "batch"],
    reconstruction_error: Float[Tensor, "batch"],
) -> None:
    write_mode_value_parquet(
        path=(
            step_root
            / "numbers"
            / "constraint_reconstruction"
            / "reconstruction_error_norms.parquet"
        ),
        mode_ids=labels,
        value_column_name="reconstruction_error_norm",
        values=reconstruction_error,
    )


class GaussianConstraintMonitorConfig(ConstraintMonitorConfig):
    def apply_to(
        self,
        *,
        rt: "MultimodalTrainingRuntime",
        step: int,
    ) -> dict[str, float]:
        with torch.no_grad():
            samples, labels = sample_mode_batch(
                data_config=rt.runtime_data_config,
                device=rt.device,
                batch_size_per_mode=self.n_sample_pairs_per_mode,
            )
            latents = rt.chart_transport_model.encoder(samples).float()
            reconstructions = rt.chart_transport_model.decoder(latents).float()

            samples_2d, _, _ = rt.runtime_data_config.decompose_projection(samples.float())
            reconstructions_2d, _, _ = rt.runtime_data_config.decompose_projection(
                reconstructions
            )
            reconstruction_error = (
                (reconstructions - samples)
                .reshape(samples.shape[0], -1)
                .norm(dim=-1)
                .float()
            )

            latent_samples, latent_labels = sample_mode_batch(
                data_config=rt.runtime_data_config,
                device=rt.device,
                batch_size_per_mode=self.n_data_latents_per_mode,
            )
            latent_values = rt.chart_transport_model.encoder(latent_samples).float()
            latent_norms = latent_values.norm(dim=-1).float()

        folder = step_folder(run_folder=rt.tc.folder, step=step)
        write_figure(
            figure=_reconstruction_figure(
                samples_2d=samples_2d,
                reconstructions_2d=reconstructions_2d,
                reconstruction_error=reconstruction_error,
                labels=labels,
            ),
            path_stem=folder / "data_reconstruction",
        )
        _write_reconstruction_error_parquet(
            step_root=folder,
            labels=labels,
            reconstruction_error=reconstruction_error,
        )
        self.save_latent_plot_to(
            latents=latent_values,
            mode_ids=latent_labels,
            save_to_folder=folder,
        )
        return {
            "constraint_reconstruction_mean": reconstruction_error.mean().item(),
            "constraint_reconstruction_max": reconstruction_error.max().item(),
            "constraint_latent_norm_mean": latent_norms.mean().item(),
        }


__all__ = ["GaussianConstraintMonitorConfig"]
