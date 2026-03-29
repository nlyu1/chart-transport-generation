from __future__ import annotations

from typing import TYPE_CHECKING

from jaxtyping import Float, Int
import plotly.graph_objects as go
import torch
from torch import Tensor

from src.monitoring.configs import SamplingMonitorConfig
from src.monitoring.utils import (
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
    generated_log_likelihood: Float[Tensor, "batch"],
) -> go.Figure:
    figure = go.Figure()
    data_labels_cpu = data_labels.cpu()
    data_samples_cpu = data_samples_2d.cpu()
    generated_samples_cpu = generated_samples_2d.cpu()
    generated_log_likelihood_cpu = generated_log_likelihood.cpu()

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
            customdata=generated_log_likelihood_cpu.unsqueeze(-1).tolist(),
            hovertemplate=(
                "x=%{x:.3f}<br>y=%{y:.3f}<br>"
                "log_likelihood=%{customdata[0]:.4f}<extra></extra>"
            ),
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


class GaussianSamplingMonitorConfig(SamplingMonitorConfig):
    def apply_to(
        self,
        *,
        rt: "MultimodalTrainingRuntime",
        step: int,
    ) -> dict[str, float]:
        with torch.no_grad():
            generated_prior = rt.tc.chart_transport_config.prior_config.sample(
                batch_size=self.n_generated_samples,
            ).to(device=rt.device, dtype=torch.float32)
            generated_samples = rt.chart_transport_model.decoder(generated_prior).float()
            generated_log_likelihood = rt.runtime_data_config.log_likelihood(
                generated_samples
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

        folder = step_folder(run_folder=rt.tc.folder, step=step)
        write_figure(
            figure=_generated_samples_figure(
                data_samples_2d=data_samples_2d,
                data_labels=data_labels,
                generated_samples_2d=generated_samples_2d,
                generated_log_likelihood=generated_log_likelihood,
            ),
            path_stem=folder / "generated_samples",
        )
        return {
            "sampling_generated_log_likelihood_mean": generated_log_likelihood.mean().item(),
            "sampling_generated_log_likelihood_std": generated_log_likelihood.std(
                unbiased=False
            ).item(),
        }


__all__ = ["GaussianSamplingMonitorConfig"]
