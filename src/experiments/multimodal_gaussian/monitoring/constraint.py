from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jaxtyping import Float, Int
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch import Tensor

from src.monitoring.configs import ConstraintMonitorConfig

if TYPE_CHECKING:
    from src.experiments.multimodal_gaussian.state import MultimodalTrainingRuntime


COLOR_BANK = (
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
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
)


def _step_folder(
    *,
    rt: "MultimodalTrainingRuntime",
    step: int,
) -> Path:
    folder = rt.tc.folder / str(step)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _write_figure(
    *,
    figure: go.Figure,
    path_stem: Path,
) -> None:
    figure.write_html(path_stem.with_suffix(".html"))
    figure.write_image(path_stem.with_suffix(".png"))


def _sample_mode_batch(
    *,
    rt: "MultimodalTrainingRuntime",
    batch_size_per_mode: int,
) -> tuple[Float[Tensor, "batch ambient_dim"], Int[Tensor, "batch"]]:
    samples = []
    labels = []
    for mode_id in range(rt.runtime_data_config.num_modes):
        samples.append(
            rt.runtime_data_config.sample_class(
                mode_id=mode_id,
                batch_size=batch_size_per_mode,
            )
        )
        labels.append(
            torch.full(
                (batch_size_per_mode,),
                fill_value=mode_id,
                device=rt.device,
                dtype=torch.long,
            )
        )
    return torch.cat(samples, dim=0), torch.cat(labels, dim=0)


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
    return COLOR_BANK[mode_id % len(COLOR_BANK)]


def _fit_projection_basis(
    *,
    values: Float[Tensor, "batch latent_dim"],
    projection_dim: int,
) -> tuple[Float[Tensor, "1 latent_dim"], Float[Tensor, "latent_dim projection_dim"]]:
    center = values.mean(dim=0, keepdim=True)
    centered = values - center
    _, _, right_singular_vectors = torch.linalg.svd(
        centered,
        full_matrices=False,
    )
    basis = right_singular_vectors[:projection_dim].transpose(0, 1).contiguous()
    if basis.shape[1] < projection_dim:
        padding = torch.zeros(
            basis.shape[0],
            projection_dim - basis.shape[1],
            device=basis.device,
            dtype=basis.dtype,
        )
        basis = torch.cat([basis, padding], dim=1)
    return center, basis


def _project_latents(
    *,
    latents: Float[Tensor, "batch latent_dim"],
    planar: bool,
) -> Float[Tensor, "batch proj_dim"]:
    projection_dim = 2 if planar else 3
    center, basis = _fit_projection_basis(
        values=latents,
        projection_dim=projection_dim,
    )
    return (latents - center) @ basis


def _reconstruction_figure(
    *,
    samples_2d: Float[Tensor, "batch 2"],
    reconstructions_2d: Float[Tensor, "batch 2"],
    reconstruction_error: Float[Tensor, "batch"],
    labels: Int[Tensor, "batch"],
) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Data", "Reconstruction"),
        horizontal_spacing=0.08,
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
                name=f"mode {mode_id}",
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
                name=f"mode {mode_id} recon",
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
    figure.update_xaxes(title="plane-x", row=1, col=1)
    figure.update_xaxes(title="plane-x", row=1, col=2)
    figure.update_yaxes(title="plane-y", scaleanchor="x", scaleratio=1.0, row=1, col=1)
    figure.update_yaxes(title="plane-y", scaleanchor="x2", scaleratio=1.0, row=1, col=2)
    figure.update_layout(
        template="plotly_white",
        width=1100,
        height=520,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
        title="Data-side reconstruction monitor",
    )
    return figure


def _latent_map_figure(
    *,
    projected_latents: Float[Tensor, "batch proj_dim"],
    latent_norms: Float[Tensor, "batch"],
    labels: Int[Tensor, "batch"],
    planar: bool,
) -> go.Figure:
    labels_cpu = labels.cpu()
    projected_cpu = projected_latents.cpu()
    norms_cpu = latent_norms.cpu().unsqueeze(-1).tolist()
    if planar:
        figure = go.Figure()
        for mode_id in range(int(labels_cpu.max().item()) + 1):
            mask = _mode_mask(labels=labels_cpu, mode_id=mode_id)
            if not mask.any():
                continue
            figure.add_trace(
                go.Scatter(
                    x=projected_cpu[mask, 0].tolist(),
                    y=projected_cpu[mask, 1].tolist(),
                    mode="markers",
                    name=f"mode {mode_id}",
                    marker={
                        "size": 6,
                        "color": _marker_color(mode_id=mode_id),
                        "opacity": 0.8,
                    },
                    customdata=[norms_cpu[index] for index in mask.nonzero().flatten().tolist()],
                    hovertemplate=(
                        "mode="
                        + str(mode_id)
                        + "<br>pc1=%{x:.3f}<br>pc2=%{y:.3f}<br>"
                        + "|latent|=%{customdata[0]:.4f}<extra></extra>"
                    ),
                )
            )
        figure.update_xaxes(title="pc1")
        figure.update_yaxes(title="pc2", scaleanchor="x", scaleratio=1.0)
    else:
        figure = go.Figure()
        for mode_id in range(int(labels_cpu.max().item()) + 1):
            mask = _mode_mask(labels=labels_cpu, mode_id=mode_id)
            if not mask.any():
                continue
            figure.add_trace(
                go.Scatter3d(
                    x=projected_cpu[mask, 0].tolist(),
                    y=projected_cpu[mask, 1].tolist(),
                    z=projected_cpu[mask, 2].tolist(),
                    mode="markers",
                    name=f"mode {mode_id}",
                    marker={
                        "size": 4,
                        "color": _marker_color(mode_id=mode_id),
                        "opacity": 0.8,
                    },
                    customdata=[norms_cpu[index] for index in mask.nonzero().flatten().tolist()],
                    hovertemplate=(
                        "mode="
                        + str(mode_id)
                        + "<br>pc1=%{x:.3f}<br>pc2=%{y:.3f}<br>pc3=%{z:.3f}<br>"
                        + "|latent|=%{customdata[0]:.4f}<extra></extra>"
                    ),
                )
            )
        figure.update_scenes(
            xaxis_title="pc1",
            yaxis_title="pc2",
            zaxis_title="pc3",
        )
    figure.update_layout(
        template="plotly_white",
        width=900,
        height=900,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        title="Data latent map",
    )
    return figure


class GaussianConstraintMonitorConfig(ConstraintMonitorConfig):
    def apply_to(
        self,
        *,
        rt: "MultimodalTrainingRuntime",
        step: int,
    ) -> dict[str, float]:
        with torch.no_grad():
            samples, labels = _sample_mode_batch(
                rt=rt,
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

            latent_samples, latent_labels = _sample_mode_batch(
                rt=rt,
                batch_size_per_mode=self.n_data_latents_per_mode,
            )
            latent_values = rt.chart_transport_model.encoder(latent_samples).float()
            projected_latents = _project_latents(
                latents=latent_values,
                planar=self.planar,
            )
            latent_norms = latent_values.norm(dim=-1).float()

        folder = _step_folder(rt=rt, step=step)
        _write_figure(
            figure=_reconstruction_figure(
                samples_2d=samples_2d,
                reconstructions_2d=reconstructions_2d,
                reconstruction_error=reconstruction_error,
                labels=labels,
            ),
            path_stem=folder / "data_reconstruction",
        )
        _write_figure(
            figure=_latent_map_figure(
                projected_latents=projected_latents,
                latent_norms=latent_norms,
                labels=latent_labels,
                planar=self.planar,
            ),
            path_stem=folder / "data_latent_map",
        )
        return {
            "constraint_reconstruction_mean": reconstruction_error.mean().item(),
            "constraint_reconstruction_max": reconstruction_error.max().item(),
            "constraint_latent_norm_mean": latent_norms.mean().item(),
        }


__all__ = ["GaussianConstraintMonitorConfig"]
