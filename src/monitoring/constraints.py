from __future__ import annotations

from pathlib import Path

from jaxtyping import Float, Int
import plotly.graph_objects as go
import torch
from torch import Tensor

from src.monitoring.configs import ConstraintMonitorConfig
from src.monitoring.utils import marker_color, write_figure


def _mode_mask(
    *,
    mode_ids: Int[Tensor, "batch"],
    mode_id: int,
) -> Tensor:
    return mode_ids == mode_id


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
) -> tuple[Float[Tensor, "batch proj_dim"], Float[Tensor, "batch"]]:
    projection_dim = 2 if planar else 3
    center, basis = _fit_projection_basis(
        values=latents,
        projection_dim=projection_dim,
    )
    projected_latents = (latents - center) @ basis
    reconstructed_latents = center + projected_latents @ basis.transpose(0, 1)
    off_plane_norm = (latents - reconstructed_latents).norm(dim=-1).float()
    return projected_latents, off_plane_norm


def latent_map_figure(
    *,
    latents: Float[Tensor, "batch latent_dim"],
    mode_ids: Int[Tensor, "batch"],
    planar: bool,
) -> go.Figure:
    projected_latents, off_plane_norm = _project_latents(
        latents=latents,
        planar=planar,
    )
    latent_norms = latents.norm(dim=-1).float()

    mode_ids_cpu = mode_ids.detach().cpu().long()
    projected_cpu = projected_latents.detach().cpu().float()
    latent_norms_cpu = latent_norms.detach().cpu().float()
    off_plane_norm_cpu = off_plane_norm.detach().cpu().float()

    figure = go.Figure()
    for mode_id in range(int(mode_ids_cpu.max().item()) + 1):
        mask = _mode_mask(mode_ids=mode_ids_cpu, mode_id=mode_id)
        if not mask.any():
            continue
        customdata = torch.stack(
            [
                latent_norms_cpu[mask],
                off_plane_norm_cpu[mask],
            ],
            dim=-1,
        ).tolist()
        if planar:
            figure.add_trace(
                go.Scatter(
                    x=projected_cpu[mask, 0].tolist(),
                    y=projected_cpu[mask, 1].tolist(),
                    mode="markers",
                    name=str(mode_id),
                    marker={
                        "size": 6,
                        "color": marker_color(group_id=mode_id),
                        "opacity": 0.8,
                    },
                    customdata=customdata,
                    hovertemplate=(
                        "mode="
                        + str(mode_id)
                        + "<br>pc1=%{x:.3f}<br>pc2=%{y:.3f}<br>"
                        + "|latent|=%{customdata[0]:.4f}<br>"
                        + "off_plane_norm=%{customdata[1]:.4f}<extra></extra>"
                    ),
                )
            )
        else:
            figure.add_trace(
                go.Scatter3d(
                    x=projected_cpu[mask, 0].tolist(),
                    y=projected_cpu[mask, 1].tolist(),
                    z=projected_cpu[mask, 2].tolist(),
                    mode="markers",
                    name=str(mode_id),
                    marker={
                        "size": 4,
                        "color": marker_color(group_id=mode_id),
                        "opacity": 0.8,
                    },
                    customdata=customdata,
                    hovertemplate=(
                        "mode="
                        + str(mode_id)
                        + "<br>pc1=%{x:.3f}<br>pc2=%{y:.3f}<br>pc3=%{z:.3f}<br>"
                        + "|latent|=%{customdata[0]:.4f}<br>"
                        + "off_plane_norm=%{customdata[1]:.4f}<extra></extra>"
                    ),
                )
            )

    if planar:
        figure.update_xaxes(title="pc1")
        figure.update_yaxes(title="pc2", scaleanchor="x", scaleratio=1.0)
    else:
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
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    return figure


def save_latent_plot_to(
    *,
    config: ConstraintMonitorConfig,
    latents: Float[Tensor, "batch latent_dim"],
    mode_ids: Int[Tensor, "batch"],
    save_to_folder: Path,
) -> None:
    figure = latent_map_figure(
        latents=latents,
        mode_ids=mode_ids,
        planar=config.planar,
    )
    write_figure(
        figure=figure,
        path_stem=save_to_folder / "data_latent_map",
    )


__all__ = ["latent_map_figure", "save_latent_plot_to"]
