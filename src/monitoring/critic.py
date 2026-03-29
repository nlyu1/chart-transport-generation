from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jaxtyping import Float, Int
import plotly.graph_objects as go
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
import torch
import torch.nn as nn
from torch import Tensor

from src.chart_transport.transport_loss import TransportLossConfig
from src.monitoring.utils import (
    build_latent_grid,
    critic_score_from_noise_prediction,
    flatten_latents,
    latent_square_limits,
    marker_color,
    normalize_vectors,
    project_latents_to_pca_space,
    project_latent_vectors_to_pca_space,
    sample_mode_batch,
    step_folder,
    vector_display_length,
    write_figure,
)
from src.priors.base import BasePriorConfig

if TYPE_CHECKING:
    from src.monitoring.configs import CriticMonitorConfig


@dataclass(
    config=ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    ),
    kw_only=True,
)
class CriticSnapshot:
    t_value: float
    cloud_latents: Float[Tensor, "batch latent_dim"]
    arrow_latents: Float[Tensor, "batch latent_dim"]
    data_score: Float[Tensor, "batch latent_dim"]


ARROW_DISPLAY_SCALE = 2.0 / 3.0


def sample_critic_snapshot(
    *,
    critic: nn.Module,
    clean_latents: Float[Tensor, "batch latent_dim"],
    t_value: float,
) -> tuple[
    Float[Tensor, "batch latent_dim"],
    Float[Tensor, "batch latent_dim"],
]:
    t = torch.full(
        (clean_latents.shape[0],),
        float(t_value),
        device=clean_latents.device,
        dtype=torch.float32,
    )
    noise = torch.randn_like(clean_latents)
    noised_latents = (1.0 - t).unsqueeze(-1) * clean_latents + t.unsqueeze(-1) * noise
    predicted_noise = critic(noised_latents, t).float()
    data_score = critic_score_from_noise_prediction(
        predicted_noise=predicted_noise,
        t=t,
    )
    return noised_latents.float(), data_score.float()


def estimate_clean_transport_field(
    *,
    critic: nn.Module,
    prior_config: BasePriorConfig,
    transport_config: TransportLossConfig,
    clean_latents: Float[Tensor, "batch latent_dim"],
    t_values: list[float],
) -> Float[Tensor, "batch latent_dim"]:
    if len(t_values) == 0:
        raise ValueError("t_values must be non-empty")

    transport_field = torch.zeros_like(clean_latents, dtype=torch.float32)
    for t_value in t_values:
        t = torch.full(
            (clean_latents.shape[0],),
            float(t_value),
            device=clean_latents.device,
            dtype=torch.float32,
        )
        pullback_weight = transport_config.kl_weight_schedule.pullback_weight(
            t.float(),
        ).unsqueeze(-1)
        noise = torch.randn_like(clean_latents)

        def evaluate_with_noise(
            *,
            sampled_noise: Float[Tensor, "batch latent_dim"],
        ) -> Float[Tensor, "batch latent_dim"]:
            noised_latents = (
                (1.0 - t).unsqueeze(-1) * clean_latents
                + t.unsqueeze(-1) * sampled_noise
            )
            predicted_noise = critic(noised_latents, t).float()
            prior_score = prior_config.analytic_score(
                noised_latents.float(),
                t.float(),
            ).float()
            return pullback_weight * (
                prior_score + predicted_noise / t.unsqueeze(-1)
            )

        transport_terms = evaluate_with_noise(sampled_noise=noise)
        if transport_config.antipodal_estimate:
            transport_terms = 0.5 * (
                transport_terms + evaluate_with_noise(sampled_noise=-noise)
            )
        transport_field = transport_field + transport_terms

    return transport_field / len(t_values)


def projection_dim(
    *,
    planar: bool,
) -> int:
    return 2 if planar else 3


def add_mode_scatter_traces(
    *,
    figure: go.Figure,
    points: Float[Tensor, "batch projection_dim"],
    mode_ids: Int[Tensor, "batch"],
    size: float,
    opacity: float,
    showlegend: bool,
    latent_norms: Float[Tensor, "batch"],
    off_plane_norms: Float[Tensor, "batch"],
    planar: bool,
) -> None:
    mode_ids_long = mode_ids.to(dtype=torch.long)
    mode_ids_float = mode_ids.to(dtype=torch.float32)
    latent_norms_float = latent_norms.to(dtype=torch.float32)
    off_plane_norms_float = off_plane_norms.to(dtype=torch.float32)

    for mode_id in range(int(mode_ids_long.max().item()) + 1):
        mask = mode_ids_long == mode_id
        if not mask.any():
            continue
        customdata = torch.stack(
            [
                mode_ids_float[mask],
                latent_norms_float[mask],
                off_plane_norms_float[mask],
            ],
            dim=-1,
        ).tolist()
        marker = {
            "size": size,
            "opacity": opacity,
            "color": marker_color(group_id=mode_id),
            "line": {"width": 0.0},
        }
        if planar:
            figure.add_trace(
                go.Scatter(
                    x=points[mask, 0].tolist(),
                    y=points[mask, 1].tolist(),
                    mode="markers",
                    name=f"mode {mode_id}",
                    showlegend=showlegend,
                    marker=marker,
                    customdata=customdata,
                    hovertemplate=(
                        "mode=%{customdata[0]:.0f}<br>"
                        "pc1=%{x:.3f}<br>"
                        "pc2=%{y:.3f}<br>"
                        "|latent|=%{customdata[1]:.4f}<br>"
                        "off_pca_norm=%{customdata[2]:.4f}<extra></extra>"
                    ),
                )
            )
            continue
        figure.add_trace(
            go.Scatter3d(
                x=points[mask, 0].tolist(),
                y=points[mask, 1].tolist(),
                z=points[mask, 2].tolist(),
                mode="markers",
                name=f"mode {mode_id}",
                showlegend=showlegend,
                marker=marker,
                customdata=customdata,
                hovertemplate=(
                    "mode=%{customdata[0]:.0f}<br>"
                    "pc1=%{x:.3f}<br>"
                    "pc2=%{y:.3f}<br>"
                    "pc3=%{z:.3f}<br>"
                    "|latent|=%{customdata[1]:.4f}<br>"
                    "off_pca_norm=%{customdata[2]:.4f}<extra></extra>"
                ),
            )
        )


def segment_coordinates(
    *,
    starts: Float[Tensor, "batch projection_dim"],
    vectors: Float[Tensor, "batch projection_dim"],
) -> tuple[list[float], list[float], list[float]]:
    stops = starts + vectors
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    separator = float("nan")
    for index in range(starts.shape[0]):
        xs.extend([float(starts[index, 0]), float(stops[index, 0]), separator])
        ys.extend([float(starts[index, 1]), float(stops[index, 1]), separator])
        if starts.shape[1] == 2:
            zs.extend([0.0, 0.0, separator])
            continue
        zs.extend([float(starts[index, 2]), float(stops[index, 2]), separator])
    return xs, ys, zs


def add_mode_vector_traces(
    *,
    figure: go.Figure,
    points: Float[Tensor, "batch projection_dim"],
    vectors: Float[Tensor, "batch projection_dim"],
    vector_norms: Float[Tensor, "batch"],
    off_plane_norms: Float[Tensor, "batch"],
    mode_ids: Int[Tensor, "batch"],
    planar: bool,
) -> None:
    mode_ids_long = mode_ids.to(dtype=torch.long)
    for mode_id in range(int(mode_ids_long.max().item()) + 1):
        mask = mode_ids_long == mode_id
        if not mask.any():
            continue
        point_subset = points[mask]
        vector_subset = vectors[mask]
        vector_norm_subset = vector_norms[mask]
        off_plane_norm_subset = off_plane_norms[mask]
        stop_subset = point_subset + vector_subset
        xs, ys, zs = segment_coordinates(
            starts=point_subset,
            vectors=vector_subset,
        )
        if planar:
            figure.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line={"color": marker_color(group_id=mode_id), "width": 1.4},
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"mode {mode_id}",
                )
            )
            arrow_angles = (
                torch.atan2(vector_subset[:, 1], vector_subset[:, 0])
                * (180.0 / torch.pi)
                - 90.0
            )
            figure.add_trace(
                go.Scatter(
                    x=stop_subset[:, 0].tolist(),
                    y=stop_subset[:, 1].tolist(),
                    mode="markers",
                    marker={
                        "size": 9,
                        "symbol": "triangle-up",
                        "angle": arrow_angles.tolist(),
                        "color": marker_color(group_id=mode_id),
                        "opacity": 0.9,
                        "line": {"width": 0.0},
                    },
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"mode {mode_id}",
                )
            )
            midpoints = point_subset + 0.5 * vector_subset
            figure.add_trace(
                go.Scatter(
                    x=midpoints[:, 0].tolist(),
                    y=midpoints[:, 1].tolist(),
                    mode="markers",
                    marker={
                        "size": 10,
                        "color": marker_color(group_id=mode_id),
                        "opacity": 0.18,
                        "line": {"width": 0.0},
                    },
                    customdata=torch.stack(
                        [
                            vector_norm_subset.to(dtype=torch.float32),
                            off_plane_norm_subset.to(dtype=torch.float32),
                        ],
                        dim=-1,
                    ).tolist(),
                    hovertemplate=(
                        "arrow_norm=%{customdata[0]:.4f}<br>"
                        "off_pca_norm=%{customdata[1]:.4f}<extra></extra>"
                    ),
                    showlegend=False,
                    name=f"mode {mode_id}",
                )
            )
            continue
        figure.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line={"color": marker_color(group_id=mode_id), "width": 5},
                hoverinfo="skip",
                showlegend=False,
                name=f"mode {mode_id}",
            )
        )
        cone_vectors = 0.24 * vector_subset
        cone_sizeref = float(
            cone_vectors.norm(dim=-1).mean().item()
        )
        figure.add_trace(
            go.Cone(
                x=stop_subset[:, 0].tolist(),
                y=stop_subset[:, 1].tolist(),
                z=stop_subset[:, 2].tolist(),
                u=cone_vectors[:, 0].tolist(),
                v=cone_vectors[:, 1].tolist(),
                w=cone_vectors[:, 2].tolist(),
                anchor="tip",
                sizemode="absolute",
                sizeref=max(cone_sizeref, 1e-6),
                showscale=False,
                colorscale=[
                    [0.0, marker_color(group_id=mode_id)],
                    [1.0, marker_color(group_id=mode_id)],
                ],
                hoverinfo="skip",
                showlegend=False,
                name=f"mode {mode_id}",
            )
        )
        midpoints = point_subset + 0.5 * vector_subset
        figure.add_trace(
            go.Scatter3d(
                x=midpoints[:, 0].tolist(),
                y=midpoints[:, 1].tolist(),
                z=midpoints[:, 2].tolist(),
                mode="markers",
                marker={
                    "size": 4,
                    "color": marker_color(group_id=mode_id),
                    "opacity": 0.22,
                    "line": {"width": 0.0},
                },
                customdata=torch.stack(
                    [
                        vector_norm_subset.to(dtype=torch.float32),
                        off_plane_norm_subset.to(dtype=torch.float32),
                    ],
                    dim=-1,
                ).tolist(),
                hovertemplate=(
                    "arrow_norm=%{customdata[0]:.4f}<br>"
                    "off_pca_norm=%{customdata[1]:.4f}<extra></extra>"
                ),
                showlegend=False,
                name=f"mode {mode_id}",
            )
        )


def critic_score_snapshot_figure(
    *,
    reference_latents: Float[Tensor, "batch latent_dim"],
    snapshot: CriticSnapshot,
    dense_mode_ids: Int[Tensor, "batch"],
    vector_mode_ids: Int[Tensor, "batch"],
    planar: bool,
) -> go.Figure:
    projection_dimension = projection_dim(planar=planar)
    projected_cloud_latents, cloud_off_plane_norm = project_latents_to_pca_space(
        reference_points=reference_latents,
        points=snapshot.cloud_latents,
        projection_dim=projection_dimension,
    )
    projected_arrow_latents, arrow_off_plane_norm = project_latents_to_pca_space(
        reference_points=reference_latents,
        points=snapshot.arrow_latents,
        projection_dim=projection_dimension,
    )
    projected_data_score = project_latent_vectors_to_pca_space(
        reference_points=reference_latents,
        vectors=snapshot.data_score,
        projection_dim=projection_dimension,
    )
    cloud_latent_norm = flatten_latents(snapshot.cloud_latents).norm(dim=-1)
    arrow_latent_norm = flatten_latents(snapshot.arrow_latents).norm(dim=-1)
    data_score_norm = flatten_latents(snapshot.data_score).norm(dim=-1)

    figure = go.Figure()
    add_mode_scatter_traces(
        figure=figure,
        points=projected_cloud_latents,
        mode_ids=dense_mode_ids,
        size=5,
        opacity=0.26,
        showlegend=True,
        latent_norms=cloud_latent_norm,
        off_plane_norms=cloud_off_plane_norm,
        planar=planar,
    )
    add_mode_scatter_traces(
        figure=figure,
        points=projected_arrow_latents,
        mode_ids=vector_mode_ids,
        size=6,
        opacity=0.48,
        showlegend=False,
        latent_norms=arrow_latent_norm,
        off_plane_norms=arrow_off_plane_norm,
        planar=planar,
    )
    display_vectors = normalize_vectors(
        vectors=projected_data_score,
        display_length=vector_display_length(
            projected_cloud_latents,
            fraction=0.02,
        ),
    ) * ARROW_DISPLAY_SCALE
    add_mode_vector_traces(
        figure=figure,
        points=projected_arrow_latents,
        vectors=display_vectors,
        vector_norms=data_score_norm,
        off_plane_norms=arrow_off_plane_norm,
        mode_ids=vector_mode_ids,
        planar=planar,
    )
    figure.update_layout(
        title=(
            f"Critic score snapshot at t={snapshot.t_value:.2f} "
            f"({'2D' if planar else '3D'} PCA view)"
        ),
        template="plotly_white",
        width=900,
        height=900,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    if planar:
        x_min, x_max, y_min, y_max = latent_square_limits(
            projected_cloud_latents,
            padding=0.18,
        )
        figure.update_xaxes(range=[x_min, x_max], title="pc1")
        figure.update_yaxes(
            range=[y_min, y_max],
            title="pc2",
            scaleanchor="x",
            scaleratio=1.0,
        )
    else:
        figure.update_scenes(
            xaxis_title="pc1",
            yaxis_title="pc2",
            zaxis_title="pc3",
        )
    return figure


def critic_transport_figure(
    *,
    reference_latents: Float[Tensor, "batch latent_dim"],
    dense_latents: Float[Tensor, "batch latent_dim"],
    dense_mode_ids: Int[Tensor, "batch"],
    vector_latents: Float[Tensor, "batch latent_dim"],
    vector_mode_ids: Int[Tensor, "batch"],
    transport_field: Float[Tensor, "batch latent_dim"],
    planar: bool,
    transport_grid_xs: Tensor | None,
    transport_grid_ys: Tensor | None,
    transport_grid_projection: Tensor | None,
    num_contour_lines: int,
) -> go.Figure:
    projection_dimension = projection_dim(planar=planar)
    projected_cloud_latents, cloud_off_plane_norm = project_latents_to_pca_space(
        reference_points=reference_latents,
        points=dense_latents,
        projection_dim=projection_dimension,
    )
    projected_arrow_latents, arrow_off_plane_norm = project_latents_to_pca_space(
        reference_points=reference_latents,
        points=vector_latents,
        projection_dim=projection_dimension,
    )
    projected_transport_field = project_latent_vectors_to_pca_space(
        reference_points=reference_latents,
        vectors=transport_field,
        projection_dim=projection_dimension,
    )
    cloud_latent_norm = flatten_latents(dense_latents).norm(dim=-1)
    arrow_latent_norm = flatten_latents(vector_latents).norm(dim=-1)
    transport_field_norm = flatten_latents(transport_field).norm(dim=-1)

    figure = go.Figure()
    if planar:
        if (
            transport_grid_xs is None
            or transport_grid_ys is None
            or transport_grid_projection is None
        ):
            raise ValueError("planar critic transport plots require contour inputs")
        figure.add_trace(
            go.Contour(
                x=transport_grid_xs.to(dtype=torch.float32).tolist(),
                y=transport_grid_ys.to(dtype=torch.float32).tolist(),
                z=transport_grid_projection.to(dtype=torch.float32).numpy(),
                contours={"coloring": "lines", "showlabels": True},
                line={"width": 1.1, "color": "rgba(70, 70, 70, 0.55)"},
                ncontours=num_contour_lines,
                showscale=False,
                name="transport norm contours",
                hoverinfo="skip",
            )
        )

    add_mode_scatter_traces(
        figure=figure,
        points=projected_cloud_latents,
        mode_ids=dense_mode_ids,
        size=5,
        opacity=0.26,
        showlegend=True,
        latent_norms=cloud_latent_norm,
        off_plane_norms=cloud_off_plane_norm,
        planar=planar,
    )
    add_mode_scatter_traces(
        figure=figure,
        points=projected_arrow_latents,
        mode_ids=vector_mode_ids,
        size=6,
        opacity=0.48,
        showlegend=False,
        latent_norms=arrow_latent_norm,
        off_plane_norms=arrow_off_plane_norm,
        planar=planar,
    )
    display_vectors = normalize_vectors(
        vectors=projected_transport_field,
        display_length=vector_display_length(
            projected_cloud_latents,
            fraction=0.02,
        ),
    ) * ARROW_DISPLAY_SCALE
    add_mode_vector_traces(
        figure=figure,
        points=projected_arrow_latents,
        vectors=display_vectors,
        vector_norms=transport_field_norm,
        off_plane_norms=arrow_off_plane_norm,
        mode_ids=vector_mode_ids,
        planar=planar,
    )
    figure.update_layout(
        title=(
            "Noise-averaged clean-latent transport field "
            f"({'2D' if planar else '3D'} PCA view)"
        ),
        template="plotly_white",
        width=900,
        height=900,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    if planar:
        x_min, x_max, y_min, y_max = latent_square_limits(
            projected_cloud_latents,
            padding=0.18,
        )
        figure.update_xaxes(range=[x_min, x_max], title="pc1")
        figure.update_yaxes(
            range=[y_min, y_max],
            title="pc2",
            scaleanchor="x",
            scaleratio=1.0,
        )
    else:
        figure.update_scenes(
            xaxis_title="pc1",
            yaxis_title="pc2",
            zaxis_title="pc3",
        )
    return figure


def write_critic_monitor_artifacts(
    *,
    output_folder: Path,
    reference_latents: Float[Tensor, "batch latent_dim"],
    dense_latents: Float[Tensor, "batch latent_dim"],
    dense_mode_ids: Int[Tensor, "batch"],
    vector_latents: Float[Tensor, "batch latent_dim"],
    vector_mode_ids: Int[Tensor, "batch"],
    score_snapshots: list[CriticSnapshot],
    transport_field: Float[Tensor, "batch latent_dim"],
    planar: bool,
    num_contour_lines: int,
    transport_grid_xs: Tensor | None,
    transport_grid_ys: Tensor | None,
    transport_grid_projection: Tensor | None,
) -> None:
    score_folder = output_folder / "score_snapshots"
    for snapshot in score_snapshots:
        write_figure(
            figure=critic_score_snapshot_figure(
                reference_latents=reference_latents,
                snapshot=snapshot,
                dense_mode_ids=dense_mode_ids,
                vector_mode_ids=vector_mode_ids,
                planar=planar,
            ),
            path_stem=score_folder / f"score_snapshot_t_{snapshot.t_value:.2f}".replace(
                ".",
                "p",
            ),
        )
    write_figure(
        figure=critic_transport_figure(
            reference_latents=reference_latents,
            dense_latents=dense_latents,
            dense_mode_ids=dense_mode_ids,
            vector_latents=vector_latents,
            vector_mode_ids=vector_mode_ids,
            transport_field=transport_field,
            planar=planar,
            transport_grid_xs=transport_grid_xs,
            transport_grid_ys=transport_grid_ys,
            transport_grid_projection=transport_grid_projection,
            num_contour_lines=num_contour_lines,
        ),
        path_stem=output_folder / "transport",
    )


def transport_t_values(
    *,
    transport_config: TransportLossConfig,
    num_time_samples: int,
) -> list[float]:
    t_min, t_max = transport_config.t_range
    return torch.linspace(
        t_min,
        t_max,
        num_time_samples,
        dtype=torch.float32,
    ).tolist()


def apply_critic_monitor(
    *,
    config: "CriticMonitorConfig",
    rt,
    step: int,
    stage: str,
) -> dict[str, float]:
    del stage

    dense_samples, dense_mode_ids = sample_mode_batch(
        data_config=rt.runtime_data_config,
        device=rt.device,
        batch_size_per_mode=config.n_data_latents_per_mode,
    )
    vector_samples, vector_mode_ids = sample_mode_batch(
        data_config=rt.runtime_data_config,
        device=rt.device,
        batch_size_per_mode=config.n_vectors_per_mode,
    )

    with torch.no_grad():
        dense_clean_latents = rt.chart_transport_model.encoder(dense_samples).float()
        vector_clean_latents = rt.chart_transport_model.encoder(vector_samples).float()

        score_snapshots: list[CriticSnapshot] = []
        for t_value in config.sample_t_values:
            cloud_latents, _ = sample_critic_snapshot(
                critic=rt.chart_transport_model.critic,
                clean_latents=dense_clean_latents,
                t_value=t_value,
            )
            arrow_latents, data_score = sample_critic_snapshot(
                critic=rt.chart_transport_model.critic,
                clean_latents=vector_clean_latents,
                t_value=t_value,
            )
            score_snapshots.append(
                CriticSnapshot(
                    t_value=t_value,
                    cloud_latents=cloud_latents.detach().cpu().float(),
                    arrow_latents=arrow_latents.detach().cpu().float(),
                    data_score=data_score.detach().cpu().float(),
                )
            )

        transport_config = rt.tc.chart_transport_config.loss_config.transport_config
        transport_times = transport_t_values(
            transport_config=transport_config,
            num_time_samples=config.transport_num_time_samples,
        )
        transport_field = estimate_clean_transport_field(
            critic=rt.chart_transport_model.critic,
            prior_config=rt.tc.chart_transport_config.prior_config,
            transport_config=transport_config,
            clean_latents=vector_clean_latents,
            t_values=transport_times,
        )

        transport_grid_xs = None
        transport_grid_ys = None
        transport_grid_projection = None
        if config.planar:
            transport_grid_points, transport_grid_xs, transport_grid_ys = build_latent_grid(
                reference_points=dense_clean_latents,
                resolution=config.transport_grid_resolution,
            )
            transport_grid_field = estimate_clean_transport_field(
                critic=rt.chart_transport_model.critic,
                prior_config=rt.tc.chart_transport_config.prior_config,
                transport_config=transport_config,
                clean_latents=transport_grid_points,
                t_values=transport_times,
            )
            transport_grid_projection = project_latent_vectors_to_pca_space(
                reference_points=dense_clean_latents,
                vectors=transport_grid_field,
                projection_dim=2,
            ).norm(dim=-1).reshape(
                transport_grid_ys.shape[0],
                transport_grid_xs.shape[0],
            )

    dense_clean_latents_cpu = dense_clean_latents.detach().cpu().float()
    vector_clean_latents_cpu = vector_clean_latents.detach().cpu().float()
    write_critic_monitor_artifacts(
        output_folder=step_folder(run_folder=rt.tc.folder, step=step),
        reference_latents=dense_clean_latents_cpu,
        dense_latents=dense_clean_latents_cpu,
        dense_mode_ids=dense_mode_ids.detach().cpu().long(),
        vector_latents=vector_clean_latents_cpu,
        vector_mode_ids=vector_mode_ids.detach().cpu().long(),
        score_snapshots=score_snapshots,
        transport_field=transport_field.detach().cpu().float(),
        planar=config.planar,
        num_contour_lines=config.num_contour_lines,
        transport_grid_xs=None if transport_grid_xs is None else transport_grid_xs.detach().cpu().float(),
        transport_grid_ys=None if transport_grid_ys is None else transport_grid_ys.detach().cpu().float(),
        transport_grid_projection=(
            None
            if transport_grid_projection is None
            else transport_grid_projection.detach().cpu().float()
        ),
    )

    snapshot_score_norms = [
        flatten_latents(snapshot.data_score).norm(dim=-1).mean()
        for snapshot in score_snapshots
    ]
    transport_field_norm = flatten_latents(transport_field).norm(dim=-1)

    return {
        "critic_monitor_snapshot_score_norm_mean": torch.stack(
            snapshot_score_norms,
        ).mean().item(),
        "critic_monitor_transport_norm_mean": transport_field_norm.mean().item(),
        "critic_monitor_transport_norm_max": transport_field_norm.max().item(),
        "critic_monitor_transport_t_min": min(transport_times),
        "critic_monitor_transport_t_max": max(transport_times),
    }


__all__ = [
    "CriticSnapshot",
    "add_mode_scatter_traces",
    "add_mode_vector_traces",
    "apply_critic_monitor",
    "critic_score_snapshot_figure",
    "critic_transport_figure",
    "estimate_clean_transport_field",
    "projection_dim",
    "sample_critic_snapshot",
    "segment_coordinates",
    "transport_t_values",
    "write_critic_monitor_artifacts",
]
