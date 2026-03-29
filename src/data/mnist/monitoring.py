from __future__ import annotations

from pathlib import Path

import plotly.figure_factory as ff
import plotly.graph_objects as go
import polars as pl
import torch
from plotly.subplots import make_subplots

from src.data.mnist.data import MNISTDataConfig


DIGIT_COLORS = (
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
)


def write_plot_artifacts(
    *,
    output_folder: Path,
    figure: go.Figure,
    plot_type: str,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_folder / f"{plot_type}.html")
    figure.write_image(output_folder / f"{plot_type}.png")


def flatten_latents(
    points: torch.Tensor,
) -> torch.Tensor:
    return points.reshape(points.shape[0], -1).to(dtype=torch.float32)


def orient_projection_basis(
    basis: torch.Tensor,
) -> torch.Tensor:
    oriented_basis = basis.clone()
    for component_index in range(oriented_basis.shape[1]):
        component = oriented_basis[:, component_index]
        dominant_coordinate = int(component.abs().argmax().item())
        if component[dominant_coordinate] < 0.0:
            oriented_basis[:, component_index] = -component
    return oriented_basis


def fit_latent_pca_projection(
    *,
    reference_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_reference_points = flatten_latents(reference_points)
    projection_center = flat_reference_points.mean(dim=0, keepdim=True)
    centered_points = flat_reference_points - projection_center

    if flat_reference_points.shape[-1] == 1:
        projection_basis = torch.zeros(
            (1, 2),
            device=flat_reference_points.device,
            dtype=flat_reference_points.dtype,
        )
        projection_basis[0, 0] = 1.0
        return projection_center, projection_basis

    _, _, right_singular_vectors = torch.linalg.svd(
        centered_points,
        full_matrices=False,
    )
    projection_basis = right_singular_vectors[:2].transpose(0, 1).contiguous()
    if projection_basis.shape[1] == 1:
        projection_basis = torch.cat(
            [
                projection_basis,
                torch.zeros(
                    (projection_basis.shape[0], 1),
                    device=projection_basis.device,
                    dtype=projection_basis.dtype,
                ),
            ],
            dim=1,
        )
    return projection_center, orient_projection_basis(projection_basis)


def project_latents_to_pca_plane(
    *,
    reference_points: torch.Tensor,
    points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    projection_center, projection_basis = fit_latent_pca_projection(
        reference_points=reference_points,
    )
    flat_points = flatten_latents(points)
    centered_points = flat_points - projection_center
    projected_points = centered_points @ projection_basis
    reconstructed_points = (
        projection_center + projected_points @ projection_basis.transpose(0, 1)
    )
    off_plane_norm = (flat_points - reconstructed_points).norm(dim=-1)
    return projected_points, off_plane_norm


def project_latent_vectors_to_pca_plane(
    *,
    reference_points: torch.Tensor,
    vectors: torch.Tensor,
) -> torch.Tensor:
    _, projection_basis = fit_latent_pca_projection(reference_points=reference_points)
    return flatten_latents(vectors) @ projection_basis


def latent_square_limits(
    points: torch.Tensor,
    *,
    padding: float,
) -> tuple[float, float, float, float]:
    mins = points.min(dim=0).values
    maxs = points.max(dim=0).values
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float((maxs - mins).max().item())
    radius = max(radius * (1.0 + padding), 1.0)
    return (
        float(center[0] - radius),
        float(center[0] + radius),
        float(center[1] - radius),
        float(center[1] + radius),
    )


def build_latent_grid(
    *,
    reference_points: torch.Tensor,
    resolution: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    projected_points, _ = project_latents_to_pca_plane(
        reference_points=reference_points,
        points=reference_points,
    )
    x_min, x_max, y_min, y_max = latent_square_limits(projected_points, padding=0.18)
    xs = torch.linspace(
        x_min,
        x_max,
        resolution,
        device=projected_points.device,
        dtype=projected_points.dtype,
    )
    ys = torch.linspace(
        y_min,
        y_max,
        resolution,
        device=projected_points.device,
        dtype=projected_points.dtype,
    )
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    projected_grid_points = torch.stack(
        [grid_x.reshape(-1), grid_y.reshape(-1)],
        dim=-1,
    )
    projection_center, projection_basis = fit_latent_pca_projection(
        reference_points=reference_points,
    )
    grid_points = (
        projection_center + projected_grid_points @ projection_basis.transpose(0, 1)
    )
    return grid_points.to(device=reference_points.device), xs, ys


def vector_display_length(
    points: torch.Tensor,
    *,
    fraction: float,
) -> float:
    x_min, x_max, y_min, y_max = latent_square_limits(points, padding=0.0)
    return fraction * max(x_max - x_min, y_max - y_min)


def normalize_vectors(
    *,
    vectors: torch.Tensor,
    display_length: float,
) -> torch.Tensor:
    magnitudes = vectors.norm(dim=-1, keepdim=True)
    return display_length * vectors / magnitudes.clamp_min(1e-6)


def add_digit_scatter_traces(
    *,
    figure: go.Figure,
    points: torch.Tensor,
    labels: torch.Tensor,
    size: float,
    opacity: float,
    showlegend: bool,
    latent_norms: torch.Tensor,
    off_plane_norms: torch.Tensor,
) -> None:
    points_float = points.to(dtype=torch.float32)
    labels_long = labels.to(dtype=torch.long)
    latent_norms_float = latent_norms.to(dtype=torch.float32)
    off_plane_norms_float = off_plane_norms.to(dtype=torch.float32)
    for digit_id in range(10):
        mask = labels_long == digit_id
        if not mask.any():
            continue
        customdata = torch.stack(
            [
                labels_long[mask].to(dtype=torch.float32),
                latent_norms_float[mask],
                off_plane_norms_float[mask],
            ],
            dim=-1,
        ).tolist()
        figure.add_trace(
            go.Scatter(
                x=points_float[mask, 0].tolist(),
                y=points_float[mask, 1].tolist(),
                mode="markers",
                name=f"digit {digit_id}",
                showlegend=showlegend,
                marker={
                    "size": size,
                    "opacity": opacity,
                    "color": DIGIT_COLORS[digit_id],
                    "line": {"width": 0.0},
                },
                customdata=customdata,
                hovertemplate=(
                    "digit=%{customdata[0]:.0f}<br>"
                    "pc1=%{x:.3f}<br>"
                    "pc2=%{y:.3f}<br>"
                    "|latent|=%{customdata[1]:.4f}<br>"
                    "off_pca_norm=%{customdata[2]:.4f}<extra></extra>"
                ),
            )
        )


def add_digit_quiver_traces(
    *,
    figure: go.Figure,
    points: torch.Tensor,
    vectors: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    points_float = points.to(dtype=torch.float32)
    vectors_float = vectors.to(dtype=torch.float32)
    labels_long = labels.to(dtype=torch.long)
    for digit_id in range(10):
        mask = labels_long == digit_id
        if not mask.any():
            continue
        quiver = ff.create_quiver(
            x=points_float[mask, 0].tolist(),
            y=points_float[mask, 1].tolist(),
            u=vectors_float[mask, 0].tolist(),
            v=vectors_float[mask, 1].tolist(),
            scale=1.0,
            arrow_scale=0.25,
            line_color=DIGIT_COLORS[digit_id],
            name=f"digit {digit_id}",
        )
        for trace in quiver.data:
            trace.showlegend = False
            trace.hoverinfo = "skip"
            trace.line.width = 1.2
            figure.add_trace(trace)


def add_digit_quiver_hover_traces(
    *,
    figure: go.Figure,
    points: torch.Tensor,
    display_vectors: torch.Tensor,
    vector_norms: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    points_float = points.to(dtype=torch.float32)
    display_vectors_float = display_vectors.to(dtype=torch.float32)
    vector_norms_float = vector_norms.to(dtype=torch.float32)
    labels_long = labels.to(dtype=torch.long)
    hover_points = points_float + 0.5 * display_vectors_float
    for digit_id in range(10):
        mask = labels_long == digit_id
        if not mask.any():
            continue
        figure.add_trace(
            go.Scatter(
                x=hover_points[mask, 0].tolist(),
                y=hover_points[mask, 1].tolist(),
                mode="markers",
                marker={
                    "size": 10,
                    "color": DIGIT_COLORS[digit_id],
                    "opacity": 0.18,
                    "line": {"width": 0.0},
                },
                customdata=vector_norms_float[mask].unsqueeze(-1).tolist(),
                hovertemplate="arrow_norm=%{customdata[0]:.4f}<extra></extra>",
                showlegend=False,
                name=f"digit {digit_id} hover",
            )
        )


def image_grid_figure(
    *,
    images: torch.Tensor,
    rows: int,
    cols: int,
    title: str,
    subplot_titles: list[str],
    row_titles: list[str],
) -> go.Figure:
    if images.shape[0] != rows * cols:
        raise ValueError("images.shape[0] must equal rows * cols")
    figure = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        row_titles=row_titles,
        horizontal_spacing=0.01,
        vertical_spacing=0.02,
    )
    image_index = 0
    for row in range(rows):
        for col in range(cols):
            figure.add_trace(
                go.Heatmap(
                    z=images[image_index].flip(0).to(dtype=torch.float32).numpy(),
                    zmin=0.0,
                    zmax=1.0,
                    colorscale="Gray",
                    showscale=False,
                    hoverinfo="skip",
                ),
                row=row + 1,
                col=col + 1,
            )
            figure.update_xaxes(
                visible=False,
                row=row + 1,
                col=col + 1,
            )
            figure.update_yaxes(
                visible=False,
                row=row + 1,
                col=col + 1,
                scaleanchor=f"x{image_index + 1}",
            )
            image_index += 1
    figure.update_layout(
        title=title,
        template="plotly_white",
        width=220 * cols,
        height=180 * rows,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return figure


def plot_reconstruction_grid(
    *,
    data_config: MNISTDataConfig,
    samples: torch.Tensor,
    reconstructions: torch.Tensor,
    labels: torch.Tensor,
    examples_per_class: int,
) -> go.Figure:
    sample_images = data_config.as_images(samples.clamp(0.0, 1.0))
    reconstruction_images = data_config.as_images(reconstructions.clamp(0.0, 1.0))
    grid_images = []
    row_titles = []
    labels_long = labels.to(dtype=torch.long)
    for digit_id in range(data_config.num_classes):
        mask = labels_long == digit_id
        if int(mask.sum().item()) != examples_per_class:
            raise ValueError("labels must contain examples_per_class items for each digit")
        row_titles.append(f"digit {digit_id}")
        originals = sample_images[mask]
        reconstructions_for_digit = reconstruction_images[mask]
        for column_index in range(examples_per_class):
            grid_images.append(originals[column_index])
            grid_images.append(reconstructions_for_digit[column_index])
    return image_grid_figure(
        images=torch.stack(grid_images, dim=0),
        rows=data_config.num_classes,
        cols=2 * examples_per_class,
        title="Validation reconstructions",
        subplot_titles=[
            title
            for _ in range(data_config.num_classes)
            for title in (["data", "recon"] * examples_per_class)
        ],
        row_titles=row_titles,
    )


def plot_generated_grid(
    *,
    data_config: MNISTDataConfig,
    generated_samples: torch.Tensor,
    rows: int,
    cols: int,
) -> go.Figure:
    generated_images = data_config.as_images(generated_samples.clamp(0.0, 1.0))
    return image_grid_figure(
        images=generated_images,
        rows=rows,
        cols=cols,
        title="Fixed prior samples",
        subplot_titles=["" for _ in range(rows * cols)],
        row_titles=["" for _ in range(rows)],
    )


def plot_latent_scatter(
    *,
    latents: torch.Tensor,
    labels: torch.Tensor,
    title: str,
) -> go.Figure:
    projected_latents, off_plane_norms = project_latents_to_pca_plane(
        reference_points=latents,
        points=latents,
    )
    latent_norms = flatten_latents(latents).norm(dim=-1)
    figure = go.Figure()
    add_digit_scatter_traces(
        figure=figure,
        points=projected_latents,
        labels=labels,
        size=5,
        opacity=0.45,
        showlegend=True,
        latent_norms=latent_norms,
        off_plane_norms=off_plane_norms,
    )
    x_min, x_max, y_min, y_max = latent_square_limits(projected_latents, padding=0.18)
    figure.update_layout(
        title=title,
        template="plotly_white",
        width=900,
        height=900,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        xaxis={"range": [x_min, x_max], "title": "pc1"},
        yaxis={"range": [y_min, y_max], "title": "pc2", "scaleanchor": "x", "scaleratio": 1.0},
    )
    return figure


def plot_critic_score_snapshot(
    *,
    reference_latents: torch.Tensor,
    cloud_latents: torch.Tensor,
    cloud_labels: torch.Tensor,
    arrow_latents: torch.Tensor,
    arrow_labels: torch.Tensor,
    data_score: torch.Tensor,
    t_value: float,
) -> go.Figure:
    projected_cloud_latents, cloud_off_plane_norm = project_latents_to_pca_plane(
        reference_points=reference_latents,
        points=cloud_latents,
    )
    projected_arrow_latents, arrow_off_plane_norm = project_latents_to_pca_plane(
        reference_points=reference_latents,
        points=arrow_latents,
    )
    projected_data_score = project_latent_vectors_to_pca_plane(
        reference_points=reference_latents,
        vectors=data_score,
    )
    cloud_latent_norm = flatten_latents(cloud_latents).norm(dim=-1)
    arrow_latent_norm = flatten_latents(arrow_latents).norm(dim=-1)

    figure = go.Figure()
    add_digit_scatter_traces(
        figure=figure,
        points=projected_cloud_latents,
        labels=cloud_labels,
        size=5,
        opacity=0.26,
        showlegend=True,
        latent_norms=cloud_latent_norm,
        off_plane_norms=cloud_off_plane_norm,
    )
    add_digit_scatter_traces(
        figure=figure,
        points=projected_arrow_latents,
        labels=arrow_labels,
        size=6,
        opacity=0.48,
        showlegend=False,
        latent_norms=arrow_latent_norm,
        off_plane_norms=arrow_off_plane_norm,
    )
    add_digit_quiver_traces(
        figure=figure,
        points=projected_arrow_latents,
        vectors=normalize_vectors(
            vectors=projected_data_score,
            display_length=vector_display_length(projected_cloud_latents, fraction=0.02),
        ),
        labels=arrow_labels,
    )
    x_min, x_max, y_min, y_max = latent_square_limits(projected_cloud_latents, padding=0.18)
    figure.update_layout(
        title=f"Critic score snapshot at t={t_value:.2f} (PCA view)",
        template="plotly_white",
        width=900,
        height=900,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        xaxis={"range": [x_min, x_max], "title": "pc1"},
        yaxis={"range": [y_min, y_max], "title": "pc2", "scaleanchor": "x", "scaleratio": 1.0},
    )
    return figure


def plot_transport_field(
    *,
    reference_latents: torch.Tensor,
    cloud_latents: torch.Tensor,
    cloud_labels: torch.Tensor,
    arrow_latents: torch.Tensor,
    arrow_labels: torch.Tensor,
    transport_field: torch.Tensor,
    grid_xs: torch.Tensor,
    grid_ys: torch.Tensor,
    grid_transport_projection: torch.Tensor,
    num_contour_lines: int,
) -> go.Figure:
    projected_cloud_latents, cloud_off_plane_norm = project_latents_to_pca_plane(
        reference_points=reference_latents,
        points=cloud_latents,
    )
    projected_arrow_latents, arrow_off_plane_norm = project_latents_to_pca_plane(
        reference_points=reference_latents,
        points=arrow_latents,
    )
    projected_transport_field = project_latent_vectors_to_pca_plane(
        reference_points=reference_latents,
        vectors=transport_field,
    )
    cloud_latent_norm = flatten_latents(cloud_latents).norm(dim=-1)
    arrow_latent_norm = flatten_latents(arrow_latents).norm(dim=-1)
    transport_field_norm = flatten_latents(transport_field).norm(dim=-1)

    figure = go.Figure()
    figure.add_trace(
        go.Contour(
            x=grid_xs.to(dtype=torch.float32).tolist(),
            y=grid_ys.to(dtype=torch.float32).tolist(),
            z=grid_transport_projection.to(dtype=torch.float32).numpy(),
            contours={"coloring": "lines", "showlabels": True},
            line={"width": 1.1, "color": "rgba(70, 70, 70, 0.55)"},
            ncontours=num_contour_lines,
            showscale=False,
            name="transport norm contours",
            hoverinfo="skip",
        )
    )
    add_digit_scatter_traces(
        figure=figure,
        points=projected_cloud_latents,
        labels=cloud_labels,
        size=5,
        opacity=0.26,
        showlegend=True,
        latent_norms=cloud_latent_norm,
        off_plane_norms=cloud_off_plane_norm,
    )
    add_digit_scatter_traces(
        figure=figure,
        points=projected_arrow_latents,
        labels=arrow_labels,
        size=6,
        opacity=0.48,
        showlegend=False,
        latent_norms=arrow_latent_norm,
        off_plane_norms=arrow_off_plane_norm,
    )
    display_vectors = normalize_vectors(
        vectors=projected_transport_field,
        display_length=vector_display_length(projected_cloud_latents, fraction=0.02),
    )
    add_digit_quiver_traces(
        figure=figure,
        points=projected_arrow_latents,
        vectors=display_vectors,
        labels=arrow_labels,
    )
    add_digit_quiver_hover_traces(
        figure=figure,
        points=projected_arrow_latents,
        display_vectors=display_vectors,
        vector_norms=transport_field_norm,
        labels=arrow_labels,
    )
    x_min, x_max, y_min, y_max = latent_square_limits(projected_cloud_latents, padding=0.18)
    figure.update_layout(
        title="Noise-averaged clean-latent transport field (PCA view)",
        template="plotly_white",
        width=900,
        height=900,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        xaxis={"range": [x_min, x_max], "title": "pc1"},
        yaxis={"range": [y_min, y_max], "title": "pc2", "scaleanchor": "x", "scaleratio": 1.0},
    )
    return figure


def plot_critic_loss_spectrum(
    *,
    loss_spectrum: pl.DataFrame,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=loss_spectrum["t"].to_list(),
            y=loss_spectrum["noise_prediction_mse"].to_list(),
            mode="lines+markers",
            line={"color": "#1f77b4", "width": 2.0},
            marker={"size": 7},
            name="noise prediction mse",
        )
    )
    figure.update_layout(
        title="Critic loss spectrum",
        template="plotly_white",
        width=1000,
        height=500,
        xaxis={"title": "t"},
        yaxis={"title": "noise prediction mse"},
    )
    return figure


def history_plot_values(
    *,
    history: pl.DataFrame,
    column: str,
) -> list[float | None]:
    values = []
    for value in history[column].to_list():
        if isinstance(value, float) and value != value:
            values.append(None)
        else:
            values.append(float(value))
    return values


def plot_constraint_history(
    *,
    history: pl.DataFrame,
) -> go.Figure:
    steps = history["step"].to_list()
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=history_plot_values(history=history, column="data_cycle_loss"),
            mode="lines",
            line={"color": "#1f77b4", "width": 2.0},
            name="data cycle loss",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=history_plot_values(history=history, column="prior_cycle_loss"),
            mode="lines",
            line={"color": "#ff7f0e", "width": 2.0},
            name="prior cycle loss",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=history_plot_values(history=history, column="data_dual"),
            mode="lines",
            line={"color": "#2ca02c", "width": 2.0},
            name="data dual",
        ),
        secondary_y=True,
    )
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=history_plot_values(history=history, column="prior_dual"),
            mode="lines",
            line={"color": "#d62728", "width": 2.0},
            name="prior dual",
        ),
        secondary_y=True,
    )
    figure.update_layout(
        title="Constraint losses and dual variables",
        template="plotly_white",
        width=1100,
        height=500,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    figure.update_xaxes(title_text="step")
    figure.update_yaxes(title_text="loss", secondary_y=False)
    figure.update_yaxes(title_text="dual", secondary_y=True)
    return figure


def plot_transport_history(
    *,
    history: pl.DataFrame,
) -> go.Figure:
    steps = history["step"].to_list()
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=history_plot_values(history=history, column="critic_loss"),
            mode="lines",
            line={"color": "#1f77b4", "width": 2.0},
            name="critic loss",
            connectgaps=True,
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=history_plot_values(history=history, column="transport_field_norm"),
            mode="lines+markers",
            line={"color": "#ff7f0e", "width": 2.5, "dash": "dash"},
            marker={"size": 4, "color": "#ff7f0e"},
            name="transport field norm",
            connectgaps=True,
        ),
        secondary_y=True,
    )
    figure.update_layout(
        title="Transport field norm and critic loss",
        template="plotly_white",
        width=1100,
        height=500,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    figure.update_xaxes(title_text="step")
    figure.update_yaxes(title_text="critic loss", secondary_y=False)
    figure.update_yaxes(title_text="transport field norm", secondary_y=True)
    return figure


def plot_generation_statistics_history(
    *,
    history: pl.DataFrame,
) -> go.Figure:
    steps = history["step"].to_list()
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=history_plot_values(history=history, column="generated_pixel_mean"),
            mode="lines",
            line={"color": "#2ca02c", "width": 2.0},
            name="generated pixel mean",
            connectgaps=True,
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=history_plot_values(history=history, column="generated_pixel_std"),
            mode="lines",
            line={"color": "#9467bd", "width": 2.0},
            name="generated pixel std",
            connectgaps=True,
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=history_plot_values(history=history, column="generated_out_of_range_fraction"),
            mode="lines+markers",
            line={"color": "#d62728", "width": 2.5, "dash": "dash"},
            marker={"size": 4, "color": "#d62728"},
            name="generated out-of-range fraction",
            connectgaps=True,
        ),
        secondary_y=True,
    )
    figure.update_layout(
        title="Generated sample statistics",
        template="plotly_white",
        width=1100,
        height=500,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    figure.update_xaxes(title_text="step")
    figure.update_yaxes(title_text="pixel statistics", secondary_y=False)
    figure.update_yaxes(title_text="fraction", secondary_y=True)
    return figure


def write_constraint_monitor_artifacts(
    *,
    data_config: MNISTDataConfig,
    output_folder: Path,
    reconstruction_samples: torch.Tensor,
    reconstruction_labels: torch.Tensor,
    reconstructions: torch.Tensor,
    latent_values: torch.Tensor,
    latent_labels: torch.Tensor,
    examples_per_class: int,
) -> None:
    reconstruction_figure = plot_reconstruction_grid(
        data_config=data_config,
        samples=reconstruction_samples,
        reconstructions=reconstructions,
        labels=reconstruction_labels,
        examples_per_class=examples_per_class,
    )
    write_plot_artifacts(
        output_folder=output_folder,
        figure=reconstruction_figure,
        plot_type="reconstructions",
    )

    latent_figure = plot_latent_scatter(
        latents=latent_values,
        labels=latent_labels,
        title="Validation latents (PCA view)",
    )
    write_plot_artifacts(
        output_folder=output_folder,
        figure=latent_figure,
        plot_type="latents",
    )


def write_critic_monitor_artifacts(
    *,
    output_folder: Path,
    dense_clean_latents: torch.Tensor,
    dense_labels: torch.Tensor,
    vector_clean_latents: torch.Tensor,
    vector_labels: torch.Tensor,
    score_snapshots: list[tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]],
    transport_field: torch.Tensor,
    transport_grid_xs: torch.Tensor,
    transport_grid_ys: torch.Tensor,
    transport_grid_projection: torch.Tensor,
    loss_spectrum: pl.DataFrame,
    num_contour_lines: int,
) -> None:
    score_folder = output_folder / "score_snapshots"
    for t_value, cloud_latents, arrow_latents, data_score in score_snapshots:
        score_figure = plot_critic_score_snapshot(
            reference_latents=dense_clean_latents,
            cloud_latents=cloud_latents,
            cloud_labels=dense_labels,
            arrow_latents=arrow_latents,
            arrow_labels=vector_labels,
            data_score=data_score,
            t_value=t_value,
        )
        write_plot_artifacts(
            output_folder=score_folder,
            figure=score_figure,
            plot_type=f"score_snapshot_t_{t_value:.2f}".replace(".", "p"),
        )

    transport_figure = plot_transport_field(
        reference_latents=dense_clean_latents,
        cloud_latents=dense_clean_latents,
        cloud_labels=dense_labels,
        arrow_latents=vector_clean_latents,
        arrow_labels=vector_labels,
        transport_field=transport_field,
        grid_xs=transport_grid_xs,
        grid_ys=transport_grid_ys,
        grid_transport_projection=transport_grid_projection,
        num_contour_lines=num_contour_lines,
    )
    write_plot_artifacts(
        output_folder=output_folder,
        figure=transport_figure,
        plot_type="transport",
    )

    loss_spectrum_figure = plot_critic_loss_spectrum(
        loss_spectrum=loss_spectrum,
    )
    write_plot_artifacts(
        output_folder=output_folder,
        figure=loss_spectrum_figure,
        plot_type="loss_spectrum",
    )


def write_integrated_monitor_artifacts(
    *,
    data_config: MNISTDataConfig,
    output_folder: Path,
    generated_samples: torch.Tensor,
    generated_grid_rows: int,
    generated_grid_cols: int,
    recent_history: pl.DataFrame,
) -> None:
    generated_figure = plot_generated_grid(
        data_config=data_config,
        generated_samples=generated_samples,
        rows=generated_grid_rows,
        cols=generated_grid_cols,
    )
    write_plot_artifacts(
        output_folder=output_folder,
        figure=generated_figure,
        plot_type="generated_samples",
    )

    constraint_history_figure = plot_constraint_history(history=recent_history)
    write_plot_artifacts(
        output_folder=output_folder,
        figure=constraint_history_figure,
        plot_type="constraint_history",
    )

    transport_history_figure = plot_transport_history(history=recent_history)
    write_plot_artifacts(
        output_folder=output_folder,
        figure=transport_history_figure,
        plot_type="transport_history",
    )

    generation_stats_figure = plot_generation_statistics_history(
        history=recent_history,
    )
    write_plot_artifacts(
        output_folder=output_folder,
        figure=generation_stats_figure,
        plot_type="generation_statistics",
    )


__all__ = [
    "build_latent_grid",
    "flatten_latents",
    "history_plot_values",
    "plot_constraint_history",
    "plot_critic_loss_spectrum",
    "plot_critic_score_snapshot",
    "plot_generated_grid",
    "plot_generation_statistics_history",
    "plot_latent_scatter",
    "plot_reconstruction_grid",
    "plot_transport_field",
    "plot_transport_history",
    "project_latents_to_pca_plane",
    "project_latent_vectors_to_pca_plane",
    "write_constraint_monitor_artifacts",
    "write_critic_monitor_artifacts",
    "write_integrated_monitor_artifacts",
    "write_plot_artifacts",
]
