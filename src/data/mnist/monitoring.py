from __future__ import annotations

from pathlib import Path
from string import ascii_uppercase

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from src.data.mnist.data import MNISTDataConfig
from src.monitoring.critic import write_critic_monitor_artifacts
from src.monitoring.utils import (
    build_latent_grid,
    flatten_latents,
    latent_square_limits,
    project_latents_to_pca_space,
)


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


def reconstruction_column_titles(
    *,
    examples_per_class: int,
) -> list[str]:
    if examples_per_class <= 0:
        raise ValueError("examples_per_class must be positive")
    if examples_per_class > len(ascii_uppercase):
        raise ValueError(
            f"examples_per_class must be at most {len(ascii_uppercase)} to label columns"
        )
    subplot_titles = []
    for example_index in range(examples_per_class):
        sample_label = ascii_uppercase[example_index]
        subplot_titles.extend(
            [
                sample_label,
                f"{sample_label} reconstruction",
            ]
        )
    return subplot_titles


def sample_constraint_monitor_batches(
    *,
    data_config: MNISTDataConfig,
    reconstruction_examples_per_class: int,
    latent_examples_per_class: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction_samples, reconstruction_labels = data_config.stratified_class_batch(
        batch_size_per_class=reconstruction_examples_per_class,
        start_index=0,
    )
    latent_samples, latent_labels = data_config.stratified_class_batch(
        batch_size_per_class=latent_examples_per_class,
        start_index=reconstruction_examples_per_class,
    )
    return (
        reconstruction_samples,
        reconstruction_labels,
        latent_samples,
        latent_labels,
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
                name=str(digit_id),
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
        title="Class-wise reconstructions",
        subplot_titles=reconstruction_column_titles(
            examples_per_class=examples_per_class,
        )
        * data_config.num_classes,
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
    projected_latents, off_plane_norms = project_latents_to_pca_space(
        reference_points=latents,
        points=latents,
        projection_dim=2,
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


def write_integrated_monitor_artifacts(
    *,
    data_config: MNISTDataConfig,
    output_folder: Path,
    generated_samples: torch.Tensor,
    generated_grid_rows: int,
    generated_grid_cols: int,
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


__all__ = [
    "build_latent_grid",
    "flatten_latents",
    "plot_generated_grid",
    "plot_latent_scatter",
    "plot_reconstruction_grid",
    "project_latents_to_pca_space",
    "reconstruction_column_titles",
    "sample_constraint_monitor_batches",
    "write_critic_monitor_artifacts",
    "write_constraint_monitor_artifacts",
    "write_integrated_monitor_artifacts",
    "write_plot_artifacts",
]
