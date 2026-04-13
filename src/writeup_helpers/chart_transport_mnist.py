from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
from jaxtyping import Float
from torch import Tensor

from src.data.mnist.data import MNISTDataConfig
from src.monitoring.utils import pca_project, save_go_detached, scatterplot


def get_data() -> MNISTDataConfig:
    """Load the MNIST training set from ``/tmp/mnist`` (downloads if needed)."""
    return MNISTDataConfig.initialize(
        root=Path("/tmp/mnist"),
        split="train",
        download=True,
    )


def get_canonical_samples(
    data_config: MNISTDataConfig,
    *,
    samples_per_class: int,
    device: str | torch.device,
) -> dict[str, Float[Tensor, "batch 784"]]:
    """Sample a fixed batch from each digit class for consistent monitoring."""
    return {
        str(d): data_config.sample_class(
            mode_id=d, batch_size=samples_per_class
        ).to(device)
        for d in range(10)
    }


def make_latent_pca_scatter(
    latent_dict: dict[str, Float[Tensor, "batch latent_dim"]],
) -> go.Figure:
    """Project latent encodings to 3D via PCA and return an interactive scatter."""
    return scatterplot(pca_project(latent_dict, pca_dim=3))


def make_sample_grid(
    decoded_samples: Float[Tensor, "batch 784"],
    data_config: MNISTDataConfig,
    *,
    nrow: int = 8,
    ncol: int = 8,
) -> plt.Figure:
    """Arrange decoded MNIST samples as a greyscale image grid."""
    images = data_config.as_images(decoded_samples[: nrow * ncol].clamp(0, 1).cpu())
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol, nrow))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.tight_layout(pad=0.3)
    return fig


def save_monitor_artifacts(
    pca_fig: go.Figure,
    grid_fig: plt.Figure,
    artifacts_dir: Path,
    step: int | str,
) -> None:
    """Save both figures to *artifacts_dir*."""
    save_go_detached(pca_fig, folder=artifacts_dir, name=f"latent_pca_{step}")
    grid_fig.savefig(
        artifacts_dir / f"samples_{step}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(grid_fig)
