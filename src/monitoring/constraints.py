from src.monitoring.utils import COLOR_BANK

# type-check preamble


def save_latent_plot_to(
    config: ConstraintMonitorConfig,
    latents: Float[Tensor, "batch *data"],
    mode_ids: Int[Tensor, "batch"],
    save_to_folder: Path,
) -> None:
    """
    Creates a single 2D, or 3D (depending on `planar`) scatterplot
    of the (2 or 3)-PCA latents. Creates two artifacts:

    1. latent_plot.html
    2. latent_plot.png

    Specifications:
    1. Different mode_ids should have different colors
    2. Upon hovering over a cell, should show its off-plane norm.
        More off-plane points should get smaller dot (normalized).
        On-plane scatters should have standard size

    No need for any axis labeling. Keep the plot otherwise as clean as possible.
    """
    alpha = 0.4
    COLOR_BANK
