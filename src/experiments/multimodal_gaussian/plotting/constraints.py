def plot_sample_pairs(
    samples: Float[Tensor, "batch ..."],
    pairs: Float[Tensor, "batch ..."],
    manifold_deviation: Float[Tensor, "batch"],
    labels: Int[Tensor, "batch"],
    alpha: float = 0.4,
) -> go.Figure:
    """
    Plot them with different markers.
    Hover to see manifold deviation.
    Different labels get different colors, drawn from a color bank (consistent, 16 colors at file top-level)
    """
    pass


def plot_latents(
    latents: Float[Tensor, "batch ..."],
    labels: Int[Tensor, "batch"],
    alpha: float = 0.4,
) -> go.Figure:
    """
    1. Performs 2-axis PCA (dominant along x axis) to get projection.
    2. Plot the latents colored by label. Hover to see off-manifold norm.
    """
