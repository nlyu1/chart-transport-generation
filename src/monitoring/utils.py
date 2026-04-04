import asyncio
from pathlib import Path

import plotly.graph_objects as go
import torch
from jaxtyping import Float
from torch import Tensor

from src.utils import dict_to_cpu


def pca_project(
    data: dict[str, Float[Tensor, "batch ..."]],
    pca_dim: int,
) -> dict[str, Float[Tensor, "batch dim"]]:
    flattened_data = {
        key: value.detach().reshape(value.shape[0], -1).to(dtype=torch.float32)
        for key, value in data.items()
    }
    concatenated = torch.cat(list(flattened_data.values()), dim=0)
    _, _, factors = torch.pca_lowrank(
        concatenated,
        q=pca_dim,
        center=False,
    )
    return {
        key: torch.einsum("bi,ij->bj", value, factors)
        for key, value in flattened_data.items()
    }


def scatterplot(data: dict[str, Float[Tensor, "batch dim"]], **kwargs) -> go.Figure:
    one_key = list(data.keys())[0]
    sample = data[one_key]

    # Determine data dimension
    planar = None
    if sample.shape[-1] == 2:
        planar = True
    elif sample.shape[-1] == 3:
        planar = False
    else:
        raise RuntimeError(f"Invalid shape at {one_key}: {sample.shape}")
    data = dict_to_cpu(data)

    # Construct figure
    fig = go.Figure()
    common_kwargs = dict(mode="markers", opacity=0.5, marker=dict(size=3))
    for k, samples in data.items():
        data_kwargs = dict(
            x=samples[:, 0],
            y=samples[:, 1],
            name=k,
        )
        if planar:
            scatter_trace = go.Scatter(**(common_kwargs | data_kwargs))
        else:
            scatter_trace = go.Scatter3d(
                z=samples[:, 2], **(common_kwargs | data_kwargs)
            )
        fig.add_trace(scatter_trace)
    return fig


async def save_go(fig: go.Figure, folder: Path | str, name: str):
    folder = Path(folder)
    await asyncio.gather(
        asyncio.to_thread(fig.write_image, folder / f"{name}.png", scale=2),
        asyncio.to_thread(fig.write_html, folder / f"{name}.html"),
    )
