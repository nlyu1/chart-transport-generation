import asyncio
import threading
from pathlib import Path

import plotly.graph_objects as go
import torch
from jaxtyping import Float
from torch import Tensor

from src.utils import dict_to_cpu


def pca_project(
    data: dict[str, Float[Tensor, "batch ..."]],
    pca_dim: int,
    return_factors: bool = False,
) -> (
    dict[str, Float[Tensor, "batch dim"]]
    | tuple[dict[str, Float[Tensor, "batch dim"]], Float[Tensor, "dim pca_dim"]]
):
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
    result = {
        key: torch.einsum("bi,ij->bj", value, factors)
        for key, value in flattened_data.items()
    }
    if return_factors:
        return result, factors
    else:
        return result


def scatterplot(
    data: dict[str, Float[Tensor, "batch dim"]], title: str | None = None
) -> go.Figure:
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


def fieldplot(
    *,
    base: dict[str, Float[Tensor, "batch 3"]],
    field: dict[str, Float[Tensor, "batch 3"]],
    title: str | None = None,
) -> go.Figure:
    fig = go.Figure()
    for key in base.keys():
        base_points = base[key]
        vectors = field[key]
        trace = go.Cone(
            x=base_points[:, 0].cpu(),
            y=base_points[:, 1].cpu(),
            z=base_points[:, 2].cpu(),
            u=vectors[:, 0].cpu(),
            v=vectors[:, 1].cpu(),
            w=vectors[:, 2].cpu(),
            name=key,
            colorscale="Blues",
            anchor="tail",
            showscale=False,
            showlegend=True,
        )
        fig.add_trace(trace)
    return fig


def save_go(fig: go.Figure, folder: Path | str, name: str):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.write_image(folder / f"{name}.png", scale=2)
    fig.write_html(folder / f"{name}.html")


async def save_go_async(fig: go.Figure, folder: Path | str, name: str):
    await asyncio.to_thread(save_go, fig, folder, name)


def save_go_detached(fig: go.Figure, folder: Path | str, name: str):
    thread = threading.Thread(
        target=save_go,
        args=(fig, folder, name),
        daemon=False,
    )
    thread.start()
    return thread
