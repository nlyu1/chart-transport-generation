"""Image-grid visualization helpers for the cifar-smoke VAE experiment.

The intended call pattern from a synchronous training loop is

    fut = save_image_grid_in_background(images=samples, path=...)

which snapshots the GPU tensor to CPU on the calling thread (cheap relative to a
PNG write) and dispatches the actual rendering + file write to a worker thread,
so the next training iteration can start immediately. ``save_image_grid_async``
exposes the same work as an awaitable for callers that already live inside an
event loop.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from pathlib import Path

import numpy as np
import plotly.express as px
import torch
from airbench.utils import CIFAR_MEAN, CIFAR_STD
from jaxtyping import Float
from torch import Tensor

_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="vae-grid-saver"
)


def _denormalize_cifar(
    images: Float[Tensor, "n 3 h w"],
) -> Float[Tensor, "n 3 h w"]:
    mean = CIFAR_MEAN.to(images.device, images.dtype).view(1, 3, 1, 1)
    std = CIFAR_STD.to(images.device, images.dtype).view(1, 3, 1, 1)
    return images * std + mean


def _tile_grid(
    images: Float[Tensor, "n 3 h w"], *, ncols: int
) -> Float[np.ndarray, "H W 3"]:
    n, c, h, w = images.shape
    nrows = (n + ncols - 1) // ncols
    pad = nrows * ncols - n
    if pad:
        filler = torch.zeros(pad, c, h, w, device=images.device, dtype=images.dtype)
        images = torch.cat([images, filler], dim=0)
    grid = (
        images.view(nrows, ncols, c, h, w)
        .permute(0, 3, 1, 4, 2)
        .reshape(nrows * h, ncols * w, c)
    )
    return grid.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()


def _write_grid_png(
    *,
    images_cpu: Float[Tensor, "n 3 h w"],
    path: Path,
    ncols: int,
    title: str | None,
    pixel_scale: int,
) -> None:
    """Blocking core: denormalize + tile + write PNG via plotly. Expects a CPU tensor."""
    images_cpu = _denormalize_cifar(images_cpu.float())
    array = _tile_grid(images_cpu, ncols=ncols)
    h, w, _ = array.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = px.imshow(array)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    title_height = 32 if title else 0
    fig.update_layout(
        margin=dict(l=0, r=0, t=title_height, b=0),
        title=dict(text=title, x=0.5) if title else None,
        coloraxis_showscale=False,
    )
    fig.write_image(
        str(path),
        width=w * pixel_scale,
        height=h * pixel_scale + title_height,
    )


def save_image_grid(
    *,
    images: Float[Tensor, "n 3 h w"],
    path: str | Path,
    ncols: int = 8,
    title: str | None = None,
    pixel_scale: int = 4,
) -> None:
    """Synchronous, blocking saver. Accepts a GPU or CPU tensor (CIFAR-normalized)."""
    cpu = images.detach().to("cpu").contiguous()
    _write_grid_png(
        images_cpu=cpu,
        path=Path(path),
        ncols=ncols,
        title=title,
        pixel_scale=pixel_scale,
    )


async def save_image_grid_async(
    *,
    images: Float[Tensor, "n 3 h w"],
    path: str | Path,
    ncols: int = 8,
    title: str | None = None,
    pixel_scale: int = 4,
) -> None:
    """Async wrapper: snapshots to CPU, awaits the blocking writer on a worker thread."""
    cpu = images.detach().to("cpu").contiguous()
    await asyncio.to_thread(
        _write_grid_png,
        images_cpu=cpu,
        path=Path(path),
        ncols=ncols,
        title=title,
        pixel_scale=pixel_scale,
    )


def save_image_grid_in_background(
    *,
    images: Float[Tensor, "n 3 h w"],
    path: str | Path,
    ncols: int = 8,
    title: str | None = None,
    pixel_scale: int = 4,
) -> concurrent.futures.Future:
    """Fire-and-forget wrapper for synchronous loops.

    Snapshots ``images`` to CPU on the calling thread (forces a cuda sync, but
    the copy itself is small) and dispatches the PNG write to a background
    thread. Returns a ``Future`` so the caller can ``.result()`` at shutdown to
    surface any rendering errors.
    """
    cpu = images.detach().to("cpu").contiguous()
    return _EXECUTOR.submit(
        _write_grid_png,
        images_cpu=cpu,
        path=Path(path),
        ncols=ncols,
        title=title,
        pixel_scale=pixel_scale,
    )
