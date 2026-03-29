from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go


COLOR_BANK = (
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
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
)


def marker_color(
    *,
    group_id: int,
) -> str:
    return COLOR_BANK[group_id % len(COLOR_BANK)]


def step_folder(
    *,
    run_folder: Path,
    step: int,
) -> Path:
    folder = run_folder / str(step)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def write_figure(
    *,
    figure: go.Figure,
    path_stem: Path,
) -> None:
    figure.write_html(path_stem.with_suffix(".html"))
    figure.write_image(path_stem.with_suffix(".png"))


__all__ = ["COLOR_BANK", "marker_color", "step_folder", "write_figure"]
