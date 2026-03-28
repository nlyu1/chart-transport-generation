from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
from jaxtyping import Float
import plotly.figure_factory as ff
import plotly.graph_objects as go
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from plotly.subplots import make_subplots
from torch import Tensor


@dataclass(
    kw_only=True,
    config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"),
)
class DataConfig:
    batch_size: int
    mode_center: tuple[float, float]
    std_cap: float
    offset: tuple[float, float]


@dataclass(
    kw_only=True,
    config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"),
)
class LossConfig:
    t_min: float
    reconstruction_weight: float
    prior_matching_weight: float
    cycle_data_weight: float
    cycle_prior_weight: float
    denoising_weight: float
    score_weight: float


@dataclass(
    kw_only=True,
    config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"),
)
class TrainConfig:
    steps: int
    lr: float
    weight_decay: float
    grad_clip: float
    decoder_attenuation: float
    log_every_steps: int


@dataclass(
    kw_only=True,
    config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"),
)
class PlotConfig:
    eval_size: int
    latent_plot_size: int
    contour_levels: tuple[float, float, float, float, float, float]
    contour_points_per_level: int
    snapshot_every_steps: int
    arrow_stride: int


@dataclass(
    kw_only=True,
    config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"),
)
class ExperimentConfig:
    data: DataConfig
    loss: LossConfig
    train: TrainConfig
    plot: PlotConfig


@dataclass(
    kw_only=True,
    config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"),
)
class LatentVectorFieldSnapshot:
    noise_level: float
    points: Float[Tensor, "batch 2"]
    mode_ids: Tensor
    arrow_points: Float[Tensor, "arrow_batch 2"]
    arrow_vectors: Float[Tensor, "arrow_batch 2"]
    arrow_mode_ids: Tensor
    arrow_magnitudes: Float[Tensor, "arrow_batch"]


def mode_centers_tensor(
    *,
    data_config: DataConfig,
    device: str,
    dtype: torch.dtype,
) -> Float[Tensor, "2 2"]:
    mode_center = torch.tensor(data_config.mode_center, device=device, dtype=dtype)
    offset = torch.tensor(data_config.offset, device=device, dtype=dtype)
    return torch.stack([offset - mode_center, offset + mode_center], dim=0)


def sample_bimodal_gaussian(
    *,
    batch_size: int,
    data_config: DataConfig,
    device: str,
    dtype: torch.dtype,
) -> tuple[Float[Tensor, "batch 2"], Tensor]:
    centers = mode_centers_tensor(
        data_config=data_config,
        device=device,
        dtype=dtype,
    )
    mode_ids = torch.randint(0, 2, (batch_size,), device=device)
    x = centers[mode_ids] + data_config.std_cap * torch.randn(
        batch_size,
        2,
        device=device,
        dtype=dtype,
    )
    return x, mode_ids


def build_gaussian_contour_points(
    *,
    contour_levels: Sequence[float],
    points_per_contour: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    contour_points: list[Tensor] = []
    contour_ids: list[Tensor] = []
    for contour_index, radius in enumerate(contour_levels):
        contour_count = max(8, int(round(points_per_contour * max(float(radius), 0.25))))
        angles = torch.linspace(
            0.0,
            2.0 * math.pi,
            contour_count + 1,
            device=device,
            dtype=dtype,
        )[:-1]
        unit_circle = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        contour_points.append(float(radius) * unit_circle)
        contour_ids.append(
            torch.full(
                (contour_count,),
                fill_value=contour_index,
                device=device,
                dtype=torch.long,
            )
        )
    return torch.cat(contour_points, dim=0), torch.cat(contour_ids, dim=0)


class TimeConditionedEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layer = nn.Linear(2, 256)
        self.time_layer = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )
        self.hidden = nn.Sequential(
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 2),
        )

    def forward(
        self,
        x: Float[Tensor, "batch 2"],
        t: Float[Tensor, "batch 1"],
    ) -> Float[Tensor, "batch 2"]:
        hidden = self.input_layer(x) + self.time_layer(t)
        return self.hidden(hidden)


class TimeConditionedNoiseCritic(TimeConditionedEncoder):
    pass


class DecoderMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 2),
        )

    def forward(
        self,
        y: Float[Tensor, "batch 2"],
    ) -> Float[Tensor, "batch 2"]:
        return self.network(y)


class BimodalRoundtripModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = TimeConditionedEncoder()
        self.decoder = DecoderMLP()

    @staticmethod
    def detached_state(module: nn.Module) -> dict[str, Tensor]:
        return {
            **{name: parameter.detach() for name, parameter in module.named_parameters()},
            **{name: buffer.detach() for name, buffer in module.named_buffers()},
        }

    @staticmethod
    def zero_time(
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Float[Tensor, "batch 1"]:
        return torch.zeros(batch_size, 1, device=device, dtype=dtype)

    def encode(
        self,
        *,
        x: Float[Tensor, "batch 2"],
        t: Float[Tensor, "batch 1"],
        frozen: bool = False,
    ) -> Float[Tensor, "batch 2"]:
        if frozen:
            return torch.func.functional_call(
                self.encoder,
                self.detached_state(self.encoder),
                args=(x, t),
            )
        return self.encoder(x, t)

    def encode_zero(
        self,
        *,
        x: Float[Tensor, "batch 2"],
        frozen: bool = False,
    ) -> Float[Tensor, "batch 2"]:
        t = self.zero_time(
            batch_size=int(x.shape[0]),
            device=x.device,
            dtype=x.dtype,
        )
        return self.encode(x=x, t=t, frozen=frozen)

    def decode(
        self,
        *,
        y: Float[Tensor, "batch 2"],
        frozen: bool = False,
    ) -> Float[Tensor, "batch 2"]:
        if frozen:
            return torch.func.functional_call(
                self.decoder,
                self.detached_state(self.decoder),
                args=(y,),
            )
        return self.decoder(y)

    def decode_with_decoder_attenuation(
        self,
        *,
        y: Float[Tensor, "batch 2"],
        attenuation: float,
    ) -> Float[Tensor, "batch 2"]:
        decoded = self.decode(y=y)
        if attenuation == 1.0:
            return decoded
        decoded_frozen = self.decode(y=y, frozen=True)
        return decoded_frozen + (decoded - decoded_frozen) / attenuation

    def roundtrip(
        self,
        *,
        y: Float[Tensor, "batch 2"],
        t: Float[Tensor, "batch 1"],
        frozen: bool = False,
    ) -> Float[Tensor, "batch 2"]:
        if frozen:
            return torch.func.functional_call(
                self,
                self.detached_state(self),
                kwargs={"y": y, "t": t},
            )
        return self.encode(x=self.decode(y=y), t=t)

    def roundtrip_with_decoder_attenuation(
        self,
        *,
        y: Float[Tensor, "batch 2"],
        t: Float[Tensor, "batch 1"],
        attenuation: float,
    ) -> Float[Tensor, "batch 2"]:
        return self.encode(
            x=self.decode_with_decoder_attenuation(y=y, attenuation=attenuation),
            t=t,
        )

    def forward(
        self,
        y: Float[Tensor, "batch 2"],
        t: Float[Tensor, "batch 1"],
    ) -> Float[Tensor, "batch 2"]:
        return self.roundtrip(y=y, t=t)


class BimodalRoundtripCriticModel(BimodalRoundtripModel):
    def __init__(self) -> None:
        super().__init__()
        self.noise_critic = TimeConditionedNoiseCritic()

    def predict_noise(
        self,
        *,
        y: Float[Tensor, "batch 2"],
        t: Float[Tensor, "batch 1"],
        frozen: bool = False,
    ) -> Float[Tensor, "batch 2"]:
        if frozen:
            return torch.func.functional_call(
                self.noise_critic,
                self.detached_state(self.noise_critic),
                args=(y, t),
            )
        return self.noise_critic(y, t)


def sample_time(
    *,
    batch_size: int,
    loss_config: LossConfig,
    device: str,
    dtype: torch.dtype,
) -> Float[Tensor, "batch 1"]:
    return loss_config.t_min + (1.0 - loss_config.t_min) * torch.rand(
        batch_size,
        1,
        device=device,
        dtype=dtype,
    )


def prior_matching_loss(
    *,
    y_data: Float[Tensor, "batch 2"],
) -> Tensor:
    mean = y_data.mean(dim=0)
    centered = y_data - mean
    cov = centered.transpose(0, 1) @ centered
    cov = cov / max(int(y_data.shape[0]) - 1, 1)
    eye = torch.eye(2, device=y_data.device, dtype=y_data.dtype)
    return mean.square().mean() + (cov - eye).square().mean()


def plot_training_history(
    *,
    history: dict[str, list[float]],
    step: int,
) -> go.Figure:
    steps = history["step"]
    grouped_metrics = [
        (
            "sample space",
            [
                ("reconstruction_loss", "reconstruction"),
                ("cycle_data_loss", "cycle data"),
                ("reconstruction_mse", "recon mse"),
            ],
        ),
        (
            "latent space",
            [
                ("total_loss", "total"),
                ("prior_matching_loss", "prior"),
                ("cycle_prior_loss", "cycle prior"),
                ("denoising_loss", "denoising"),
                ("score_loss", "score"),
            ],
        ),
    ]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"sample space at step {step}",
            f"latent space at step {step}",
        ],
        horizontal_spacing=0.10,
    )
    for col, (_, metrics) in enumerate(grouped_metrics, start=1):
        for metric_name, trace_name in metrics:
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=history[metric_name],
                    mode="lines",
                    name=trace_name,
                    line={"width": 2.5},
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text="step", row=1, col=col)
    fig.update_layout(
        height=420,
        width=1250,
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
    )
    return fig


def save_plotly_figure(
    *,
    figure: go.Figure,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        str(path),
        include_plotlyjs="cdn",
        full_html=True,
    )
    figure.write_image(str(path.with_suffix(".png")))


def add_quiver_traces(
    *,
    figure: go.Figure,
    x: Tensor,
    y: Tensor,
    u: Tensor,
    v: Tensor,
    color: str,
    name: str,
    row: int,
    col: int,
    magnitudes: Tensor,
    showlegend: bool = True,
    visible: bool = True,
) -> None:
    quiver_figure = ff.create_quiver(
        x=x.detach().cpu().float().tolist(),
        y=y.detach().cpu().float().tolist(),
        u=u.detach().cpu().float().tolist(),
        v=v.detach().cpu().float().tolist(),
        scale=1.0,
        arrow_scale=0.48,
        line_color=color,
        name=name,
    )
    for trace_index, trace in enumerate(quiver_figure.data):
        trace.showlegend = showlegend and trace_index == 0
        trace.name = name
        trace.line.width = 3.2
        trace.visible = visible
        figure.add_trace(
            trace,
            row=row,
            col=col,
        )
    hover_text = [
        f"{name}<br>|v|={float(magnitude):.4f}"
        for magnitude in magnitudes.detach().cpu().float().tolist()
    ]
    figure.add_trace(
        go.Scatter(
            x=x.detach().cpu().float().tolist(),
            y=y.detach().cpu().float().tolist(),
            mode="markers",
            marker={
                "size": 10,
                "color": color,
                "opacity": 0.001,
            },
            hovertemplate="%{text}<extra></extra>",
            text=hover_text,
            showlegend=False,
            name=name,
            visible=visible,
        ),
        row=row,
        col=col,
    )


def normalize_vectors_for_display(
    *,
    vectors: Tensor,
    display_length: float,
) -> tuple[Tensor, Tensor]:
    magnitudes = vectors.norm(dim=-1, keepdim=True)
    normalized = vectors / magnitudes.clamp_min(1e-8)
    return display_length * normalized, magnitudes.squeeze(-1)


def estimate_latent_score(
    *,
    noisy_latents: Float[Tensor, "batch 2"],
    denoised_latents: Float[Tensor, "batch 2"],
    t: Float[Tensor, "batch 1"],
) -> Float[Tensor, "batch 2"]:
    return ((1.0 - t) * denoised_latents - noisy_latents) / t.square()


def build_latent_vector_field_snapshot(
    *,
    noise_level: float,
    points: Float[Tensor, "batch 2"],
    mode_ids: Tensor,
    vectors: Float[Tensor, "batch 2"],
    arrow_stride: int,
    arrow_display_length: float,
) -> LatentVectorFieldSnapshot:
    display_vectors, vector_magnitudes = normalize_vectors_for_display(
        vectors=vectors,
        display_length=arrow_display_length,
    )
    arrow_indices = torch.arange(
        0,
        points.shape[0],
        arrow_stride,
        device=points.device,
    )
    return LatentVectorFieldSnapshot(
        noise_level=noise_level,
        points=points.detach().cpu(),
        mode_ids=mode_ids.detach().cpu(),
        arrow_points=points[arrow_indices].detach().cpu(),
        arrow_vectors=display_vectors[arrow_indices].detach().cpu(),
        arrow_mode_ids=mode_ids[arrow_indices].detach().cpu(),
        arrow_magnitudes=vector_magnitudes[arrow_indices].detach().cpu(),
    )


def build_latent_score_snapshots(
    *,
    model: BimodalRoundtripModel,
    latent_points: Float[Tensor, "batch 2"],
    latent_mode_ids: Tensor,
    noise_levels: Sequence[float],
    arrow_stride: int,
    arrow_display_length: float,
) -> list[LatentVectorFieldSnapshot]:
    snapshots: list[LatentVectorFieldSnapshot] = []
    for noise_level in noise_levels:
        t = torch.full(
            (latent_points.shape[0], 1),
            fill_value=float(noise_level),
            device=latent_points.device,
            dtype=latent_points.dtype,
        )
        noisy_latents = (1.0 - t) * latent_points + t * torch.randn_like(latent_points)
        denoised_latents = model.roundtrip(y=noisy_latents, t=t)
        score_vectors = estimate_latent_score(
            noisy_latents=noisy_latents,
            denoised_latents=denoised_latents,
            t=t,
        )
        snapshots.append(
            build_latent_vector_field_snapshot(
                noise_level=float(noise_level),
                points=noisy_latents,
                mode_ids=latent_mode_ids,
                vectors=score_vectors,
                arrow_stride=arrow_stride,
                arrow_display_length=arrow_display_length,
            )
        )
    return snapshots


def estimate_denoising_loss_curve(
    *,
    model: BimodalRoundtripModel,
    clean_latents: Float[Tensor, "batch 2"],
    noise_levels: Sequence[float],
    decoder_attenuation: float,
    num_noise_draws: int,
) -> list[float]:
    denoising_losses: list[float] = []
    for noise_level in noise_levels:
        t = torch.full(
            (clean_latents.shape[0], 1),
            fill_value=float(noise_level),
            device=clean_latents.device,
            dtype=clean_latents.dtype,
        )
        accumulated_loss = clean_latents.new_tensor(0.0)
        for _ in range(num_noise_draws):
            noisy_latents = (1.0 - t) * clean_latents + t * torch.randn_like(
                clean_latents
            )
            denoised_latents = model.roundtrip_with_decoder_attenuation(
                y=noisy_latents,
                t=t,
                attenuation=decoder_attenuation,
            )
            accumulated_loss = accumulated_loss + (
                ((denoised_latents - clean_latents).square()) / t.square()
            ).mean()
        denoising_losses.append(float((accumulated_loss / num_noise_draws).item()))
    return denoising_losses


def _latent_score_visibility_mask(
    *,
    snapshot_index: int,
    traces_per_snapshot: list[int],
) -> list[bool]:
    visibility_mask = [False] * sum(traces_per_snapshot)
    start_index = sum(traces_per_snapshot[:snapshot_index])
    end_index = start_index + traces_per_snapshot[snapshot_index]
    for trace_index in range(start_index, end_index):
        visibility_mask[trace_index] = True
    return visibility_mask


def plot_latent_score_snapshot_selector(
    *,
    score_snapshots: Sequence[LatentVectorFieldSnapshot],
    step: int,
) -> go.Figure:
    data_palette = ["#1f77b4", "#ff7f0e"]
    fig = make_subplots(rows=1, cols=1)

    all_points = torch.cat(
        [
            point_set
            for snapshot in score_snapshots
            for point_set in (
                snapshot.points,
                snapshot.arrow_points,
                snapshot.arrow_points + snapshot.arrow_vectors,
            )
        ],
        dim=0,
    )
    x_min = float(all_points[:, 0].min().item())
    x_max = float(all_points[:, 0].max().item())
    y_min = float(all_points[:, 1].min().item())
    y_max = float(all_points[:, 1].max().item())
    x_pad = 0.08 * max(x_max - x_min, 1.0)
    y_pad = 0.08 * max(y_max - y_min, 1.0)
    x_range = [x_min - x_pad, x_max + x_pad]
    y_range = [y_min - y_pad, y_max + y_pad]

    traces_per_snapshot: list[int] = []
    for snapshot_index, snapshot in enumerate(score_snapshots):
        start_trace_count = len(fig.data)
        points_np = snapshot.points.detach().cpu().float().numpy()
        mode_ids_np = snapshot.mode_ids.detach().cpu().numpy()
        arrow_mode_ids_np = snapshot.arrow_mode_ids.detach().cpu().numpy()
        for mode_index, color in enumerate(data_palette):
            mode_mask = mode_ids_np == mode_index
            fig.add_trace(
                go.Scatter(
                    x=points_np[mode_mask, 0],
                    y=points_np[mode_mask, 1],
                    mode="markers",
                    marker={"size": 4, "color": color, "opacity": 0.26},
                    name=f"noisy latent mode {mode_index}",
                    showlegend=snapshot_index == 0,
                    visible=snapshot_index == 0,
                ),
                row=1,
                col=1,
            )
            arrow_mask = arrow_mode_ids_np == mode_index
            if arrow_mask.any():
                add_quiver_traces(
                    figure=fig,
                    x=snapshot.arrow_points[snapshot.arrow_mode_ids == mode_index, 0],
                    y=snapshot.arrow_points[snapshot.arrow_mode_ids == mode_index, 1],
                    u=snapshot.arrow_vectors[snapshot.arrow_mode_ids == mode_index, 0],
                    v=snapshot.arrow_vectors[snapshot.arrow_mode_ids == mode_index, 1],
                    color=color,
                    name=f"score mode {mode_index}",
                    row=1,
                    col=1,
                    magnitudes=snapshot.arrow_magnitudes[
                        snapshot.arrow_mode_ids == mode_index
                    ],
                    showlegend=snapshot_index == 0,
                    visible=snapshot_index == 0,
                )
        traces_per_snapshot.append(len(fig.data) - start_trace_count)

    buttons = []
    for snapshot_index, snapshot in enumerate(score_snapshots):
        buttons.append(
            {
                "label": f"t={snapshot.noise_level:.2f}",
                "method": "update",
                "args": [
                    {
                        "visible": _latent_score_visibility_mask(
                            snapshot_index=snapshot_index,
                            traces_per_snapshot=traces_per_snapshot,
                        )
                    },
                    {
                        "title": (
                            f"encoded latents + estimated score at step {step}"
                            f" for t={snapshot.noise_level:.2f}"
                        )
                    },
                ],
            }
        )

    fig.update_xaxes(
        range=x_range,
        showgrid=True,
        zeroline=False,
        scaleanchor="y",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        range=y_range,
        showgrid=True,
        zeroline=False,
        row=1,
        col=1,
    )
    fig.update_layout(
        height=430,
        width=760,
        margin={"l": 30, "r": 20, "t": 60, "b": 30},
        legend={"x": 1.02, "xanchor": "left", "y": 1.0, "yanchor": "top"},
        title=(
            f"encoded latents + estimated score at step {step}"
            f" for t={score_snapshots[0].noise_level:.2f}"
        ),
        updatemenus=[
            {
                "type": "dropdown",
                "direction": "down",
                "x": 0.0,
                "xanchor": "left",
                "y": 1.16,
                "yanchor": "top",
                "buttons": buttons,
                "showactive": True,
            }
        ],
        annotations=[
            {
                "text": "noise level",
                "x": 0.0,
                "xanchor": "left",
                "y": 1.22,
                "yanchor": "top",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 13},
            }
        ],
    )
    return fig


def plot_denoising_loss_curve(
    *,
    noise_levels: Sequence[float],
    denoising_losses: Sequence[float],
    step: int,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(noise_levels),
            y=list(denoising_losses),
            mode="lines+markers",
            name="estimated denoising loss",
            line={"width": 3.0, "color": "#d62728"},
            marker={"size": 9, "color": "#d62728"},
        )
    )
    fig.update_layout(
        title=f"estimated denoising loss by noise level at step {step}",
        height=420,
        width=760,
        margin={"l": 50, "r": 20, "t": 70, "b": 50},
    )
    fig.update_xaxes(
        title_text="noise level t",
        tickmode="array",
        tickvals=list(noise_levels),
        showgrid=True,
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="E[t^-2 ||f(y_t, t) - y_0||^2]",
        showgrid=True,
        zeroline=False,
    )
    return fig


def plot_bimodal_snapshot(
    *,
    data_points: Tensor,
    reconstruction_points: Tensor,
    data_mode_ids: Tensor,
    model_points: Tensor,
    contour_ids: Tensor,
    contour_levels: Sequence[float],
    mode_centers: Tensor,
    latent_points: Tensor,
    latent_mode_ids: Tensor,
    latent_arrow_points: Tensor,
    latent_arrow_vectors: Tensor,
    latent_arrow_mode_ids: Tensor,
    latent_arrow_magnitudes: Tensor,
    step: int,
) -> go.Figure:
    data_palette = ["#1f77b4", "#ff7f0e"]
    contour_palette = [
        f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"
        for r, g, b, _ in [
            tuple(color)
            for color in torch.tensor(
                [
                    [0.282623, 0.140926, 0.457517, 1.0],
                    [0.253935, 0.265254, 0.529983, 1.0],
                    [0.206756, 0.371758, 0.553117, 1.0],
                    [0.163625, 0.471133, 0.558148, 1.0],
                    [0.127568, 0.566949, 0.550556, 1.0],
                    [0.266941, 0.748751, 0.440573, 1.0],
                    [0.741388, 0.873449, 0.149561, 1.0],
                ]
            )[
                torch.linspace(0, 6, len(contour_levels)).round().long()
            ].tolist()
        ]
    ]

    data_points_np = data_points.detach().cpu().float().numpy()
    reconstruction_points_np = reconstruction_points.detach().cpu().float().numpy()
    model_points_np = model_points.detach().cpu().float().numpy()
    data_mode_ids_np = data_mode_ids.detach().cpu().numpy()
    contour_ids_np = contour_ids.detach().cpu().numpy()
    mode_centers_np = mode_centers.detach().cpu().float().numpy()
    latent_points_np = latent_points.detach().cpu().float().numpy()
    latent_mode_ids_np = latent_mode_ids.detach().cpu().numpy()
    latent_arrow_mode_ids_np = latent_arrow_mode_ids.detach().cpu().numpy()
    all_points = torch.cat(
        [data_points, reconstruction_points, model_points, mode_centers],
        dim=0,
    ).detach()
    x_min = float(all_points[:, 0].min().item())
    x_max = float(all_points[:, 0].max().item())
    y_min = float(all_points[:, 1].min().item())
    y_max = float(all_points[:, 1].max().item())
    x_pad = 0.08 * max(x_max - x_min, 1.0)
    y_pad = 0.08 * max(y_max - y_min, 1.0)
    x_range = [x_min - x_pad, x_max + x_pad]
    y_range = [y_min - y_pad, y_max + y_pad]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            f"data + reconstruction at step {step}",
            f"model distribution at step {step}",
            f"encoded latents + drift at step {step}",
        ],
        horizontal_spacing=0.05,
    )
    fig.add_trace(
        go.Scatter(
            x=data_points_np[:, 0],
            y=data_points_np[:, 1],
            mode="markers",
            marker={"size": 4, "color": "#202020", "opacity": 0.20},
            name="data",
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    for mode_index, color in enumerate(data_palette):
        mode_mask = data_mode_ids_np == mode_index
        fig.add_trace(
            go.Scatter(
                x=reconstruction_points_np[mode_mask, 0],
                y=reconstruction_points_np[mode_mask, 1],
                mode="markers",
                marker={"size": 5, "color": color, "opacity": 0.55},
                name=f"recon mode {mode_index}",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    for contour_index, contour_level in enumerate(contour_levels):
        contour_mask = contour_ids_np == contour_index
        fig.add_trace(
            go.Scatter(
                x=model_points_np[contour_mask, 0],
                y=model_points_np[contour_mask, 1],
                mode="markers",
                marker={
                    "size": 7,
                    "color": contour_palette[contour_index],
                    "opacity": 0.95,
                },
                name=f"z={contour_level}",
                showlegend=True,
            ),
            row=1,
            col=2,
        )

    for mode_index, color in enumerate(data_palette):
        latent_mode_mask = latent_mode_ids_np == mode_index
        fig.add_trace(
            go.Scatter(
                x=latent_points_np[latent_mode_mask, 0],
                y=latent_points_np[latent_mode_mask, 1],
                mode="markers",
                marker={"size": 4, "color": color, "opacity": 0.28},
                name=f"latent mode {mode_index}",
                showlegend=True,
            ),
            row=1,
            col=3,
        )
        arrow_mode_mask = latent_arrow_mode_ids_np == mode_index
        if arrow_mode_mask.any():
            add_quiver_traces(
                figure=fig,
                x=latent_arrow_points[latent_arrow_mode_ids == mode_index, 0],
                y=latent_arrow_points[latent_arrow_mode_ids == mode_index, 1],
                u=latent_arrow_vectors[latent_arrow_mode_ids == mode_index, 0],
                v=latent_arrow_vectors[latent_arrow_mode_ids == mode_index, 1],
                color=color,
                name=f"drift mode {mode_index}",
                row=1,
                col=3,
                magnitudes=latent_arrow_magnitudes[
                    latent_arrow_mode_ids == mode_index
                ],
            )

    for col in [1, 2]:
        fig.add_trace(
            go.Scatter(
                x=mode_centers_np[:, 0],
                y=mode_centers_np[:, 1],
                mode="markers",
                marker={
                    "size": 11,
                    "color": "#111111",
                    "symbol": "x",
                    "line": {"width": 2},
                },
                showlegend=False,
            ),
            row=1,
            col=col,
        )
    for col in [1, 2]:
        fig.update_xaxes(
            range=x_range,
            showgrid=True,
            zeroline=False,
            scaleanchor=f"y{col if col > 1 else ''}",
            row=1,
            col=col,
        )
        fig.update_yaxes(range=y_range, showgrid=True, zeroline=False, row=1, col=col)
    latent_all_points = torch.cat(
        [latent_points, latent_arrow_points, latent_arrow_points + latent_arrow_vectors],
        dim=0,
    )
    latent_x_min = float(latent_all_points[:, 0].min().item())
    latent_x_max = float(latent_all_points[:, 0].max().item())
    latent_y_min = float(latent_all_points[:, 1].min().item())
    latent_y_max = float(latent_all_points[:, 1].max().item())
    latent_x_pad = 0.08 * max(latent_x_max - latent_x_min, 1.0)
    latent_y_pad = 0.08 * max(latent_y_max - latent_y_min, 1.0)
    fig.update_xaxes(
        range=[latent_x_min - latent_x_pad, latent_x_max + latent_x_pad],
        showgrid=True,
        zeroline=False,
        scaleanchor="y3",
        row=1,
        col=3,
    )
    fig.update_yaxes(
        range=[latent_y_min - latent_y_pad, latent_y_max + latent_y_pad],
        showgrid=True,
        zeroline=False,
        row=1,
        col=3,
    )

    fig.update_layout(
        height=430,
        width=1250,
        margin={"l": 30, "r": 20, "t": 60, "b": 30},
        legend={"x": 1.02, "xanchor": "left", "y": 1.0, "yanchor": "top"},
    )
    return fig


def plot_pullback_snapshot(
    *,
    reconstruction_points: Tensor,
    pullback_arrow_points: Tensor,
    pullback_arrow_vectors: Tensor,
    pullback_arrow_mode_ids: Tensor,
    pullback_arrow_magnitudes: Tensor,
    mode_centers: Tensor,
    step: int,
) -> go.Figure:
    data_palette = ["#1f77b4", "#ff7f0e"]
    reconstruction_points_np = reconstruction_points.detach().cpu().float().numpy()
    pullback_arrow_mode_ids_np = pullback_arrow_mode_ids.detach().cpu().numpy()
    mode_centers_np = mode_centers.detach().cpu().float().numpy()
    all_points = torch.cat(
        [reconstruction_points, pullback_arrow_points, mode_centers],
        dim=0,
    )
    x_min = float(all_points[:, 0].min().item())
    x_max = float(all_points[:, 0].max().item())
    y_min = float(all_points[:, 1].min().item())
    y_max = float(all_points[:, 1].max().item())
    x_pad = 0.08 * max(x_max - x_min, 1.0)
    y_pad = 0.08 * max(y_max - y_min, 1.0)
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
            go.Scatter(
                x=reconstruction_points_np[:, 0],
                y=reconstruction_points_np[:, 1],
                mode="markers",
                marker={"size": 6, "color": "#202020", "opacity": 0.12},
                name="reconstruction",
            ),
            row=1,
        col=1,
    )
    for mode_index, color in enumerate(data_palette):
        arrow_mask = pullback_arrow_mode_ids_np == mode_index
        if arrow_mask.any():
            add_quiver_traces(
                figure=fig,
                x=pullback_arrow_points[pullback_arrow_mode_ids == mode_index, 0],
                y=pullback_arrow_points[pullback_arrow_mode_ids == mode_index, 1],
                u=pullback_arrow_vectors[pullback_arrow_mode_ids == mode_index, 0],
                v=pullback_arrow_vectors[pullback_arrow_mode_ids == mode_index, 1],
                color=color,
                name=f"pullback mode {mode_index}",
                row=1,
                col=1,
                magnitudes=pullback_arrow_magnitudes[
                    pullback_arrow_mode_ids == mode_index
                ],
            )
    fig.add_trace(
        go.Scatter(
            x=mode_centers_np[:, 0],
            y=mode_centers_np[:, 1],
            mode="markers",
            marker={
                "size": 11,
                "color": "#111111",
                "symbol": "x",
                "line": {"width": 2},
            },
            name="mode centers",
        ),
        row=1,
        col=1,
    )
    fig.update_layout(
        title=f"reconstruction + score pullback at step {step}",
        height=480,
        width=760,
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
        legend={"x": 1.02, "xanchor": "left", "y": 1.0, "yanchor": "top"},
    )
    fig.update_xaxes(
        range=[x_min - x_pad, x_max + x_pad],
        showgrid=True,
        zeroline=False,
        scaleanchor="y",
    )
    fig.update_yaxes(
        range=[y_min - y_pad, y_max + y_pad],
        showgrid=True,
        zeroline=False,
    )
    return fig
