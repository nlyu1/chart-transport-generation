from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jaxtyping import Int
import plotly.graph_objects as go
import torch
import wandb

from src.monitoring.configs import ConditioningMonitorConfig

if TYPE_CHECKING:
    from src.experiments.multimodal_gaussian.state import MultimodalTrainingRuntime


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


def _step_folder(
    *,
    rt: "MultimodalTrainingRuntime",
    step: int,
) -> Path:
    folder = rt.tc.folder / str(step)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _write_figure(
    *,
    figure: go.Figure,
    path_stem: Path,
) -> None:
    figure.write_html(path_stem.with_suffix(".html"))
    figure.write_image(path_stem.with_suffix(".png"))


def _sample_mode_batch(
    *,
    rt: "MultimodalTrainingRuntime",
    batch_size_per_mode: int,
) -> tuple[torch.Tensor, Int[torch.Tensor, "batch"]]:
    samples = []
    labels = []
    for mode_id in range(rt.runtime_data_config.num_modes):
        samples.append(
            rt.runtime_data_config.sample_class(
                mode_id=mode_id,
                batch_size=batch_size_per_mode,
            )
        )
        labels.append(
            torch.full(
                (batch_size_per_mode,),
                fill_value=mode_id,
                device=rt.device,
                dtype=torch.long,
            )
        )
    return torch.cat(samples, dim=0), torch.cat(labels, dim=0)


def _conditioning_figure(
    *,
    singular_values: torch.Tensor,
    labels: Int[torch.Tensor, "batch"],
    title: str,
) -> go.Figure:
    figure = go.Figure()
    singular_values_cpu = singular_values.detach().cpu().float()
    labels_cpu = labels.detach().cpu().long()
    for mode_id in range(int(labels_cpu.max().item()) + 1):
        mask = labels_cpu == mode_id
        if not mask.any():
            continue
        figure.add_trace(
            go.Violin(
                x=[f"mode {mode_id}"] * int(mask.sum().item()),
                y=singular_values_cpu[mask].tolist(),
                name=f"mode {mode_id}",
                box_visible=True,
                meanline_visible=True,
                points="suspectedoutliers",
                fillcolor=COLOR_BANK[mode_id % len(COLOR_BANK)],
                line={"color": COLOR_BANK[mode_id % len(COLOR_BANK)]},
                opacity=0.72,
            )
        )
    figure.update_layout(
        template="plotly_white",
        width=1100,
        height=560,
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
        title=title,
        yaxis_title="Largest singular value",
        xaxis_title="Data mode",
        showlegend=False,
        violinmode="group",
    )
    return figure


def _conditioning_summary(
    *,
    prefix: str,
    singular_values: torch.Tensor,
    labels: Int[torch.Tensor, "batch"],
) -> dict[str, float]:
    summary = {
        f"{prefix}_mean": singular_values.mean().item(),
        f"{prefix}_max": singular_values.max().item(),
    }
    labels_cpu = labels.detach().cpu().long()
    singular_values_cpu = singular_values.detach().cpu().float()
    for mode_id in range(int(labels_cpu.max().item()) + 1):
        mask = labels_cpu == mode_id
        if not mask.any():
            continue
        summary[f"{prefix}_mean_mode_{mode_id}"] = singular_values_cpu[mask].mean().item()
    return summary


class GaussianConditioningMonitorConfig(ConditioningMonitorConfig):
    def apply_to(
        self,
        *,
        rt: "MultimodalTrainingRuntime",
        step: int,
    ) -> dict[str, float]:
        samples, labels = _sample_mode_batch(
            rt=rt,
            batch_size_per_mode=self.n_data_samples_per_mode,
        )
        with torch.no_grad():
            latents = rt.chart_transport_model.encoder(samples).float()

        encoder_singular_values = self.largest_singular_values(
            model=rt.chart_transport_model.encoder,
            inputs=samples.float(),
        ).float()
        decoder_singular_values = self.largest_singular_values(
            model=rt.chart_transport_model.decoder,
            inputs=latents.float(),
        ).float()

        folder = _step_folder(rt=rt, step=step)
        encoder_path_stem = folder / "encoder_conditioning"
        decoder_path_stem = folder / "decoder_conditioning"
        _write_figure(
            figure=_conditioning_figure(
                singular_values=encoder_singular_values,
                labels=labels,
                title="Encoder conditioning by mode",
            ),
            path_stem=encoder_path_stem,
        )
        _write_figure(
            figure=_conditioning_figure(
                singular_values=decoder_singular_values,
                labels=labels,
                title="Decoder conditioning by mode",
            ),
            path_stem=decoder_path_stem,
        )

        if rt.tc.monitor_config.use_wandb:
            rt.wandb_run.log(
                {
                    "pretrain/local_step": step,
                    "pretrain/encoder_conditioning_plot": wandb.Image(
                        str(encoder_path_stem.with_suffix(".png"))
                    ),
                    "pretrain/decoder_conditioning_plot": wandb.Image(
                        str(decoder_path_stem.with_suffix(".png"))
                    ),
                }
            )

        return {
            **_conditioning_summary(
                prefix="encoder_conditioning",
                singular_values=encoder_singular_values,
                labels=labels,
            ),
            **_conditioning_summary(
                prefix="decoder_conditioning",
                singular_values=decoder_singular_values,
                labels=labels,
            ),
        }


__all__ = ["GaussianConditioningMonitorConfig"]
