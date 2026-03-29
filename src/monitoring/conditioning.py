from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float
from jaxtyping import Int
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch import Tensor
from torch.func import functional_call, jvp, vjp, vmap
import wandb

from src.monitoring.configs import ConditioningMonitorConfig
from src.monitoring.utils import marker_color, step_folder, write_figure


def _flatten_batch(
    tensor: Tensor,
) -> Tensor:
    if tensor.ndim < 2:
        raise ValueError(
            "expected tensor with batch dimension and at least one data dimension"
        )
    return tensor.reshape(tensor.shape[0], -1)


def _model_state(
    *,
    model: nn.Module,
) -> dict[str, Tensor]:
    return {
        **dict(model.named_parameters()),
        **dict(model.named_buffers()),
    }


def _normalize_vector(
    *,
    vector: Tensor,
    eps: float,
) -> Tensor:
    return vector / vector.norm().clamp_min(eps)


def _single_sample_forward(
    *,
    model: nn.Module,
    input_shape: torch.Size,
):
    state = _model_state(model=model)

    def single_sample_forward(
        sample: Float[Tensor, "..."],
    ) -> Float[Tensor, "flattened_output"]:
        output = functional_call(
            model,
            state,
            args=(sample.reshape(input_shape).unsqueeze(0),),
        )
        if not isinstance(output, Tensor):
            raise TypeError("model(inputs) must return a Tensor")
        if output.shape[0] != 1:
            raise ValueError("model must preserve the leading batch dimension")
        return output.squeeze(0).reshape(-1)

    return single_sample_forward


def _microbatch_largest_singular_values(
    *,
    model: nn.Module,
    inputs: Float[Tensor, "batch ..."],
    num_power_iterations: int,
    eps: float,
) -> Float[Tensor, "batch"]:
    flattened_inputs = _flatten_batch(inputs)
    single_sample_forward = _single_sample_forward(
        model=model,
        input_shape=inputs.shape[1:],
    )

    def single_sample_largest_singular_value(
        sample: Float[Tensor, "..."],
    ) -> Float[Tensor, ""]:
        vector = _normalize_vector(
            vector=torch.randn_like(sample),
            eps=eps,
        )
        _, vjp_fn = vjp(
            single_sample_forward,
            sample,
        )
        for _ in range(num_power_iterations):
            _, jacobian_vector_product = jvp(
                single_sample_forward,
                (sample,),
                (vector,),
            )
            vector = _normalize_vector(
                vector=vjp_fn(jacobian_vector_product)[0],
                eps=eps,
            )
        _, jacobian_vector_product = jvp(
            single_sample_forward,
            (sample,),
            (vector,),
        )
        return jacobian_vector_product.norm().float()

    device_type = inputs.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        return vmap(
            single_sample_largest_singular_value,
            randomness="different",
        )(flattened_inputs).detach()


def largest_jacobian_singular_values(
    *,
    model: nn.Module,
    inputs: Float[Tensor, "batch ..."],
    config: ConditioningMonitorConfig,
) -> Float[Tensor, "batch"]:
    """
    Return matrix-free estimates of the largest per-sample Jacobian singular value.

    This is intended for monitoring. It assumes sample independence across the
    batch dimension and processes the incoming batch in monitor-configured
    microbatches.
    """
    if not inputs.is_floating_point():
        raise TypeError("inputs must be floating-point")
    if config.num_power_iterations <= 0:
        raise ValueError("config.num_power_iterations must be positive")
    if config.microbatch_size <= 0:
        raise ValueError("config.microbatch_size must be positive")

    outputs = []
    for batch_start in range(0, inputs.shape[0], config.microbatch_size):
        batch_stop = min(batch_start + config.microbatch_size, inputs.shape[0])
        outputs.append(
            _microbatch_largest_singular_values(
                model=model,
                inputs=inputs[batch_start:batch_stop],
                num_power_iterations=config.num_power_iterations,
                eps=1e-8,
            )
        )
    return torch.cat(outputs, dim=0)

def _sample_mode_batch(
    *,
    rt,
    batch_size_per_mode: int,
) -> tuple[Tensor, Int[Tensor, "batch"]]:
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


def _swarm_offsets(
    *,
    count: int,
    device: torch.device,
) -> Tensor:
    if count <= 1:
        return torch.zeros(count, device=device, dtype=torch.float32)
    left = torch.arange(count, device=device)
    offsets = torch.div(left + 1, 2, rounding_mode="floor").float()
    signs = torch.where(left % 2 == 0, 1.0, -1.0)
    offsets = offsets * signs
    offsets = offsets / offsets.abs().max().clamp_min(1.0)
    return 0.24 * offsets


def _conditioning_panel_traces(
    *,
    figure: go.Figure,
    singular_values: Tensor,
    labels: Int[Tensor, "batch"],
    col: int,
) -> None:
    singular_values_cpu = singular_values.detach().cpu().float()
    labels_cpu = labels.detach().cpu().long()
    num_modes = int(labels_cpu.max().item()) + 1
    for mode_id in range(num_modes):
        mask = labels_cpu == mode_id
        if not mask.any():
            continue
        mode_values = singular_values_cpu[mask]
        sorted_indices = torch.argsort(mode_values)
        sorted_values = mode_values[sorted_indices]
        offsets = _swarm_offsets(
            count=int(sorted_values.shape[0]),
            device=sorted_values.device,
        )
        y_values = torch.full_like(sorted_values, float(mode_id)) + offsets
        figure.add_trace(
            go.Scatter(
                x=sorted_values.tolist(),
                y=y_values.tolist(),
                mode="markers",
                marker={
                    "size": 7,
                    "color": marker_color(group_id=mode_id),
                    "opacity": 0.72,
                    "line": {"width": 0.5, "color": "rgba(0, 0, 0, 0.18)"},
                },
                name=f"mode {mode_id}",
                customdata=[[mode_id]] * int(sorted_values.shape[0]),
                hovertemplate=(
                    "mode=%{customdata[0]}"
                    + "<br>sigma_max=%{x:.4f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=col,
        )


def conditioning_figure(
    *,
    encoder_singular_values: Tensor,
    decoder_singular_values: Tensor,
    labels: Int[Tensor, "batch"],
) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        subplot_titles=("Encoder", "Decoder"),
    )
    _conditioning_panel_traces(
        figure=figure,
        singular_values=encoder_singular_values,
        labels=labels,
        col=1,
    )
    _conditioning_panel_traces(
        figure=figure,
        singular_values=decoder_singular_values,
        labels=labels,
        col=2,
    )

    labels_cpu = labels.detach().cpu().long()
    num_modes = int(labels_cpu.max().item()) + 1
    tick_values = list(range(num_modes))
    tick_text = [str(mode_id) for mode_id in tick_values]

    figure.update_xaxes(title="Largest singular value", row=1, col=1)
    figure.update_xaxes(title="Largest singular value", row=1, col=2)
    figure.update_yaxes(
        title="Mode",
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_text,
        autorange="reversed",
        row=1,
        col=1,
    )
    figure.update_yaxes(
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_text,
        autorange="reversed",
        row=1,
        col=2,
    )
    figure.update_layout(
        template="plotly_white",
        width=1100,
        height=max(480, 120 * num_modes),
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
        title="Encoder and decoder conditioning by mode",
    )
    return figure


def _conditioning_summary(
    *,
    prefix: str,
    singular_values: Tensor,
) -> dict[str, float]:
    return {
        f"{prefix}_mean": singular_values.mean().item(),
        f"{prefix}_max": singular_values.max().item(),
    }


def apply_conditioning_monitor(
    *,
    config: ConditioningMonitorConfig,
    rt,
    step: int,
) -> dict[str, float]:
    samples, labels = _sample_mode_batch(
        rt=rt,
        batch_size_per_mode=config.n_data_samples_per_mode,
    )
    with torch.no_grad():
        latents = rt.chart_transport_model.encoder(samples).float()

    encoder_singular_values = config.largest_singular_values(
        model=rt.chart_transport_model.encoder,
        inputs=samples.float(),
    ).float()
    decoder_singular_values = config.largest_singular_values(
        model=rt.chart_transport_model.decoder,
        inputs=latents.float(),
    ).float()

    folder = step_folder(run_folder=rt.tc.folder, step=step)
    path_stem = folder / "conditioning"
    write_figure(
        figure=conditioning_figure(
            encoder_singular_values=encoder_singular_values,
            decoder_singular_values=decoder_singular_values,
            labels=labels,
        ),
        path_stem=path_stem,
    )

    if rt.tc.monitor_config.use_wandb:
        rt.wandb_run.log(
            {
                "pretrain/local_step": step,
                "pretrain/conditioning_plot": wandb.Image(
                    str(path_stem.with_suffix(".png"))
                ),
            }
        )

    return {
        **_conditioning_summary(
            prefix="encoder_conditioning",
            singular_values=encoder_singular_values,
        ),
        **_conditioning_summary(
            prefix="decoder_conditioning",
            singular_values=decoder_singular_values,
        ),
    }


__all__ = [
    "apply_conditioning_monitor",
    "conditioning_figure",
    "largest_jacobian_singular_values",
]
