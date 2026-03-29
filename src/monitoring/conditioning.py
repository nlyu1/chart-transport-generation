from __future__ import annotations

from pathlib import Path

import polars as pl
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from torch.func import functional_call, jvp, vjp, vmap

from src.monitoring.configs import ConditioningMonitorConfig
from src.monitoring.distribution import mode_beeswarm_figure
from src.monitoring.utils import MonitorStage
from src.monitoring.utils import (
    sample_mode_batch,
    step_folder,
    write_figure,
)


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


def conditioning_figure(
    *,
    encoder_singular_values: Tensor,
    decoder_singular_values: Tensor,
    labels: Int[Tensor, "batch"],
) -> go.Figure:
    return mode_beeswarm_figure(
        panel_values=[
            encoder_singular_values.float(),
            decoder_singular_values.float(),
        ],
        labels=labels,
        panel_titles=["Encoder", "Decoder"],
        xaxis_title="Largest singular value",
        title="Encoder and decoder conditioning by class",
        value_label="sigma_max",
    )


def _conditioning_summary(
    *,
    prefix: str,
    singular_values: Tensor,
) -> dict[str, float]:
    return {
        f"{prefix}_mean": singular_values.mean().item(),
        f"{prefix}_max": singular_values.max().item(),
    }


def _write_max_singular_values_parquet(
    *,
    step_root: Path,
    labels: Int[Tensor, "batch"],
    encoder_singular_values: Float[Tensor, "batch"],
    decoder_singular_values: Float[Tensor, "batch"],
) -> None:
    path = step_root / "numbers" / "max_singular_values.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    mode_ids = labels.detach().cpu().long().tolist()
    frame = pl.DataFrame(
        {
            "kind": ["encoder"] * len(mode_ids) + ["decoder"] * len(mode_ids),
            "mode_id": mode_ids + mode_ids,
            "sigma_max": (
                encoder_singular_values.detach().cpu().float().tolist()
                + decoder_singular_values.detach().cpu().float().tolist()
            ),
        }
    ).with_columns(
        pl.col("kind").cast(pl.Categorical),
        pl.col("mode_id").cast(pl.Int64),
        pl.col("sigma_max").cast(pl.Float32),
    ).sort(
        by=["kind", "mode_id", "sigma_max"],
    )
    frame.write_parquet(path)


def apply_conditioning_monitor(
    *,
    config: ConditioningMonitorConfig,
    rt,
    step: int,
    stage: MonitorStage,
) -> dict[str, float]:
    samples, labels = sample_mode_batch(
        data_config=rt.runtime_data_config,
        device=rt.device,
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

    folder = step_folder(
        run_folder=rt.tc.folder,
        stage=stage,
        step=step,
    )
    path_stem = folder / "conditioning"
    write_figure(
        figure=conditioning_figure(
            encoder_singular_values=encoder_singular_values,
            decoder_singular_values=decoder_singular_values,
            labels=labels,
        ),
        path_stem=path_stem,
    )
    _write_max_singular_values_parquet(
        step_root=folder,
        labels=labels,
        encoder_singular_values=encoder_singular_values,
        decoder_singular_values=decoder_singular_values,
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
