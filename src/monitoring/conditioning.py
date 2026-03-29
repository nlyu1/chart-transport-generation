from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from torch.func import functional_call, jvp, vjp, vmap

from src.monitoring.configs import ConditioningMonitorConfig


def _flatten_batch(
    tensor: Tensor,
) -> Tensor:
    if tensor.ndim < 2:
        raise ValueError("expected tensor with batch dimension and at least one data dimension")
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
    eps: float = 1e-8,
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
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    outputs = []
    for batch_start in range(0, inputs.shape[0], config.microbatch_size):
        batch_stop = min(batch_start + config.microbatch_size, inputs.shape[0])
        outputs.append(
            _microbatch_largest_singular_values(
                model=model,
                inputs=inputs[batch_start:batch_stop],
                num_power_iterations=config.num_power_iterations,
                eps=eps,
            )
        )
    return torch.cat(outputs, dim=0)


__all__ = ["largest_jacobian_singular_values"]
