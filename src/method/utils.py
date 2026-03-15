from __future__ import annotations

from einops import einsum
from jaxtyping import Float
from torch import Tensor


def apply_batch_scalars(
    tensor: Tensor,
    scalars: Float[Tensor, "batch 1"],
) -> Tensor:
    return einsum(
        tensor,
        scalars[:, 0],
        "batch ..., batch -> batch ...",
    )


def gaussian_posterior_mean(
    *,
    y_t: Tensor,
    t: Float[Tensor, "batch 1"],
) -> Tensor:
    coef = (1.0 - t) / ((1.0 - t).square() + t.square())
    return apply_batch_scalars(y_t, coef)


def weighted_mse(
    *,
    pred: Tensor,
    target: Tensor,
    weight: Float[Tensor, "batch 1"],
) -> Tensor:
    return apply_batch_scalars(
        (pred - target).square(),
        weight,
    ).mean()
