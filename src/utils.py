from __future__ import annotations

import random

import numpy as np
import torch
from torch import Tensor


class _GradientAttenuation(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tensor: Tensor,
        attenuation_factor: float,
    ) -> Tensor:
        if attenuation_factor <= 0.0:
            raise ValueError(
                "attenuation_factor must be positive, "
                f"got {attenuation_factor}",
            )
        ctx.attenuation_factor = attenuation_factor
        return tensor

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Tensor,
    ) -> tuple[Tensor, None]:
        return grad_output / ctx.attenuation_factor, None


def attenuate_gradient(
    tensor: Tensor,
    *,
    attenuation_factor: float,
) -> Tensor:
    return _GradientAttenuation.apply(tensor, attenuation_factor)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
