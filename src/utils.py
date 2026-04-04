from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor


def clip_norm(
    tensor: Float[Tensor, "batch ..."], *, max_norm: float
) -> Float[Tensor, "batch ..."]:
    flat_tensor = tensor.flatten(start_dim=1)
    norms = flat_tensor.norm(dim=1, keepdim=True)
    scale = torch.clamp(max_norm / norms.clamp_min(1e-6), max=1.0)
    view_shape = (tensor.shape[0],) + (1,) * (tensor.ndim - 1)
    return tensor * scale.reshape(view_shape)
