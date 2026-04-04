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


def dict_to_cpu(tensor_dict: dict[Any, Tensor]) -> dict[Any, Tensor]:
    result = {}
    for k, v in tensor_dict.items():
        if not v.is_cuda:
            result[k] = v
            continue
        buf = torch.empty(v.shape, dtype=v.dtype, pin_memory=True)
        buf.copy_(v.detach(), non_blocking=True)
        result[k] = buf
    torch.cuda.synchronize()
    return result
