from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

if TYPE_CHECKING:
    from src.method.latent_generation.config import LatentGenerationModelConfig


class LatentGenerationModel(nn.Module):
    def __init__(self, *, config: "LatentGenerationModelConfig") -> None:
        super().__init__()
        self.config = config
        self.encoder = config.encoder_config.get_model()
        self.decoder = config.decoder_config.get_model()

    @staticmethod
    def _detached_state(module: nn.Module) -> dict[str, Tensor]:
        return {
            **{name: parameter.detach() for name, parameter in module.named_parameters()},
            **{name: buffer.detach() for name, buffer in module.named_buffers()},
        }

    def encode(
        self,
        *,
        x: Tensor,
        t: Float[Tensor, "batch 1"] | None = None,
        apply_as_frozen: bool = False,
    ) -> Tensor:
        if t is None:
            t = torch.zeros(
                x.shape[0],
                1,
                device=x.device,
                dtype=x.dtype,
            )
        if apply_as_frozen:
            return torch.func.functional_call(
                self.encoder,
                self._detached_state(self.encoder),
                args=(x, t),
            )
        return self.encoder(x, t)

    def decode(
        self,
        *,
        y: Tensor,
        apply_as_frozen: bool = False,
    ) -> Tensor:
        if apply_as_frozen:
            return torch.func.functional_call(
                self.decoder,
                self._detached_state(self.decoder),
                args=(y,),
            )
        return self.decoder(y)

    def reconstruct(
        self,
        *,
        x: Tensor,
        apply_as_frozen: bool = False,
    ) -> Tensor:
        return self.decode(
            y=self.encode(x=x, apply_as_frozen=apply_as_frozen),
            apply_as_frozen=apply_as_frozen,
        )

    def roundtrip(
        self,
        *,
        y: Tensor,
        t: Float[Tensor, "batch 1"] | None = None,
        apply_as_frozen: bool = False,
    ) -> Tensor:
        if apply_as_frozen:
            return torch.func.functional_call(
                self,
                self._detached_state(self),
                kwargs={"y": y, "t": t},
            )
        x = self.decode(y=y)
        return self.encode(x=x, t=t)

    def forward(
        self,
        y: Tensor,
        t: Float[Tensor, "batch 1"] | None = None,
        apply_as_frozen: bool = False,
    ) -> Tensor:
        return self.roundtrip(y=y, t=t, apply_as_frozen=apply_as_frozen)
