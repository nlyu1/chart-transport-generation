from __future__ import annotations

import torch
from torch import Tensor

from src.method.latent_generation.model import LatentGenerationModel


class LatentPushGenerationModel(LatentGenerationModel):
    def decode_jvp(
        self,
        *,
        y: Tensor,
        tangent: Tensor,
        apply_as_frozen: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if apply_as_frozen:
            decoder_state = self._detached_state(self.decoder)

            def decode_fn(latent: Tensor) -> Tensor:
                return torch.func.functional_call(
                    self.decoder,
                    decoder_state,
                    args=(latent,),
                )

        else:

            def decode_fn(latent: Tensor) -> Tensor:
                return self.decoder(latent)

        return torch.func.jvp(
            decode_fn,
            (y,),
            (tangent,),
        )
