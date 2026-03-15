from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

if False:
    from src.method.latent_generation.config import LatentGenerationModelConfig


class LatentGenerationModel(nn.Module):
    def __init__(self, *, config: "LatentGenerationModelConfig") -> None:
        super().__init__()
        self.config = config
        self.encoder = config.encoder_config.get_model()
        self.decoder = config.decoder_config.get_model()

    def encode(
        self,
        *,
        x: Tensor,
        t: Float[Tensor, "batch 1"],
    ) -> Tensor:
        return self.encoder(x, t)

    def encode_plain(self, *, x: Tensor) -> Tensor:
        t = torch.zeros(
            x.shape[0],
            1,
            device=x.device,
            dtype=x.dtype,
        )
        return self.encode(x=x, t=t)

    def decode(self, *, y: Tensor) -> Tensor:
        return self.decoder(y)

    def reconstruct(self, *, x: Tensor) -> Tensor:
        return self.decode(y=self.encode_plain(x=x))

    def roundtrip(
        self,
        *,
        y: Tensor,
        t: Float[Tensor, "batch 1"],
    ) -> Tensor:
        x = self.decode(y=y)
        return self.encode(x=x, t=t)

    def forward(
        self,
        y: Tensor,
        t: Float[Tensor, "batch 1"],
    ) -> Tensor:
        return self.roundtrip(y=y, t=t)
