from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from src.method.drifting.config import DriftingModelConfig


class DriftingModel(nn.Module):
    def __init__(self, *, config: "DriftingModelConfig") -> None:
        super().__init__()
        self.config = config
        self.decoder = config.decoder_config.get_model()

    def decode(
        self,
        *,
        y: Tensor,
    ) -> Tensor:
        return self.decoder(y)

    def forward(self, y: Tensor) -> Tensor:
        return self.decode(y=y)
