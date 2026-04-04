from __future__ import annotations

import torch.nn as nn
from torch import optim

from src.config.base import BaseConfig
from src.model.base import ModelConfig


class ChartTransportModel(nn.Module):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        decoder: nn.Module,
        critic: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.critic = critic


class ChartTransportModelConfig(BaseConfig):
    """
    The critic must be time-conditioned.
    Defaults to AdamW with standard weight decay.
    """

    encoder: ModelConfig
    decoder: ModelConfig
    critic: ModelConfig

    chart_lr: float
    """Encoder-decoder learning rate."""

    critic_lr: float
    """Critic learning rate."""

    grad_clip_norm: float
    """Gradient clipping norm applied to the bundled optimizer."""

    def get_model(self) -> ChartTransportModel:
        return ChartTransportModel(
            encoder=self.encoder.get_model(),
            decoder=self.decoder.get_model(),
            critic=self.critic.get_model(),
        )

    def get_optimizer(
        self,
        model: ChartTransportModel,
    ) -> optim.Optimizer:
        return optim.AdamW(
            params=[
                {
                    "params": [
                        *model.encoder.parameters(),
                        *model.decoder.parameters(),
                    ],
                    "lr": self.chart_lr,
                },
                {
                    "params": list(model.critic.parameters()),
                    "lr": self.critic_lr,
                },
            ],
        )


__all__ = ["ChartTransportModel", "ChartTransportModelConfig"]
