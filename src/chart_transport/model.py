from __future__ import annotations

from src.config.base import BaseConfig
from src.model.base import ModelConfig


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


__all__ = ["ChartTransportModelConfig"]
