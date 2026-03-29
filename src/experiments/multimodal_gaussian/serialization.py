from __future__ import annotations

from src.config.base import BaseConfig


class MultimodalSerializationConfig(BaseConfig):
    save_after_chart_pretrain: bool
    save_after_critic_pretrain: bool


__all__ = ["MultimodalSerializationConfig"]
