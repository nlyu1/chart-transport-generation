from __future__ import annotations

from src.config.base import BaseConfig


class TrainingConfig(BaseConfig):
    """
    Reusable across methods, specifies method-agnostic training specifics.
    """

    train_batch_size: int
    eval_batch_size: int


__all__ = ["TrainingConfig"]
