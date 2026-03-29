from __future__ import annotations

from pathlib import Path
from typing import Self

from src.common.training import TrainingConfig


class MultimodalGaussianTrainingConfig(TrainingConfig):
    integrated_n_steps: int

    @classmethod
    def initialize(
        cls,
        *,
        seed: int,
        train_batch_size: int,
        eval_batch_size: int,
        integrated_n_steps: int,
        folder: Path,
        raise_on_existing_folder: bool = True,
    ) -> Self:
        cls.prepare_folder(
            folder=folder,
            raise_on_existing_folder=raise_on_existing_folder,
        )
        return cls(
            seed=seed,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            integrated_n_steps=integrated_n_steps,
            folder=folder,
            raise_on_existing_folder=raise_on_existing_folder,
        )


__all__ = ["MultimodalGaussianTrainingConfig"]
