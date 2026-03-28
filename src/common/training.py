from __future__ import annotations

from pathlib import Path
from typing import Self

from src.config.base import BaseConfig


class TrainingConfig(BaseConfig):
    """
    Reusable across methods, specifies method-agnostic training specifics.
    """

    train_batch_size: int
    eval_batch_size: int
    eval_every_n_training_steps: int
    folder: Path
    raise_on_existing_folder: bool = True

    @classmethod
    def initialize(
        cls,
        *,
        train_batch_size: int,
        eval_batch_size: int,
        eval_every_n_training_steps: int,
        folder: Path,
        raise_on_existing_folder: bool = True,
    ) -> Self:
        if folder.exists():
            if not folder.is_dir():
                raise NotADirectoryError(f"{folder} exists and is not a directory")
            if raise_on_existing_folder:
                raise FileExistsError(f"{folder} already exists")
        else:
            folder.mkdir(parents=True, exist_ok=False)

        return cls(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            eval_every_n_training_steps=eval_every_n_training_steps,
            folder=folder,
            raise_on_existing_folder=raise_on_existing_folder,
        )


__all__ = ["TrainingConfig"]
