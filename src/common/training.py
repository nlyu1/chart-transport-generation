from __future__ import annotations

from pathlib import Path
from typing import Self

from src.config.base import BaseConfig


class TrainingConfig(BaseConfig):
    """
    Reusable across methods, specifies method-agnostic training specifics.
    """

    seed: int
    train_batch_size: int
    eval_batch_size: int
    folder: Path
    raise_on_existing_folder: bool = True

    @staticmethod
    def prepare_folder(
        *,
        folder: Path,
        raise_on_existing_folder: bool,
    ) -> None:
        if folder.exists():
            if not folder.is_dir():
                raise NotADirectoryError(f"{folder} exists and is not a directory")
            if raise_on_existing_folder:
                raise FileExistsError(f"{folder} already exists")
            return
        folder.mkdir(parents=True, exist_ok=False)

    @classmethod
    def initialize(
        cls,
        *,
        seed: int,
        train_batch_size: int,
        eval_batch_size: int,
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
            folder=folder,
            raise_on_existing_folder=raise_on_existing_folder,
        )


__all__ = ["TrainingConfig"]
