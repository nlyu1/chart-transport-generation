from __future__ import annotations

from typing import Any, Callable

import torch
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset

from src.config.base import ConfigMethodsMixin


def make_generator(*, seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


@dataclass(
    kw_only=True,
    config=ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    ),
)
class DataLoaderConfig(ConfigMethodsMixin):
    batch_size: int
    shuffle: bool
    num_workers: int
    pin_memory: bool
    drop_last: bool

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


def build_dataloader(
    *,
    dataset: Dataset[Any],
    dataloader_config: DataLoaderConfig,
    collate_fn: Callable[[list[Any]], Any],
    generator: torch.Generator,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=dataloader_config.batch_size,
        shuffle=dataloader_config.shuffle,
        generator=generator,
        num_workers=dataloader_config.num_workers,
        pin_memory=dataloader_config.pin_memory,
        drop_last=dataloader_config.drop_last,
        collate_fn=collate_fn,
    )
