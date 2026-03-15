from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset

from src.config.base import ConfigMethodsMixin

from .dataloading import DataLoaderConfig, build_dataloader, make_generator

TRAIN_DATALOADER_SEED_OFFSET = 20_000
VAL_DATALOADER_SEED_OFFSET = 30_000


class DatasetConfig(ConfigMethodsMixin, ABC):
    @abstractmethod
    def get_dataset(
        self,
        *,
        split: Literal["train", "val"],
        seed: int,
    ) -> Dataset[Any]:
        raise NotImplementedError

    @abstractmethod
    def collate_batch(self, samples: list[Any]) -> Any:
        raise NotImplementedError


@dataclass(
    kw_only=True,
    config=ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    ),
)
class DataConfig(ConfigMethodsMixin):
    seed: int
    dataset_config: DatasetConfig
    trainloader_config: DataLoaderConfig
    valloader_config: DataLoaderConfig

    def get_trainloader(self) -> DataLoader[Any]:
        dataset = self.dataset_config.get_dataset(split="train", seed=self.seed)
        return build_dataloader(
            dataset=dataset,
            dataloader_config=self.trainloader_config,
            collate_fn=self.dataset_config.collate_batch,
            generator=make_generator(seed=self.seed + TRAIN_DATALOADER_SEED_OFFSET),
        )

    def get_valloader(self) -> DataLoader[Any]:
        dataset = self.dataset_config.get_dataset(split="val", seed=self.seed)
        return build_dataloader(
            dataset=dataset,
            dataloader_config=self.valloader_config,
            collate_fn=self.dataset_config.collate_batch,
            generator=make_generator(seed=self.seed + VAL_DATALOADER_SEED_OFFSET),
        )
