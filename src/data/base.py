from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torch.utils.data import DataLoader


@dataclass(
    kw_only=True,
    config=ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    ),
)
class DataConfig(ABC):
    """Declarative contract for train/validation data materialization."""

    seed: int

    @abstractmethod
    def get_trainloader(self) -> DataLoader[Any]:
        """Return the train split loader."""
        raise NotImplementedError

    @abstractmethod
    def get_valloader(self) -> DataLoader[Any]:
        """Return the validation split loader."""
        raise NotImplementedError
