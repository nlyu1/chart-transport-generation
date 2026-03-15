from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from pydantic import ConfigDict

from src.config.base import BaseConfig


class ModelConfig(BaseConfig, ABC):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        extra="forbid",
        frozen=True,
    )

    @abstractmethod
    def get_model(self) -> nn.Module:
        raise NotImplementedError
