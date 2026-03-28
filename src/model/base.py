from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn

from src.config.base import BaseConfig


class ModelConfig(BaseConfig, ABC):
    @abstractmethod
    def get_model(self) -> nn.Module:
        raise NotImplementedError
