from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from src.config.base import BaseConfig
from src.data.base import GenerativeBatch
from src.model.base import ModelConfig


class MethodStepOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    total_loss: Tensor
    loss_terms: dict[str, Tensor]


class MethodState(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @abstractmethod
    def compute_losses(
        self,
        *,
        model: nn.Module,
        batch: GenerativeBatch,
    ) -> MethodStepOutput:
        raise NotImplementedError


class MethodConfig(BaseConfig, ABC):
    @abstractmethod
    def get_model_config(self) -> ModelConfig:
        raise NotImplementedError

    def get_model(self) -> nn.Module:
        return self.get_model_config().get_model()

    @abstractmethod
    def initialize_state(self) -> MethodState:
        raise NotImplementedError
