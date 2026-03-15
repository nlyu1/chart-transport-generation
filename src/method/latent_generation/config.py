from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
from jaxtyping import Float
from pydantic import model_validator
from torch import Tensor

from src.config.base import BaseConfig
from src.method.base import MethodConfig
from src.model.base import ModelConfig


class LatentNoiseWeightConfig(BaseConfig, ABC):
    @abstractmethod
    def get_weight(
        self,
        *,
        t: Float[Tensor, "batch 1"],
    ) -> Float[Tensor, "batch 1"]:
        raise NotImplementedError


class MLELatentNoiseWeightConfig(LatentNoiseWeightConfig):
    kind: Literal["mle"] = "mle"

    def get_weight(
        self,
        *,
        t: Float[Tensor, "batch 1"],
    ) -> Float[Tensor, "batch 1"]:
        return (1.0 - t) / t.pow(3)


class FlowMatchingLatentNoiseWeightConfig(LatentNoiseWeightConfig):
    kind: Literal["flow_matching"] = "flow_matching"

    def get_weight(
        self,
        *,
        t: Float[Tensor, "batch 1"],
    ) -> Float[Tensor, "batch 1"]:
        return 1.0 / t.pow(2)


class LatentGenerationLossConfig(BaseConfig):
    t_min: float
    noise_weight: LatentNoiseWeightConfig
    reconstruction_weight: float
    cycle_data_weight: float
    cycle_prior_weight: float
    denoising_weight: float
    score_weight: float

    @model_validator(mode="after")
    def _validate_config(self) -> "LatentGenerationLossConfig":
        if not (0.0 < self.t_min < 1.0):
            raise ValueError(f"t_min must lie in (0, 1), got {self.t_min}")
        return self

    def sample_time(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Float[Tensor, "batch 1"]:
        return self.t_min + (1.0 - self.t_min) * torch.rand(
            batch_size,
            1,
            device=device,
            dtype=dtype,
        )

    def get_noise_weight(
        self,
        *,
        t: Float[Tensor, "batch 1"],
    ) -> Float[Tensor, "batch 1"]:
        return self.noise_weight.get_weight(t=t)


class LatentGenerationModelConfig(ModelConfig):
    encoder_config: ModelConfig
    decoder_config: ModelConfig
    data_shape: tuple[int, ...]
    latent_shape: tuple[int, ...]

    def get_model(self):
        from src.method.latent_generation.model import LatentGenerationModel

        return LatentGenerationModel(config=self)


class LatentGenerationMethodConfig(MethodConfig):
    kind: Literal["latent_generation"] = "latent_generation"
    model: LatentGenerationModelConfig
    loss: LatentGenerationLossConfig

    def get_model_config(self) -> LatentGenerationModelConfig:
        return self.model

    def initialize_state(self):
        from src.method.latent_generation.state import LatentGenerationState

        return LatentGenerationState(
            loss_config=self.loss,
            latent_shape=self.model.latent_shape,
        )
