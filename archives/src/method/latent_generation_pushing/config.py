from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from jaxtyping import Float
from pydantic import model_validator
from torch import Tensor

from src.config.base import BaseConfig
from src.method.base import MethodConfig
from src.method.latent_generation.config import LatentNoiseWeightConfig
from src.model.base import ModelConfig

if TYPE_CHECKING:
    from src.method.latent_generation_pushing.model import LatentPushGenerationModel
    from src.method.latent_generation_pushing.state import LatentPushGenerationState


class LatentPushGenerationLossConfig(BaseConfig):
    t_min: float
    noise_weight: LatentNoiseWeightConfig
    reconstruction_weight: float
    prior_matching_weight: float
    cycle_prior_weight: float
    denoising_weight: float
    score_matching_weight: float

    @model_validator(mode="after")
    def _validate_config(self) -> "LatentPushGenerationLossConfig":
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


class LatentPushGenerationModelConfig(ModelConfig):
    encoder_config: ModelConfig
    decoder_config: ModelConfig
    latent_shape: tuple[int, ...]

    def get_model(self) -> "LatentPushGenerationModel":
        from src.method.latent_generation_pushing.model import LatentPushGenerationModel

        return LatentPushGenerationModel(config=self)


class LatentPushGenerationMethodConfig(MethodConfig):
    kind: Literal["latent_generation_pushing"] = "latent_generation_pushing"
    model: LatentPushGenerationModelConfig
    loss: LatentPushGenerationLossConfig
    decoder_attenuation: float

    @model_validator(mode="after")
    def _validate_config(self) -> "LatentPushGenerationMethodConfig":
        if self.decoder_attenuation <= 0.0:
            raise ValueError(
                "decoder_attenuation must be positive, "
                f"got {self.decoder_attenuation}",
            )
        return self

    def get_model_config(self) -> LatentPushGenerationModelConfig:
        return self.model

    def initialize_state(self) -> "LatentPushGenerationState":
        from src.method.latent_generation_pushing.state import LatentPushGenerationState

        return LatentPushGenerationState(
            loss_config=self.loss,
            latent_shape=self.model.latent_shape,
            decoder_attenuation=self.decoder_attenuation,
        )
