from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import model_validator

from src.config.base import BaseConfig
from src.method.base import MethodConfig
from src.model.base import ModelConfig

if TYPE_CHECKING:
    from src.method.drifting.model import DriftingModel
    from src.method.drifting.state import DriftingState


class GaussianDriftingLossConfig(BaseConfig):
    objective: Literal["reverse_kl", "forward_kl"]
    bandwidth: float
    drift_scale: float
    exclude_self_interactions: bool
    stability_eps: float

    @model_validator(mode="after")
    def _validate_config(self) -> "GaussianDriftingLossConfig":
        if self.bandwidth <= 0.0:
            raise ValueError("bandwidth must be positive")
        if self.drift_scale <= 0.0:
            raise ValueError("drift_scale must be positive")
        if self.stability_eps <= 0.0:
            raise ValueError("stability_eps must be positive")
        return self


class DriftingModelConfig(ModelConfig):
    decoder_config: ModelConfig
    latent_shape: tuple[int, ...]

    @model_validator(mode="after")
    def _validate_config(self) -> "DriftingModelConfig":
        if len(self.latent_shape) < 1:
            raise ValueError("latent_shape must be non-empty")
        if any(dim <= 0 for dim in self.latent_shape):
            raise ValueError("latent_shape must contain only positive dimensions")
        return self

    def get_model(self) -> "DriftingModel":
        from src.method.drifting.model import DriftingModel

        return DriftingModel(config=self)


class DriftingMethodConfig(MethodConfig):
    kind: Literal["drifting"] = "drifting"
    model: DriftingModelConfig
    loss: GaussianDriftingLossConfig

    def get_model_config(self) -> DriftingModelConfig:
        return self.model

    def initialize_state(self) -> "DriftingState":
        from src.method.drifting.state import DriftingState

        return DriftingState(
            loss_config=self.loss,
            latent_shape=self.model.latent_shape,
        )
