from __future__ import annotations

from torch.nn.utils import clip_grad_norm_

import torch
from jaxtyping import Float
from torch import Tensor

from src.config.base import BaseConfig
from src.data.base import BaseDataConfig
from src.drifting.model import (
    AffineGaussianTransportModel,
    AffineGaussianTransportModelConfig,
)
from src.drifting.transport import ReverseKLDriftingLossConfig
from src.priors.base import BasePriorConfig


class DriftingStudyConfig(BaseConfig):
    data: BaseDataConfig
    prior: BasePriorConfig
    model: AffineGaussianTransportModelConfig
    drifting: ReverseKLDriftingLossConfig


class DriftingStudyState(BaseConfig):
    config: DriftingStudyConfig
    model: AffineGaussianTransportModel
    op: torch.optim.Optimizer
    device: torch.device

    @classmethod
    def initialize(
        cls,
        *,
        config: DriftingStudyConfig,
        device: torch.device,
    ) -> "DriftingStudyState":
        data_config = config.data
        if hasattr(data_config, "to"):
            data_config = data_config.to(device=device)
            config = config.replace(path="data", replacement=data_config)
        model = config.model.get_model().to(device)
        op = config.model.get_optimizer(model)
        return cls(
            config=config,
            model=model,
            op=op,
            device=device,
        )

    def sample_prior(
        self,
        *,
        batch_size: int,
    ) -> Float[Tensor, "batch 2"]:
        return self.prior_config.sample(batch_size=batch_size).to(self.device)

    def decode(
        self,
        latent: Float[Tensor, "batch 2"],
    ) -> Float[Tensor, "batch 2"]:
        return self.model(latent)

    def sample_model(
        self,
        *,
        batch_size: int,
        latent: Float[Tensor, "batch 2"] | None = None,
    ) -> Float[Tensor, "batch 2"]:
        if latent is None:
            latent = self.sample_prior(batch_size=batch_size)
        return self.decode(latent)

    def compute_drifting_loss(
        self,
        *,
        data_samples: Float[Tensor, "batch 2"],
    ) -> ReverseKLDriftingLossConfig.Loss:
        batch_size = data_samples.shape[0]
        latent = self.sample_prior(batch_size=batch_size).type_as(data_samples)
        return self.config.drifting.apply(
            self,
            data_samples=data_samples,
            latent=latent,
        )

    def step_and_zero_grad(self) -> None:
        clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.model.grad_clip_norm,
        )
        self.op.step()
        self.op.zero_grad(set_to_none=True)

    @property
    def prior_config(self) -> BasePriorConfig:
        return self.config.prior

    def model_distribution(self) -> torch.distributions.MultivariateNormal:
        return self.model.distribution()


__all__ = [
    "DriftingStudyConfig",
    "DriftingStudyState",
]
