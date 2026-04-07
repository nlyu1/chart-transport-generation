from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from src.common.training import BaseLoss
from src.config.base import BaseConfig
from src.data.base import BaseDataConfig
from src.deterministic_chart_transport.constraint import (
    ChartPretrainConfig,
    IntegratedChartConstraintConfig,
)
from src.deterministic_chart_transport.critic import CriticLossConfig
from src.deterministic_chart_transport.model import ChartTransportModelConfig
from src.deterministic_chart_transport.transport import (
    DeterministicChartTransportLossConfig,
)
from src.priors.base import BasePriorConfig


class DeterministicChartTransportStudyConfig(BaseConfig):
    data: BaseDataConfig
    prior: BasePriorConfig

    model: ChartTransportModelConfig
    pretrain: ChartPretrainConfig
    critic: CriticLossConfig
    transport: DeterministicChartTransportLossConfig
    integrated_constraint: IntegratedChartConstraintConfig


class DeterministicChartTransportStudyState(BaseConfig):
    config: DeterministicChartTransportStudyConfig

    model: nn.Module
    op: torch.optim.Optimizer
    device: torch.device

    @classmethod
    def initialize(
        cls,
        *,
        config: DeterministicChartTransportStudyConfig,
        device: torch.device,
    ):
        model = config.model.get_model().to(device)
        op = config.model.get_optimizer(model)
        return cls(
            config=config,
            model=model,
            op=op,
            device=device,
        )

    def get_critic_loss(
        self,
        *,
        data_latent: Float[Tensor, "batch ..."],
    ) -> CriticLossConfig.Loss:
        return self.config.critic.apply(
            state=self,
            data_latent=data_latent,
        )

    def get_transport_loss(
        self,
        *,
        data_sample: Float[Tensor, "batch ..."],
        data_latent: Float[Tensor, "batch ..."],
    ) -> DeterministicChartTransportLossConfig.Loss:
        return self.config.transport.apply(
            self,
            data_sample=data_sample,
            data_latent=data_latent,
        )

    def compute_chart_pretrain_loss(
        self,
        *,
        data: Float[Tensor, "batch ..."],
    ) -> ChartPretrainConfig.Loss:
        batch_size = data.shape[0]
        prior = self.prior_config.sample(batch_size=batch_size).type_as(data)
        data_latent = self.encode(data=data)
        model_sample = self.decode(latent=prior)
        return self.config.pretrain.apply(
            self,
            data=data,
            data_latent=data_latent,
            model_sample=model_sample,
            prior=prior,
        )

    def compute_critic_only_loss(
        self,
        *,
        data: Float[Tensor, "batch ..."],
    ) -> CriticLossConfig.Loss:
        with torch.no_grad():
            data_latent = self.encode(data=data)
        return self.get_critic_loss(data_latent=data_latent)

    @dataclass
    class IntegratedLoss(BaseLoss):
        constraint_loss: IntegratedChartConstraintConfig.Loss
        critic_loss: CriticLossConfig.Loss
        transport_loss: DeterministicChartTransportLossConfig.Loss

        def sum(self):
            return (
                self.constraint_loss.sum()
                + self.critic_loss.sum()
                + self.transport_loss.sum()
            )

    def compute_integrated_loss(
        self,
        *,
        data: Float[Tensor, "batch ..."],
        compute_transport_loss: bool,
    ) -> IntegratedLoss:
        batch_size = data.shape[0]
        prior = self.prior_config.sample(batch_size=batch_size).type_as(data)
        data_latent = self.encode(data=data)
        model_sample = self.decode(latent=prior)

        constraint_loss = self.config.integrated_constraint.apply(
            self,
            data=data,
            data_latent=data_latent,
            model_sample=model_sample,
            prior=prior,
        )
        critic_loss = self.get_critic_loss(data_latent=data_latent.detach())

        transport_loss = self.config.transport.zero(
            device=data.device,
            dtype=data.dtype,
        )
        if compute_transport_loss:
            transport_loss = self.get_transport_loss(
                data_sample=data,
                data_latent=data_latent,
            )
        return self.IntegratedLoss(
            constraint_loss=constraint_loss,
            critic_loss=critic_loss,
            transport_loss=transport_loss,
        )

    def encode(
        self,
        *,
        data: Float[Tensor, "batch ..."],
    ) -> Float[Tensor, "batch ..."]:
        return self.model.encoder(data)

    def decode(
        self,
        latent: Float[Tensor, "batch ..."],
    ) -> Float[Tensor, "batch ..."]:
        return self.model.decoder(latent)

    def roundtrip(
        self,
        latent: Float[Tensor, "batch ..."],
    ) -> Float[Tensor, "batch ..."]:
        return self.encode(data=self.decode(latent))

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


__all__ = [
    "DeterministicChartTransportStudyConfig",
    "DeterministicChartTransportStudyState",
]
