from dataclasses import dataclass

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from src.common.training import BaseLoss
from src.config.base import BaseConfig
from src.data.base import BaseDataConfig
from src.priors.base import BasePriorConfig
from src.stochastic_chart_transport.critic import CriticLossConfig
from src.stochastic_chart_transport.fibers import FiberPacking
from src.stochastic_chart_transport.model import ChartTransportModelConfig
from src.stochastic_chart_transport.reconstruction import (
    ChartPretrainConfig,
)
from src.stochastic_chart_transport.transport import (
    StochasticChartTransportLossConfig,
)


class StochasticChartTransportStudyConfig(BaseConfig):
    data: BaseDataConfig
    prior: BasePriorConfig

    model: ChartTransportModelConfig
    fiber_packing: FiberPacking
    # Component specifications
    pretrain: ChartPretrainConfig
    critic: CriticLossConfig
    transport: StochasticChartTransportLossConfig


class StochasticChartTransportStudyState(BaseConfig):
    config: StochasticChartTransportStudyConfig

    model: nn.Module
    op: torch.optim.Optimizer

    fiber_packing: FiberPacking

    device: torch.device

    @classmethod
    def initialize(
        cls, *, config: StochasticChartTransportStudyConfig, device: torch.device
    ):
        model = config.model.get_model().to(device)
        op = config.model.get_optimizer(model)

        return cls(
            config=config,
            model=model,
            op=op,
            fiber_packing=config.fiber_packing,
            device=device,
        )

    def get_constraint_loss(
        self, *, data: Float[Tensor, "batch ..."], compute_anchor_loss: bool
    ) -> ChartPretrainConfig.Loss:
        return self.config.pretrain.apply(
            state=self, data=data, compute_anchor_loss=compute_anchor_loss
        )

    def get_critic_loss(
        self,
        *,
        data_latent: Float[Tensor, "batch ..."],
        model_latent: Float[Tensor, "batch ..."],
    ) -> CriticLossConfig.Loss:
        return self.config.critic.apply(
            state=self,
            data_latent=data_latent,
            model_latent=model_latent,
        )

    def get_transport_loss(
        self,
        *,
        data_sample: Float[Tensor, "batch ..."],
        model_sample: Float[Tensor, "batch ..."],
        data_fiber: Float[Tensor, "batch ..."],
        model_fiber: Float[Tensor, "batch ..."],
        data_latent: Float[Tensor, "batch ..."],
        model_latent: Float[Tensor, "batch ..."],
    ) -> StochasticChartTransportLossConfig.Loss:
        return self.config.transport.apply(
            self,
            data_sample=data_sample,
            model_sample=model_sample,
            data_fiber=data_fiber,
            model_fiber=model_fiber,
            data_latent=data_latent,
            model_latent=model_latent,
        )

    def compute_chart_only_loss(
        self, data: Float[Tensor, "batch ..."], compute_anchor_loss: bool
    ) -> ChartPretrainConfig.Loss:
        batch_size = data.shape[0]
        prior = self.prior_config.sample(batch_size=batch_size).type_as(data)

        data_fiber = self.get_fiber(batch_size=batch_size * 2).type_as(data)

        data_latent = self.model.encoder(self.pack_fiber(data, data_fiber))
        model_sample, model_fiber = self.unpack_fiber(self.model.decoder(data_latent))
        return self.get_constraint_loss(
            data=data,
            data_fiber=data_fiber,
            data_latent=data_latent,
            model_sample=model_sample,
            model_fiber=model_fiber,
            prior=prior,
            compute_anchor_loss=compute_anchor_loss,
        )

    def compute_critic_only_loss(
        self, data: Float[Tensor, "batch ..."]
    ) -> CriticLossConfig.Loss:
        batch_size = data.shape[0]
        prior = self.prior_config.sample(batch_size=batch_size).type_as(data)
        with torch.no_grad():
            model_samples, _ = self.unpack_fiber(self.model.decoder(prior))
            combined_fibers = self.get_fiber(batch_size=batch_size * 2).type_as(data)
            data_latent, model_latent = self.model.encoder(
                self.pack_fiber(torch.cat([data, model_samples]), combined_fibers)
            ).chunk(2, dim=0)
        return self.get_critic_loss(data_latent=data_latent, model_latent=model_latent)

    @dataclass
    class IntegratedLoss(BaseLoss):
        chart_loss: ChartPretrainConfig.Loss
        critic_loss: CriticLossConfig.Loss
        transport_loss: StochasticChartTransportLossConfig | None

        def sum(self):
            return (
                self.constraint_loss.sum()
                + self.critic_loss.sum()
                + (self.transport_loss.sum() if self.transport_loss else 0.0)
            )

    def compute_integrated_loss(
        self,
        data: Float[Tensor, "batch ..."],
        compute_transport_loss: bool,
    ) -> IntegratedLoss:
        """
        Always computes chart + critic losses.
        1. Chart + critic + transport
        2. Chart + critic

        If we're only computing transport, we can always cheaply compute
            the other two losses ~for free. If only chart or critic,
            can use optimized versions
        """
        # Model-independent sampling quantities
        prior = self.prior_config.sample(batch_size=batch_size).type_as(data)

        # Sample fibers
        combined_fiber = self.get_fiber(batch_size=state.batch_size * 2).type_as(data)
        data_fiber, model_fiber = combined_fiber.chunk(2, dim=0)

        # Decode to obtain model samples
        decoded_prior = state.model.decoder(prior)
        model_sample, model_decoded_fiber = self.unpack_fiber(decoded_prior)

        # Compute model and data latent
        combined_sample = torch.cat([data, model_sample])
        data_latent, model_latent = state.model.encoder(
            state.fiber_packing.pack(combined_sample, combined_fiber)
        ).chunk(2, dim=0)

        # Compute losses
        chart_loss = state.get_constraint_loss(
            state,
            data=data,
            data_fiber=data_fiber,
            data_latent=data_latent,
            model_sample=model_sample,
            # Note that we're not providing a fresh fiber here!
            model_fiber=model_decoded_fiber,
            prior=prior,
            compute_anchor_loss=False,
        )

        critic_loss = state.get_critic_loss(
            data_latent=data_latent.detach(), model_latent=model_latent.detach()
        )

        transport_loss = None
        if compute_transport_loss:
            transport_loss = state.get_transport_loss(
                state,
                data_sample=data,
                model_sample=model_sample,
                data_latent=data_latent,
                model_latent=model_latent,
            )
        return IntegratedLoss(
            chart_loss=chart_loss,
            critic_loss=critic_loss,
            transport_loss=transport_loss,
        )

    def get_fiber(self, *, batch_size):
        return self.config.fiber_packing.get_fiber

    def pack_fiber(self, *, data, fiber):
        return self.config.fiber_packing.pack(data=data, fiber=fiber)

    def unpack_fiber(self, data_with_fiber):
        return self.config.fiber_packing.unpack(data_with_fiber)

    @property
    def prior_config(self) -> BasePriorConfig:
        return self.config.prior
