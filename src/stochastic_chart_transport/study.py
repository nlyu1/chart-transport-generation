from dataclasses import dataclass

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

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
        self,
        *,
        data: Float[Tensor, "batch ..."],
        data_fiber: Float[Tensor, "batch ..."],
        data_latent: Float[Tensor, "batch ..."],
        model_sample: Float[Tensor, "batch ..."],
        model_fiber: Float[Tensor, "batch ..."],
        prior: Float[Tensor, "batch ..."],
        compute_anchor_loss: bool,
    ) -> ChartPretrainConfig.Loss:
        return self.config.pretrain.apply(
            state=self,
            data=data,
            data_fiber=data_fiber,
            data_latent=data_latent,
            model_sample=model_sample,
            model_fiber=model_fiber,
            prior=prior,
            compute_anchor_loss=compute_anchor_loss,
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
        data_latent: Float[Tensor, "batch ..."],
        model_latent: Float[Tensor, "batch ..."],
    ) -> StochasticChartTransportLossConfig.Loss:
        return self.config.transport.apply(
            self,
            data_sample=data_sample,
            model_sample=model_sample,
            data_latent=data_latent,
            model_latent=model_latent,
        )

    def compute_chart_only_loss(
        self, data: Float[Tensor, "batch ..."], compute_anchor_loss: bool
    ) -> ChartPretrainConfig.Loss:
        batch_size = data.shape[0]
        prior = self.prior_config.sample(batch_size=batch_size).type_as(data)

        data_fiber = self.get_fiber(batch_size=batch_size).type_as(data)

        data_latent = self.encode(data=data, fiber=data_fiber)
        model_sample, model_fiber = self.decode(prior)
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
            model_samples, _ = self.decode(prior)
            combined_fibers = self.get_fiber(batch_size=batch_size * 2).type_as(data)
            data_latent, model_latent = self.encode(
                data=torch.cat([data, model_samples], dim=0),
                fiber=combined_fibers,
            ).chunk(2, dim=0)
        return self.get_critic_loss(data_latent=data_latent, model_latent=model_latent)

    @dataclass
    class IntegratedLoss(BaseLoss):
        chart_loss: ChartPretrainConfig.Loss
        critic_loss: CriticLossConfig.Loss
        transport_loss: StochasticChartTransportLossConfig.Loss | None

        def sum(self):
            return (
                self.chart_loss.sum()
                + self.critic_loss.sum()
                + (self.transport_loss.sum() if self.transport_loss else 0.0)
            )

    def compute_integrated_loss(
        self,
        data: Float[Tensor, "batch ..."],
        compute_transport_loss: bool,
    ) -> IntegratedLoss:
        """
        Always computes chart + critic losses, and optionally transport.

        Gradient contract:
        1. `chart_loss` must see the attached decoded prior sample so the
           prior-roundtrip continues to train the decoder through both sample
           and fiber outputs.
        2. `critic_loss` must see detached latents.
        3. `transport_loss.encoder` must *not* backpropagate into the decoder
           through the model branch. We therefore reuse a detached copy of the
           decoded model sample only for the shared latent-estimation path.
        """
        batch_size = data.shape[0]
        # Model-independent sampling quantities
        prior = self.prior_config.sample(batch_size=batch_size).type_as(data)

        # Sample fibers
        combined_fiber = self.get_fiber(batch_size=batch_size * 2).type_as(data)
        data_fiber, _ = combined_fiber.chunk(2, dim=0)

        # Decode once. The attached tensors feed the chart constraint path.
        model_sample, model_decoded_fiber = self.decode(prior)
        # Only the latent-estimation path should ignore decoder gradients from
        # the model branch. Keep `model_sample` itself attached for chart loss.
        model_sample_for_latent = model_sample.detach()

        # Reuse one encoder pass for data and model latents. The model half uses
        # the detached sample copy so transport encoder supervision only trains
        # the encoder, not the decoder.
        combined_sample = torch.cat([data, model_sample_for_latent], dim=0)
        data_latent, model_latent = self.encode(
            data=combined_sample,
            fiber=combined_fiber,
        ).chunk(2, dim=0)

        # Compute losses
        chart_loss = self.get_constraint_loss(
            data=data,
            data_fiber=data_fiber,
            data_latent=data_latent,
            model_sample=model_sample,
            # Note that we're not providing a fresh fiber here!
            model_fiber=model_decoded_fiber,
            prior=prior,
            compute_anchor_loss=False,
        )

        critic_loss = self.get_critic_loss(
            data_latent=data_latent.detach(), model_latent=model_latent.detach()
        )

        transport_loss = None
        if compute_transport_loss:
            transport_loss = self.get_transport_loss(
                data_sample=data,
                model_sample=model_sample,
                data_latent=data_latent,
                model_latent=model_latent,
            )
        return self.IntegratedLoss(
            chart_loss=chart_loss,
            critic_loss=critic_loss,
            transport_loss=transport_loss,
        )

    def get_fiber(self, *, batch_size):
        return self.config.fiber_packing.get_fiber(batch_size=batch_size)

    def encode(
        self,
        *,
        data: Float[Tensor, "batch ..."],
        fiber: Float[Tensor, "batch ..."],
    ) -> Float[Tensor, "batch ..."]:
        """Pack `(data, fiber)` and run the encoder.

        Gradients flow through both `data` and encoder parameters unless the
        caller explicitly detaches or uses `torch.no_grad()`.
        """
        return self.model.encoder(
            self.config.fiber_packing.pack(data=data, fiber=fiber)
        )

    def decode(
        self,
        latent: Float[Tensor, "batch ..."],
    ) -> tuple[Float[Tensor, "batch ..."], Float[Tensor, "batch ..."]]:
        """Run the decoder and unpack `(sample, fiber)`.

        Returned tensors remain attached to the decoder graph unless the caller
        explicitly detaches or uses `torch.no_grad()`.
        """
        return self.config.fiber_packing.unpack(self.model.decoder(latent))

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
