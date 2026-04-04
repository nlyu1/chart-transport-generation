import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

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

    batch_size: int


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

    def get_pretrain_loss(
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

    @property
    def batch_size(self) -> int:
        return self.config.batch_size

    @property
    def fiber_packing(self) -> FiberPacking:
        return self.config.fiber_packing

    @property
    def critic_config(self) -> CriticLossConfig:
        return self.config.critic

    @property
    def prior_config(self) -> BasePriorConfig:
        return self.config.prior
