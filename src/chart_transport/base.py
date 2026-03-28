from __future__ import annotations

from src.chart_transport.aux_loss import ChartPretrainConfig, CriticLossConfig
from src.chart_transport.constraint import ManifoldConstraintConfig
from src.chart_transport.model import ChartTransportModelConfig
from src.chart_transport.scheduling import ChartTransportSchedulingConfig
from src.chart_transport.transport_loss import TransportLossConfig
from src.config.base import BaseConfig
from src.data.base import BaseDataConfig
from src.priors.base import BasePriorConfig


class ChartTransportLossConfig(BaseConfig):
    """
    Specifies all the loss components:
    1. manifold constraint: dual variable or loss weighting
    2. transport: how to estimate the transport field, and loss weighting
    3. critic: just the weight of the loss
    """

    constraint_config: ManifoldConstraintConfig
    """Constraint for manifold invariants."""

    chart_pretrain_config: ChartPretrainConfig
    """Auxiliary losses used to initialize the chart before transport."""

    transport_config: TransportLossConfig
    """Specification for transport-field estimation."""

    critic_config: CriticLossConfig
    """Specification for the critic loss."""


class ChartTransportConfig(BaseConfig):
    data_config: BaseDataConfig
    prior_config: BasePriorConfig
    loss_config: ChartTransportLossConfig
    scheduling_config: ChartTransportSchedulingConfig
    architecture_config: ChartTransportModelConfig


__all__ = ["ChartTransportLossConfig", "ChartTransportConfig"]
