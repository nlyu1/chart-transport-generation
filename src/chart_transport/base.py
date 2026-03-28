from __future__ import annotations

from src.chart_transport.loss import ChartTransportLossConfig
from src.chart_transport.model import ChartTransportModelConfig
from src.config.base import BaseConfig
from src.data.base import BaseDataConfig
from src.priors.base import BasePriorConfig


class ChartTransportConfig(BaseConfig):
    data_config: BaseDataConfig
    prior_config: BasePriorConfig
    loss_config: ChartTransportLossConfig
    architecture_config: ChartTransportModelConfig


__all__ = ["ChartTransportConfig"]
