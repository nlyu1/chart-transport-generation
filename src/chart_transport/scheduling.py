from __future__ import annotations

from src.config.base import BaseConfig


class ChartTransportSchedulingConfig(BaseConfig):
    """
    The exact schedule proceeds as:

    1. Pretrain the chart for some steps
    2. Freezing the chart, train the critic for some steps
    3. Full training
    """

    pretrain_chart_n_steps: int
    pretrain_critic_n_steps: int
    update_chart_every_n_critic_steps: int


__all__ = ["ChartTransportSchedulingConfig"]
