from __future__ import annotations

from src.experiments.multimodal_gaussian.monitoring.constraint import (
    GaussianConstraintMonitorConfig,
)
from src.experiments.multimodal_gaussian.monitoring.critic import (
    GaussianCriticMonitorConfig,
)
from src.monitoring.configs import (
    BaseMonitorConfig,
    ConditioningMonitorConfig,
    MonitorScheduleConfig,
    SamplingMonitorConfig,
)


class MonitorConfig(BaseMonitorConfig):
    constraint_monitor_config: GaussianConstraintMonitorConfig
    critic_monitor_config: GaussianCriticMonitorConfig
    sampling_monitor_config: SamplingMonitorConfig
    conditioning_monitor_config: ConditioningMonitorConfig
    schedule_config: MonitorScheduleConfig
    use_wandb: bool


__all__ = ["MonitorConfig"]
