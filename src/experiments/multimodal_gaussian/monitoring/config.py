from __future__ import annotations

from src.experiments.multimodal_gaussian.monitoring.conditioning import (
    GaussianConditioningMonitorConfig,
)
from src.experiments.multimodal_gaussian.monitoring.constraint import (
    GaussianConstraintMonitorConfig,
)
from src.monitoring.configs import (
    BaseMonitorConfig,
    CriticMonitorConfig,
    MonitorScheduleConfig,
    SamplingMonitorConfig,
)


class MonitorConfig(BaseMonitorConfig):
    constraint_monitor_config: GaussianConstraintMonitorConfig
    critic_monitor_config: CriticMonitorConfig
    sampling_monitor_config: SamplingMonitorConfig
    conditioning_monitor_config: GaussianConditioningMonitorConfig
    schedule_config: MonitorScheduleConfig
    use_wandb: bool


__all__ = ["MonitorConfig"]
