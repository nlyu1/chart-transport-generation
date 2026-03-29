from __future__ import annotations

from src.experiments.multimodal_gaussian.monitoring.constraint import (
    GaussianConstraintMonitorConfig,
)
from src.experiments.multimodal_gaussian.monitoring.sampling import (
    GaussianSamplingMonitorConfig,
)
from src.monitoring.configs import (
    BaseMonitorConfig,
    ConditioningMonitorConfig,
    CriticMonitorConfig,
    MonitorScheduleConfig,
)


class MonitorConfig(BaseMonitorConfig):
    constraint_monitor_config: GaussianConstraintMonitorConfig
    critic_monitor_config: CriticMonitorConfig
    sampling_monitor_config: GaussianSamplingMonitorConfig
    conditioning_monitor_config: ConditioningMonitorConfig
    schedule_config: MonitorScheduleConfig
    use_wandb: bool


__all__ = ["MonitorConfig"]
