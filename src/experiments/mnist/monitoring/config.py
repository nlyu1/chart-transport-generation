from __future__ import annotations

from src.config.base import BaseConfig
from src.experiments.mnist.monitoring.constraint import MNISTConstraintMonitorConfig
from src.monitoring.configs import CriticMonitorConfig, MonitorScheduleConfig


class MNISTMonitorConfig(BaseConfig):
    use_wandb: bool
    constraint_monitor_config: MNISTConstraintMonitorConfig
    critic_monitor_config: CriticMonitorConfig
    schedule_config: MonitorScheduleConfig

__all__ = ["MNISTMonitorConfig"]
