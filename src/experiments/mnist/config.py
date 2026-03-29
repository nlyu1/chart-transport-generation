from __future__ import annotations

from pathlib import Path
from typing import Self

from src.chart_transport.base import ChartTransportConfig
from src.common.training import TrainingConfig
from src.experiments.mnist.monitoring.config import MNISTMonitorConfig


class MNISTTrainingConfig(TrainingConfig):
    monitor_config: MNISTMonitorConfig
    chart_transport_config: ChartTransportConfig

    @classmethod
    def initialize(
        cls,
        *,
        seed: int,
        train_batch_size: int,
        eval_batch_size: int,
        monitor_config: MNISTMonitorConfig,
        chart_transport_config: ChartTransportConfig,
        folder: Path,
        raise_on_existing_folder: bool = True,
    ) -> Self:
        cls.prepare_folder(
            folder=folder,
            raise_on_existing_folder=raise_on_existing_folder,
        )
        return cls(
            seed=seed,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            monitor_config=monitor_config,
            chart_transport_config=chart_transport_config,
            folder=folder,
            raise_on_existing_folder=raise_on_existing_folder,
        )


__all__ = ["MNISTTrainingConfig"]
