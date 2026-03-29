from __future__ import annotations

from pathlib import Path
from typing import Self

from src.chart_transport.base import ChartTransportConfig
from src.common.training import TrainingConfig
from src.experiments.multimodal_gaussian.monitoring.config import MonitorConfig


class MultimodalGaussianTrainingConfig(TrainingConfig):
    integrated_n_steps: int
    monitor_config: MonitorConfig
    chart_transport_config: ChartTransportConfig

    @classmethod
    def initialize(
        cls,
        *,
        seed: int,
        train_batch_size: int,
        eval_batch_size: int,
        integrated_n_steps: int,
        monitor_config: MonitorConfig,
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
            chart_transport_config=chart_transport_config,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            integrated_n_steps=integrated_n_steps,
            monitor_config=monitor_config,
            folder=folder,
            raise_on_existing_folder=raise_on_existing_folder,
        )


__all__ = ["MultimodalGaussianTrainingConfig"]
