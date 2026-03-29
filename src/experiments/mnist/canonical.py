from __future__ import annotations

from src.chart_transport.base import ChartTransportConfig
from src.data.mnist.chart_transport import get_canonical_chart_transport_config
from src.data.mnist.data import MNISTDataConfig
from src.experiments.mnist.monitoring.config import MNISTMonitorConfig
from src.experiments.mnist.monitoring.constraint import MNISTConstraintMonitorConfig
from src.monitoring.configs import CriticMonitorConfig, MonitorScheduleConfig


def get_canonical_mnist_chart_transport_config(
    *,
    data_config: MNISTDataConfig,
    latent_dimension: int,
    num_hidden_layers: int,
) -> ChartTransportConfig:
    return get_canonical_chart_transport_config(
        data_config=data_config,
        latent_dimension=latent_dimension,
        num_hidden_layers=num_hidden_layers,
    )


def get_canonical_mnist_monitor_config() -> MNISTMonitorConfig:
    return MNISTMonitorConfig(
        use_wandb=True,
        constraint_monitor_config=MNISTConstraintMonitorConfig(
            activate_on_steps=[],
            n_sample_pairs_per_mode=2,
            n_data_latents_per_mode=256,
            planar=True,
        ),
        critic_monitor_config=CriticMonitorConfig(
            activate_on_steps=[],
            sample_t_values=[0.03, 0.20, 0.50],
            num_contour_lines=10,
            n_data_latents_per_mode=256,
            n_vectors_per_mode=16,
            planar=True,
            transport_grid_resolution=31,
            transport_num_time_samples=19,
        ),
        schedule_config=MonitorScheduleConfig(
            activate_on_steps=[],
            log_every_n_steps_chart_pretrain=1_000,
            log_every_n_steps_critic_pretrain=1_000,
            log_every_n_steps_integrated=5_000,
        ),
    )


__all__ = [
    "get_canonical_mnist_chart_transport_config",
    "get_canonical_mnist_monitor_config",
]
