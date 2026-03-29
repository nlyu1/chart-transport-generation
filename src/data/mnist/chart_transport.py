from __future__ import annotations

from pathlib import Path
from typing import Self

from src.chart_transport.aux_loss import ChartPretrainConfig, CriticLossConfig
from src.chart_transport.base import ChartTransportConfig, ChartTransportLossConfig
from src.chart_transport.constraint import (
    LagrangianConstraintConfig,
    ManifoldConstraintConfig,
)
from src.chart_transport.field import UniformVelocityMatchingSchedule
from src.chart_transport.model import ChartTransportModelConfig
from src.chart_transport.scheduling import ChartTransportSchedulingConfig
from src.chart_transport.transport_loss import TransportLossConfig
from src.common.training import TrainingConfig
from src.config.base import BaseConfig
from src.data.mnist.data import MNISTDataConfig
from src.model.mlp import StackedResidualMLPConfig
from src.model.time_conditioning import TimeConditioningConfig
from src.priors.anchored import AnchoredGaussianScaleMixturePriorConfig


def residual_mlp_layer_dims(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_hidden_layers: int,
) -> list[int]:
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    return [
        input_dim,
        *([hidden_dim] * num_hidden_layers),
        output_dim,
    ]


class MNISTChartTransportTrainingConfig(TrainingConfig):
    integrated_n_steps: int

    @classmethod
    def initialize(
        cls,
        *,
        seed: int,
        train_batch_size: int,
        eval_batch_size: int,
        integrated_n_steps: int,
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
            integrated_n_steps=integrated_n_steps,
            folder=folder,
            raise_on_existing_folder=raise_on_existing_folder,
        )


class ConstraintMonitorConfig(BaseConfig):
    reconstruction_examples_per_class: int
    latent_examples_per_class: int


class CriticMonitorConfig(BaseConfig):
    sample_t_values: list[float]
    num_contour_lines: int
    dense_latent_examples_per_class: int
    vector_latent_examples_per_class: int


class IntegratedMonitorConfig(BaseConfig):
    generated_grid_rows: int
    generated_grid_cols: int


class MonitorConfig(BaseConfig):
    constraint_monitor_config: ConstraintMonitorConfig
    critic_monitor_config: CriticMonitorConfig
    integrated_monitor_config: IntegratedMonitorConfig
    log_every_n_steps_constraint_pretrain: int
    log_every_n_steps_critic_pretrain: int
    log_every_n_steps_integrated: int


def get_canonical_chart_transport_config(
    *,
    data_config: MNISTDataConfig,
    latent_dimension: int,
    num_hidden_layers: int,
) -> ChartTransportConfig:
    hidden_dimension = 1024
    time_embedding_dimension = 1024
    prior_precision = 3.0

    prior_config = AnchoredGaussianScaleMixturePriorConfig.initialize(
        latent_shape=[latent_dimension],
        precision=prior_precision,
    )

    constraint_method = LagrangianConstraintConfig(
        data_constraint_budget_per_dim=1e-2 / (data_config.data_numel() ** 0.5),
        prior_constraint_budget_per_dim=1e-2 / (latent_dimension**0.5),
        dual_variable_lr=5e-3,
    )
    constraint_config = ManifoldConstraintConfig(
        huber_delta=2.0,
        constraint_method=constraint_method,
    )
    chart_pretrain_config = ChartPretrainConfig(
        zero_mean_weight=1e-2,
        latent_norm_weight=1e-2,
        latent_norm_delta=1.5 * prior_precision,
    )
    transport_config = TransportLossConfig(
        kl_weight_schedule=UniformVelocityMatchingSchedule(),
        transport_step_size=0.1,
        transport_step_cap=0.05,
        num_time_samples=8,
        t_range=(0.03, 0.95),
        antipodal_estimate=True,
        decoder_transport_weight=1.0,
        encoder_transport_weight=1.0,
    )
    critic_config = CriticLossConfig(
        huber_delta=2.0,
    )
    loss_config = ChartTransportLossConfig(
        constraint_config=constraint_config,
        chart_pretrain_config=chart_pretrain_config,
        transport_config=transport_config,
        critic_config=critic_config,
    )
    scheduling_config = ChartTransportSchedulingConfig(
        pretrain_chart_n_steps=2_000,
        pretrain_critic_n_steps=2_000,
        n_critic_updates_every_transport_step=3,
    )

    critic_time_conditioning_config = TimeConditioningConfig(
        min_t_lambda=0.03,
        max_t_lambda=0.97,
        sinusoidal_dim=time_embedding_dimension,
        hidden_dim=hidden_dimension,
        output_dim=time_embedding_dimension,
    )
    architecture_config = ChartTransportModelConfig(
        encoder=StackedResidualMLPConfig.initialize(
            layer_dims=residual_mlp_layer_dims(
                input_dim=data_config.data_numel(),
                output_dim=latent_dimension,
                hidden_dim=hidden_dimension,
                num_hidden_layers=num_hidden_layers,
            ),
        ),
        decoder=StackedResidualMLPConfig.initialize(
            layer_dims=residual_mlp_layer_dims(
                input_dim=latent_dimension,
                output_dim=data_config.data_numel(),
                hidden_dim=hidden_dimension,
                num_hidden_layers=num_hidden_layers,
            ),
        ),
        critic=StackedResidualMLPConfig.initialize(
            layer_dims=residual_mlp_layer_dims(
                input_dim=latent_dimension,
                output_dim=latent_dimension,
                hidden_dim=hidden_dimension,
                num_hidden_layers=num_hidden_layers,
            ),
            time_conditioning_config=critic_time_conditioning_config,
        ),
        chart_lr=3e-4,
        critic_lr=3e-4,
        grad_clip_norm=1.0,
    )

    return ChartTransportConfig(
        data_config=data_config,
        prior_config=prior_config,
        loss_config=loss_config,
        scheduling_config=scheduling_config,
        architecture_config=architecture_config,
    )


__all__ = [
    "ConstraintMonitorConfig",
    "CriticMonitorConfig",
    "IntegratedMonitorConfig",
    "MNISTChartTransportTrainingConfig",
    "MonitorConfig",
    "get_canonical_chart_transport_config",
    "residual_mlp_layer_dims",
]
