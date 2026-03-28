from __future__ import annotations

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
from src.data.gaussian_mixture.data import MultimodalGaussianDataConfig
from src.model.mlp import StackedResidualMLPConfig
from src.model.time_conditioning import TimeConditioningConfig
from src.priors.anchored import AnchoredGaussianScaleMixturePriorConfig


def get_canonical_chart_transport_configs(
    *,
    data_config: MultimodalGaussianDataConfig,
) -> ChartTransportConfig:
    latent_dimension = 2
    prior_precision = 3.0
    hidden_dimension = 256
    time_embedding_dimension = 256

    prior_config = AnchoredGaussianScaleMixturePriorConfig.initialize(
        latent_shape=[latent_dimension],
        precision=prior_precision,
    )

    constraint_method = LagrangianConstraintConfig(
        data_constraint_budget=0.01,
        prior_constraint_budget=0.01,
        dual_variable_lr=5e-3,
    )
    constraint_config = ManifoldConstraintConfig(
        huber_delta=2.0,
        constraint_method=constraint_method,
    )
    chart_pretrain_config = ChartPretrainConfig(
        zero_mean_weight=1e-2,
        softplus_weight=1e-2,
        softplus_radius=1.5 * prior_precision,
    )
    transport_config = TransportLossConfig(
        kl_weight_schedule=UniformVelocityMatchingSchedule(),
        transport_step_size=0.1,
        num_time_samples=8,
        t_range=(0.03, 0.95),
        antipodal_estimate=True,
        decoder_transport_weight=1.0,
        encoder_transport_weight=1.0,
        huber_delta=2.0,
    )
    critic_config = CriticLossConfig(
        loss_weight=1.0,
        huber_delta=2.0,
    )
    loss_config = ChartTransportLossConfig(
        constraint_config=constraint_config,
        chart_pretrain_config=chart_pretrain_config,
        transport_config=transport_config,
        critic_config=critic_config,
    )
    scheduling_config = ChartTransportSchedulingConfig(
        pretrain_chart_n_steps=1000,
        pretrain_critic_n_steps=1000,
        update_chart_every_n_critic_steps=1,
    )

    critic_time_conditioning_config = TimeConditioningConfig(
        min_t_lambda=0.03,
        max_t_lambda=0.99,
        sinusoidal_dim=time_embedding_dimension,
        hidden_dim=hidden_dimension,
        output_dim=time_embedding_dimension,
    )
    architecture_config = ChartTransportModelConfig(
        encoder=StackedResidualMLPConfig.initialize(
            layer_dims=[
                data_config.data_numel(),
                hidden_dimension,
                hidden_dimension,
                latent_dimension,
            ],
        ),
        decoder=StackedResidualMLPConfig.initialize(
            layer_dims=[
                latent_dimension,
                hidden_dimension,
                hidden_dimension,
                data_config.data_numel(),
            ],
        ),
        critic=StackedResidualMLPConfig.initialize(
            layer_dims=[
                latent_dimension,
                hidden_dimension,
                hidden_dimension,
                latent_dimension,
            ],
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


__all__ = ["get_canonical_chart_transport_configs"]
