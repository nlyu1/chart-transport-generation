from __future__ import annotations

from src.chart_transport.aux_loss import ChartPretrainConfig, CriticLossConfig
from src.chart_transport.base import ChartTransportConfig, ChartTransportLossConfig
from src.chart_transport.constraint import (
    LossConstraintConfig,
    ManifoldConstraintConfig,
)
from src.chart_transport.field import UniformVelocityMatchingSchedule
from src.chart_transport.model import ChartTransportModelConfig
from src.chart_transport.scheduling import ChartTransportSchedulingConfig
from src.chart_transport.transport_loss import TransportLossConfig
from src.data.gaussian_mixture.data import MultimodalGaussianDataConfig
from src.experiments.multimodal_gaussian.monitoring.config import MonitorConfig
from src.experiments.multimodal_gaussian.monitoring.constraint import (
    GaussianConstraintMonitorConfig,
)
from src.experiments.multimodal_gaussian.monitoring.sampling import (
    GaussianSamplingMonitorConfig,
)
from src.model.mlp import StackedResidualMLPConfig
from src.model.time_conditioning import TimeConditioningConfig
from src.monitoring.configs import (
    ConditioningMonitorConfig,
    CriticMonitorConfig,
    MonitorScheduleConfig,
)
from src.priors.anchored import AnchoredGaussianScaleMixturePriorConfig

MIN_T = 0.01
PLANAR = True


def get_canonical_chart_transport_configs(
    *,
    latent_dimension: int,
    data_config: MultimodalGaussianDataConfig,
) -> ChartTransportConfig:
    prior_precision = 2.0
    hidden_dimension = 512
    time_embedding_dimension = 512

    prior_config = AnchoredGaussianScaleMixturePriorConfig.initialize(
        latent_shape=[latent_dimension],
        precision=prior_precision,
    )

    """Defines the Huber loss that governs pretrain"""
    chart_pretrain_config = ChartPretrainConfig(
        zero_mean_weight=0.0,
        latent_norm_weight=1e-4,
        latent_norm_delta=1.5 * prior_precision,
    )

    """Defines how the manifold constraints are enforced during integration training"""
    # constraint_method = LagrangianConstraintConfig(
    #     data_constraint_budget_per_dim=1e-4,
    #     prior_constraint_budget_per_dim=1e-4,
    #     dual_variable_lr=5e-3,
    # )
    constraint_method = LossConstraintConfig(
        data_loss_weight=1.0,
        prior_loss_weight=1.0,
    )
    constraint_config = ManifoldConstraintConfig(
        huber_delta=5.0,
        constraint_method=constraint_method,
    )

    """Defining the transport field & approximations"""
    transport_config = TransportLossConfig(
        kl_weight_schedule=UniformVelocityMatchingSchedule(),
        transport_step_size=0.1,
        transport_step_cap=0.1,
        num_time_samples=8,
        t_range=(MIN_T, 0.99),
        antipodal_estimate=True,
        decoder_transport_weight=1.0,
        encoder_transport_weight=1.0,
    )
    critic_config = CriticLossConfig(
        huber_delta=5.0,
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
        n_critic_updates_every_transport_step=2,
    )

    critic_time_conditioning_config = TimeConditioningConfig(
        min_t_lambda=MIN_T,
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


def get_canonical_chart_transport_monitor_configs() -> MonitorConfig:
    return MonitorConfig(
        use_wandb=True,
        constraint_monitor_config=GaussianConstraintMonitorConfig(
            activate_on_steps=[],
            n_sample_pairs_per_mode=500,
            n_data_latents_per_mode=500,
            planar=PLANAR,
        ),
        critic_monitor_config=CriticMonitorConfig(
            activate_on_steps=[],  # 1, 2, 5, 10, 20, 100],
            sample_t_values=[MIN_T, 0.2],
            num_contour_lines=10,
            n_data_latents_per_mode=500,
            n_vectors_per_mode=100,
            planar=PLANAR,
            transport_grid_resolution=31,
            transport_num_time_samples=19,
        ),
        sampling_monitor_config=GaussianSamplingMonitorConfig(
            activate_on_steps=[],
            n_generated_samples=3000,
            n_data_samples_per_mode=1000,
            kde_scales=[1e-3, 3e-3, 1e-2, 3e-2],
            kl_num_samples=512,
            avg_kl_num_batches=8,
        ),
        conditioning_monitor_config=ConditioningMonitorConfig(
            activate_on_steps=[],
            n_data_samples_per_mode=500,
            num_power_iterations=32,
            microbatch_size=128,
        ),
        schedule_config=MonitorScheduleConfig(
            activate_on_steps=[],
            log_every_n_steps_chart_pretrain=1000,
            log_every_n_steps_critic_pretrain=1000,
            log_every_n_steps_integrated=1000,
        ),
    )


__all__ = [
    "get_canonical_chart_transport_configs",
    "get_canonical_chart_transport_monitor_configs",
]
