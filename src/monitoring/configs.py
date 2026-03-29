from __future__ import annotations

from typing import TYPE_CHECKING

from jaxtyping import Float
from torch import Tensor

from src.config.base import BaseConfig

if TYPE_CHECKING:
    import torch.nn as nn


class ConstraintMonitorConfig(BaseConfig):
    """
    Generates two subplots:
    1. (data, reconstruction).
        Experiments can decide how to instantiate implementation
    2. Scatterplot of the PCA of data-latent, with

    Experiments can decide how to instantiate implementation
    """

    n_sample_pairs_per_mode: int
    n_data_latents_per_mode: int
    planar: bool

    def save_latent_plot_to(**kwargs) -> None:
        pass


class CriticMonitorConfig(BaseConfig):
    """
    Config for visualizing the score implied by the critic. By one plot, we mean "html + png"
    1. Saves one plot for each sample_t_values, with two arrows attached to each point
        corresponding to the "data score" and the "prior - data score" at the noise level
    2. Saves one plot "transport "
        corresponding to the implied drift field for **clean data latent**,
        averaged across the noise spectrum. Background corresponds to contour lines
    Keep these specs after modification
    """

    sample_t_values: list[float]
    num_contour_lines: int
    n_data_latents_per_mode: int
    n_vectors_per_mode: int


class SamplingMonitorConfig(BaseConfig):
    """
    Integrated training needs to execute constraint-monitor, critic-monitor, and
    save a scatterplot of the generated samples.
    """

    n_generated_samples: int
    n_data_samples_per_mode: int


class ConditioningMonitorConfig(BaseConfig):
    """
    Config for determining the distribution of singular values
    """

    n_data_samples_per_mode: int
    num_power_iterations: int
    microbatch_size: int

    def largest_singular_values(
        self,
        *,
        model: "nn.Module",
        inputs: Float[Tensor, "batch ..."],
    ) -> Float[Tensor, "batch"]:
        from src.monitoring.conditioning import largest_jacobian_singular_values

        return largest_jacobian_singular_values(
            model=model,
            inputs=inputs,
            config=self,
        )


class MonitorScheduleConfig(BaseConfig):
    """
    Chart pretrain executes (conditioning, constraint)
    Critic pretrain executes (critic monitoring)
    Integrated executes (conditioning, constraint, critic monitoring, sampling)
    """

    log_every_n_steps_chart_pretrain: int
    log_every_n_steps_critic_pretrain: int
    log_every_n_steps_integrated: int


class BaseMonitorConfig(BaseConfig):
    constraint_monitor_config: ConstraintMonitorConfig
    critic_monitor_config: CriticMonitorConfig
    sampling_monitor_config: SamplingMonitorConfig
    conditioning_monitor_config: ConditioningMonitorConfig
    schedule_config: MonitorScheduleConfig
