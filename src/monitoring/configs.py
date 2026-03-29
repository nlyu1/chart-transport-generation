from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jaxtyping import Float, Int
from torch import Tensor

from src.config.base import BaseConfig
from src.monitoring.utils import MonitorStage

if TYPE_CHECKING:
    import torch.nn as nn


class BaseMonitorComponentConfig(BaseConfig):
    activate_on_steps: list[int]

    def should_activate(
        self,
        *,
        step: int,
        total_steps: int,
        every_n_steps: int,
    ) -> bool:
        return (
            step in self.activate_on_steps
            or step == 1
            or step == total_steps
            or step % every_n_steps == 0
        )


class ConstraintMonitorConfig(BaseMonitorComponentConfig):
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

    def save_latent_plot_to(
        self,
        *,
        latents: Float[Tensor, "batch latent_dim"],
        mode_ids: Int[Tensor, "batch"],
        save_to_folder: Path,
    ) -> None:
        from src.monitoring.constraints import save_latent_plot_to

        save_latent_plot_to(
            config=self,
            latents=latents,
            mode_ids=mode_ids,
            save_to_folder=save_to_folder,
        )


class CriticMonitorConfig(BaseMonitorComponentConfig):
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
    planar: bool
    transport_grid_resolution: int
    transport_num_time_samples: int

    def apply_to(
        self,
        *,
        rt,
        step: int,
        stage: MonitorStage,
    ) -> dict[str, float]:
        from src.monitoring.critic import apply_critic_monitor

        return apply_critic_monitor(
            config=self,
            rt=rt,
            step=step,
            stage=stage,
        )


class SamplingMonitorConfig(BaseMonitorComponentConfig):
    """
    Integrated training needs to execute constraint-monitor, critic-monitor, and
    save a scatterplot of the generated samples.
    """

    n_generated_samples: int
    n_data_samples_per_mode: int

    def apply_to(
        self,
        *,
        rt,
        step: int,
        stage: MonitorStage,
    ) -> dict[str, float]:
        raise NotImplementedError(
            "Sampling monitors must be implemented by experiment-specific subclasses"
        )


class ConditioningMonitorConfig(BaseMonitorComponentConfig):
    """
    Config for determining the distribution of singular values
    """

    n_data_samples_per_mode: int
    num_power_iterations: int
    microbatch_size: int

    def apply_to(
        self,
        *,
        rt,
        step: int,
        stage: MonitorStage,
    ) -> dict[str, float]:
        from src.monitoring.conditioning import apply_conditioning_monitor

        return apply_conditioning_monitor(
            config=self,
            rt=rt,
            step=step,
            stage=stage,
        )

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


class MonitorScheduleConfig(BaseMonitorComponentConfig):
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

    def every_n_steps_for_stage(
        self,
        *,
        stage: MonitorStage,
    ) -> int:
        if stage is MonitorStage.CHART:
            return self.schedule_config.log_every_n_steps_chart_pretrain
        if stage is MonitorStage.CRITIC:
            return self.schedule_config.log_every_n_steps_critic_pretrain
        if stage is MonitorStage.INTEGRATED:
            return self.schedule_config.log_every_n_steps_integrated
        raise ValueError(f"Unknown monitor stage: {stage}")

    def relevant_component_configs(
        self,
        *,
        stage: MonitorStage,
    ) -> tuple[BaseMonitorComponentConfig, ...]:
        if stage is MonitorStage.CHART:
            return (
                self.constraint_monitor_config,
                self.conditioning_monitor_config,
            )
        if stage is MonitorStage.CRITIC:
            return (self.critic_monitor_config,)
        if stage is MonitorStage.INTEGRATED:
            return (
                self.constraint_monitor_config,
                self.critic_monitor_config,
                self.conditioning_monitor_config,
                self.sampling_monitor_config,
            )
        raise ValueError(f"Unknown monitor stage: {stage}")

    def should_force_stage(
        self,
        *,
        stage: MonitorStage,
        step: int,
        total_steps: int,
    ) -> bool:
        every_n_steps = self.every_n_steps_for_stage(stage=stage)
        return self.schedule_config.should_activate(
            step=step,
            total_steps=total_steps,
            every_n_steps=every_n_steps,
        )

    def should_activate_component(
        self,
        *,
        component_config: BaseMonitorComponentConfig,
        stage: MonitorStage,
        step: int,
        total_steps: int,
        force: bool = False,
    ) -> bool:
        if force:
            return True
        if stage is MonitorStage.CRITIC:
            return False
        every_n_steps = self.every_n_steps_for_stage(stage=stage)
        return component_config.should_activate(
            step=step,
            total_steps=total_steps,
            every_n_steps=every_n_steps,
        )

    def should_run_stage(
        self,
        *,
        stage: MonitorStage,
        step: int,
        total_steps: int,
    ) -> bool:
        force = self.should_force_stage(
            stage=stage,
            step=step,
            total_steps=total_steps,
        )
        if force:
            return True
        if stage is MonitorStage.CRITIC:
            return False
        return any(
            self.should_activate_component(
                component_config=component_config,
                stage=stage,
                step=step,
                total_steps=total_steps,
            )
            for component_config in self.relevant_component_configs(stage=stage)
        )
