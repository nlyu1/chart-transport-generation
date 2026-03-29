from __future__ import annotations

from typing import Literal, Self

import torch
import torch.nn as nn
import wandb
from lightning import Fabric
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torch import Tensor, optim
from wandb.sdk.wandb_run import Run

from src.data.mnist.data import MNISTDataConfig
from src.experiments.mnist.chart_pretrain import (
    chart_pretrain_,
    chart_pretrain_eval_step_,
    chart_pretrain_train_step_,
)
from src.experiments.mnist.config import MNISTTrainingConfig
from src.experiments.mnist.critic_pretrain import (
    critic_pretrain_,
    critic_pretrain_eval_step_,
    critic_pretrain_train_step_,
)


@dataclass(
    config=ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    ),
    kw_only=True,
)
class MNISTTrainingRuntime:
    tc: MNISTTrainingConfig
    runtime_data_config: MNISTDataConfig

    cuda_device: int
    fabric: Fabric
    device: torch.device
    precision: Literal["bf16-mixed"]
    wandb_project: str
    run_name: str
    wandb_run: Run

    chart_transport_model: nn.Module
    optimizer: optim.Optimizer

    fixed_reconstruction_samples: Tensor
    fixed_reconstruction_labels: Tensor
    fixed_latent_samples: Tensor
    fixed_latent_labels: Tensor

    def _chart_pretrain_train_step(
        self,
    ) -> dict[str, float]:
        return chart_pretrain_train_step_(rt=self)

    def _chart_pretrain_eval_step(
        self,
        *,
        step: int,
    ) -> dict[str, float]:
        return chart_pretrain_eval_step_(
            rt=self,
            step=step,
        )

    def chart_pretrain(
        self,
    ) -> dict[str, float]:
        return chart_pretrain_(rt=self)

    def _critic_pretrain_train_step(
        self,
    ) -> dict[str, float]:
        return critic_pretrain_train_step_(rt=self)

    def _critic_pretrain_eval_step(
        self,
        *,
        step: int,
    ) -> dict[str, float]:
        return critic_pretrain_eval_step_(
            rt=self,
            step=step,
        )

    def critic_pretrain(
        self,
    ) -> dict[str, float]:
        return critic_pretrain_(rt=self)

    @classmethod
    def initialize(
        cls,
        *,
        tc: MNISTTrainingConfig,
        cuda_device: int,
        wandb_project: str,
        run_name: str,
    ) -> Self:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for MNISTTrainingRuntime")

        num_cuda_devices = torch.cuda.device_count()
        if cuda_device < 0 or cuda_device >= num_cuda_devices:
            raise ValueError(
                f"cuda_device must be in [0, {num_cuda_devices}), got {cuda_device}"
            )

        data_config = tc.chart_transport_config.data_config
        if not isinstance(data_config, MNISTDataConfig):
            raise TypeError("MNISTTrainingRuntime requires MNISTDataConfig")

        precision: Literal["bf16-mixed"] = "bf16-mixed"
        torch.set_float32_matmul_precision("medium")
        fabric = Fabric(
            accelerator="cuda",
            devices=[cuda_device],
            precision=precision,
        )
        fabric.launch()
        fabric.seed_everything(tc.seed)

        device = fabric.device
        runtime_data_config = data_config.to(device=device)

        chart_transport_model = tc.chart_transport_config.architecture_config.get_model()
        optimizer = tc.chart_transport_config.architecture_config.get_optimizer(
            model=chart_transport_model,
        )
        chart_transport_model, optimizer = fabric.setup(
            chart_transport_model,
            optimizer,
        )

        monitor_config = tc.monitor_config.constraint_monitor_config
        fixed_reconstruction_samples, fixed_reconstruction_labels = (
            runtime_data_config.stratified_class_batch(
                batch_size_per_class=monitor_config.n_sample_pairs_per_mode,
                start_index=0,
            )
        )
        fixed_latent_samples, fixed_latent_labels = runtime_data_config.stratified_class_batch(
            batch_size_per_class=monitor_config.n_data_latents_per_mode,
            start_index=monitor_config.n_sample_pairs_per_mode,
        )

        wandb_mode: Literal["online", "disabled"]
        if tc.monitor_config.use_wandb:
            wandb_mode = "online"
        else:
            wandb_mode = "disabled"
        wandb_run = wandb.init(
            project=wandb_project,
            name=run_name,
            dir=str(tc.folder),
            mode=wandb_mode,
            tags=["chart-transport", "mnist"],
        )
        for stage_name in ("pretrain", "critic_pretrain"):
            wandb.define_metric(f"{stage_name}/local_step")
            wandb.define_metric(
                f"{stage_name}/*",
                step_metric=f"{stage_name}/local_step",
            )

        return cls(
            tc=tc,
            runtime_data_config=runtime_data_config,
            cuda_device=cuda_device,
            fabric=fabric,
            device=device,
            precision=precision,
            wandb_project=wandb_project,
            run_name=run_name,
            wandb_run=wandb_run,
            chart_transport_model=chart_transport_model,
            optimizer=optimizer,
            fixed_reconstruction_samples=fixed_reconstruction_samples,
            fixed_reconstruction_labels=fixed_reconstruction_labels,
            fixed_latent_samples=fixed_latent_samples,
            fixed_latent_labels=fixed_latent_labels,
        )


__all__ = ["MNISTTrainingRuntime"]
