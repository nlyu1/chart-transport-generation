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

from src.data.gaussian_mixture.data import MultimodalGaussianDataConfig
from src.experiments.multimodal_gaussian.config import (
    MultimodalGaussianTrainingConfig,
)
from src.experiments.multimodal_gaussian.chart_pretrain import (
    chart_pretrain_,
    chart_pretrain_eval_step_,
    chart_pretrain_train_step_,
)
from src.experiments.multimodal_gaussian.critic_pretrain import (
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
class MultimodalTrainingRuntime:
    tc: MultimodalGaussianTrainingConfig
    runtime_data_config: MultimodalGaussianDataConfig

    cuda_device: int
    fabric: Fabric
    device: torch.device
    precision: Literal["bf16-mixed"]
    wandb_project: str
    run_name: str
    wandb_run: Run

    chart_transport_model: nn.Module
    optimizer: optim.Optimizer

    data_dual: Tensor
    prior_dual: Tensor

    def _chart_pretrain_train_step(
        self,
    ) -> dict[str, float]:
        return chart_pretrain_train_step_(rt=self)

    def _chart_pretrain_eval_step(
        self,
        *,
        step: int,
        run_constraint_monitor: bool,
        run_conditioning_monitor: bool,
    ) -> dict[str, float]:
        return chart_pretrain_eval_step_(
            rt=self,
            step=step,
            run_constraint_monitor=run_constraint_monitor,
            run_conditioning_monitor=run_conditioning_monitor,
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
        tc: MultimodalGaussianTrainingConfig,
        cuda_device: int,
        wandb_project: str,
        run_name: str,
    ) -> Self:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for MultimodalTrainingRuntime")

        num_cuda_devices = torch.cuda.device_count()
        if cuda_device < 0 or cuda_device >= num_cuda_devices:
            raise ValueError(
                f"cuda_device must be in [0, {num_cuda_devices}), got {cuda_device}"
            )

        data_config = tc.chart_transport_config.data_config
        if not isinstance(data_config, MultimodalGaussianDataConfig):
            raise TypeError(
                "MultimodalTrainingRuntime requires MultimodalGaussianDataConfig"
            )

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
            tags=["chart-transport", "multimodal-gaussian"],
        )
        for stage_name in ("pretrain", "critic_pretrain", "integrated"):
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
            data_dual=torch.zeros((), device=device),
            prior_dual=torch.zeros((), device=device),
        )


__all__ = ["MultimodalTrainingRuntime"]
