from __future__ import annotations

from pathlib import Path
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
from src.experiments.multimodal_gaussian.integrated import (
    integrated_,
    integrated_constraint_repair_step_,
    integrated_eval_step_,
    integrated_train_step_,
    integrated_transport_step_,
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

    def _integrated_constraint_repair_step(
        self,
    ) -> dict[str, float]:
        return integrated_constraint_repair_step_(rt=self)

    def _integrated_transport_step(
        self,
    ) -> dict[str, float]:
        return integrated_transport_step_(rt=self)

    def _integrated_train_step(
        self,
    ) -> dict[str, float]:
        return integrated_train_step_(rt=self)

    def _integrated_eval_step(
        self,
        *,
        step: int,
        run_constraint_monitor: bool,
        run_critic_monitor: bool,
        run_conditioning_monitor: bool,
        run_sampling_monitor: bool,
    ) -> dict[str, float]:
        return integrated_eval_step_(
            rt=self,
            step=step,
            run_constraint_monitor=run_constraint_monitor,
            run_critic_monitor=run_critic_monitor,
            run_conditioning_monitor=run_conditioning_monitor,
            run_sampling_monitor=run_sampling_monitor,
        )

    def integrated(
        self,
    ) -> dict[str, float]:
        return integrated_(rt=self)

    def _runtime_artifact_directory(
        self,
    ) -> Path:
        return self.tc.folder / "runtime"

    def _model_artifact_path(
        self,
        *,
        stage: Literal["chart_pretrain", "critic_pretrain"],
    ) -> Path:
        return self._runtime_artifact_directory() / f"{stage}_model.pt"

    def _model_state_dict_payload(
        self,
    ) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "chart_transport_model_state_dict": {
                key: value.detach().cpu()
                for key, value in self.chart_transport_model.state_dict().items()
            }
        }

    def save_model_artifact(
        self,
        *,
        stage: Literal["chart_pretrain", "critic_pretrain"],
    ) -> Path:
        model_artifact_path = self._model_artifact_path(stage=stage)
        model_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self._model_state_dict_payload(),
            model_artifact_path,
        )
        return model_artifact_path

    def save_chart_pretrain_model(
        self,
    ) -> Path:
        return self.save_model_artifact(stage="chart_pretrain")

    def save_critic_pretrain_model(
        self,
    ) -> Path:
        return self.save_model_artifact(stage="critic_pretrain")

    def load_model_artifact(
        self,
        *,
        stage: Literal["chart_pretrain", "critic_pretrain"],
    ) -> Path:
        model_artifact_path = self._model_artifact_path(stage=stage)
        if not model_artifact_path.exists():
            raise FileNotFoundError(
                f"Missing model artifact for {stage}: {model_artifact_path}"
            )
        payload = torch.load(
            model_artifact_path,
            map_location="cpu",
        )
        self.chart_transport_model.load_state_dict(
            payload["chart_transport_model_state_dict"],
        )
        return model_artifact_path

    def load_latest_pretrain_model(
        self,
    ) -> Path:
        critic_pretrain_model_path = self._model_artifact_path(stage="critic_pretrain")
        if critic_pretrain_model_path.exists():
            return self.load_model_artifact(stage="critic_pretrain")

        chart_pretrain_model_path = self._model_artifact_path(stage="chart_pretrain")
        if chart_pretrain_model_path.exists():
            return self.load_model_artifact(stage="chart_pretrain")

        raise FileNotFoundError(
            "No saved pretrain model found under "
            f"{self._runtime_artifact_directory()}"
        )

    def finish(
        self,
    ) -> None:
        self.wandb_run.finish()

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
