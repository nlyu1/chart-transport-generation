from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.data.mnist.monitoring import write_constraint_monitor_artifacts
from src.monitoring.configs import ConstraintMonitorConfig
from src.monitoring.utils import MonitorStage, step_folder

if TYPE_CHECKING:
    from src.experiments.mnist.state import MNISTTrainingRuntime


class MNISTConstraintMonitorConfig(ConstraintMonitorConfig):
    def apply_to(
        self,
        *,
        rt: "MNISTTrainingRuntime",
        step: int,
        stage: MonitorStage,
    ) -> dict[str, float]:
        with torch.no_grad():
            latents = rt.chart_transport_model.encoder(rt.fixed_reconstruction_samples).float()
            reconstructions = rt.chart_transport_model.decoder(latents).float()
            latent_values = rt.chart_transport_model.encoder(rt.fixed_latent_samples).float()

        reconstruction_error = (
            (reconstructions - rt.fixed_reconstruction_samples)
            .reshape(reconstructions.shape[0], -1)
            .norm(dim=-1)
            .float()
        )
        latent_norms = latent_values.norm(dim=-1).float()

        output_folder = step_folder(
            run_folder=rt.tc.folder,
            stage=stage,
            step=step,
        ) / "constraint"
        write_constraint_monitor_artifacts(
            data_config=rt.runtime_data_config,
            output_folder=output_folder,
            reconstruction_samples=rt.fixed_reconstruction_samples.detach().cpu().float(),
            reconstruction_labels=rt.fixed_reconstruction_labels.detach().cpu().long(),
            reconstructions=reconstructions.detach().cpu().float(),
            latent_values=latent_values.detach().cpu().float(),
            latent_labels=rt.fixed_latent_labels.detach().cpu().long(),
            examples_per_class=self.n_sample_pairs_per_mode,
        )
        return {
            "constraint_reconstruction_mean": reconstruction_error.mean().item(),
            "constraint_reconstruction_max": reconstruction_error.max().item(),
            "constraint_latent_norm_mean": latent_norms.mean().item(),
        }


__all__ = ["MNISTConstraintMonitorConfig"]
