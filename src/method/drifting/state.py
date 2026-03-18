from __future__ import annotations

import torch
import torch.nn.functional as F
from pydantic import ConfigDict
from torch import Tensor

from src.data.base import GenerativeBatch
from src.method.base import MethodState, MethodStepOutput
from src.method.drifting.config import GaussianDriftingLossConfig
from src.method.drifting.kernel import compute_gaussian_drifting_statistics
from src.method.drifting.model import DriftingModel


class DriftingStepOutput(MethodStepOutput):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    weighted_loss_terms: dict[str, Tensor]
    unweighted_loss_terms: dict[str, Tensor]


class DriftingState(MethodState):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    loss_config: GaussianDriftingLossConfig
    latent_shape: tuple[int, ...]

    def draw_prior_latents(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        return torch.randn(
            batch_size,
            *self.latent_shape,
            device=device,
            dtype=dtype,
        )

    def compute_losses(
        self,
        *,
        model: DriftingModel,
        batch: GenerativeBatch,
    ) -> DriftingStepOutput:
        x_data = batch.data()
        batch_size = int(x_data.shape[0])
        z = self.draw_prior_latents(
            batch_size=batch_size,
            device=x_data.device,
            dtype=x_data.dtype,
        )
        x_model = model.decode(y=z)
        statistics = compute_gaussian_drifting_statistics(
            model_samples=x_model.detach(),
            data_samples=x_data.detach(),
            bandwidth=self.loss_config.bandwidth,
            objective=self.loss_config.objective,
            stability_eps=self.loss_config.stability_eps,
            exclude_self_interactions=self.loss_config.exclude_self_interactions,
        )
        x_target = (
            x_model.detach() + self.loss_config.drift_scale * statistics.drift
        ).detach()

        unweighted_loss_terms = {
            "drift_matching": F.mse_loss(x_model, x_target),
            "drift_norm": statistics.drift.norm(dim=-1).mean(),
            "data_density": statistics.data_density.mean(),
            "model_density": statistics.model_density.mean(),
            "density_ratio": statistics.density_ratio.mean(),
        }
        weighted_loss_terms = {
            "drift_matching": unweighted_loss_terms["drift_matching"],
        }
        total_loss = weighted_loss_terms["drift_matching"]

        return DriftingStepOutput(
            total_loss=total_loss,
            loss_terms={
                "total": total_loss,
                "drift_matching": weighted_loss_terms["drift_matching"],
                "drift_norm": unweighted_loss_terms["drift_norm"],
                "data_density": unweighted_loss_terms["data_density"],
                "model_density": unweighted_loss_terms["model_density"],
                "density_ratio": unweighted_loss_terms["density_ratio"],
            },
            weighted_loss_terms=weighted_loss_terms,
            unweighted_loss_terms=unweighted_loss_terms,
        )
