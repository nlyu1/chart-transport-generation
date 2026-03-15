from __future__ import annotations

import torch
import torch.nn.functional as F
from pydantic import ConfigDict
from torch import Tensor

from src.data.base import GenerativeBatch, GenerativeData
from src.method.base import MethodState, MethodStepOutput
from src.method.latent_generation.config import LatentGenerationLossConfig
from src.method.latent_generation.model import LatentGenerationModel
from src.method.utils import (
    apply_batch_scalars,
    gaussian_posterior_mean,
    weighted_mse,
)


class LatentGenerationStepOutput(MethodStepOutput):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    weighted_loss_terms: dict[str, Tensor]
    unweighted_loss_terms: dict[str, Tensor]


class LatentGenerationState(MethodState):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    loss_config: LatentGenerationLossConfig
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
        model: LatentGenerationModel,
        batch: GenerativeBatch[GenerativeData],
    ) -> LatentGenerationStepOutput:
        x = batch.data().x
        batch_size = int(x.shape[0])
        zeros = torch.zeros(
            batch_size,
            1,
            device=x.device,
            dtype=x.dtype,
        )

        y_data = model.encode_plain(x=x)
        y_data_target = y_data.detach()
        x_recon = model.decode(y=y_data)

        y_cycle_data = model.roundtrip(y=y_data_target, t=zeros)
        z = self.draw_prior_latents(
            batch_size=batch_size,
            device=x.device,
            dtype=x.dtype,
        )
        z_cycle = model.roundtrip(y=z, t=zeros)

        t = self.loss_config.sample_time(
            batch_size=batch_size,
            device=x.device,
            dtype=x.dtype,
        )
        alpha = self.loss_config.get_noise_weight(t=t)

        eps_denoise = torch.randn_like(y_data_target)
        y_t = apply_batch_scalars(y_data_target, 1.0 - t) + apply_batch_scalars(
            eps_denoise,
            t,
        )
        y_denoised = model.roundtrip(y=y_t, t=t)

        y_roundtrip = model.roundtrip(y=y_data_target, t=zeros)
        eps_score = torch.randn_like(y_roundtrip)
        y_roundtrip_t = apply_batch_scalars(y_roundtrip, 1.0 - t) + apply_batch_scalars(
            eps_score,
            t,
        )
        y_score = model.roundtrip(y=y_roundtrip_t, t=t)
        y_score_target = gaussian_posterior_mean(
            y_t=y_roundtrip_t.detach(),
            t=t,
        )

        unweighted_loss_terms = {
            "reconstruction": F.mse_loss(x_recon, x),
            "cycle_data": F.mse_loss(y_cycle_data, y_data_target),
            "cycle_prior": F.mse_loss(z_cycle, z),
            "denoising": weighted_mse(
                pred=y_denoised,
                target=y_data_target,
                weight=alpha,
            ),
            "score": weighted_mse(
                pred=y_score,
                target=y_score_target,
                weight=alpha,
            ),
        }
        weighted_loss_terms = {
            "reconstruction": self.loss_config.reconstruction_weight
            * unweighted_loss_terms["reconstruction"],
            "cycle_data": self.loss_config.cycle_data_weight
            * unweighted_loss_terms["cycle_data"],
            "cycle_prior": self.loss_config.cycle_prior_weight
            * unweighted_loss_terms["cycle_prior"],
            "denoising": self.loss_config.denoising_weight
            * unweighted_loss_terms["denoising"],
            "score": self.loss_config.score_weight * unweighted_loss_terms["score"],
        }
        total_loss = sum(weighted_loss_terms.values())

        return LatentGenerationStepOutput(
            total_loss=total_loss,
            loss_terms={"total": total_loss, **weighted_loss_terms},
            weighted_loss_terms=weighted_loss_terms,
            unweighted_loss_terms=unweighted_loss_terms,
        )
