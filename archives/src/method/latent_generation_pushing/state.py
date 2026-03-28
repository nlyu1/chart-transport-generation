from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from src.data.base import GenerativeBatch
from src.method.latent_generation.state import (
    LatentGenerationState,
    LatentGenerationStepOutput,
)
from src.method.latent_generation_pushing.config import LatentPushGenerationLossConfig
from src.method.latent_generation_pushing.model import LatentPushGenerationModel
from src.utils import attenuate_gradient


class LatentPushGenerationStepOutput(LatentGenerationStepOutput):
    pass


class LatentPushGenerationState(LatentGenerationState):
    loss_config: LatentPushGenerationLossConfig
    decoder_attenuation: float

    def compute_pushed_decoder_drift(
        self,
        *,
        model: LatentPushGenerationModel,
        y: Tensor,
        t: Tensor,
        eps: Tensor,
    ) -> tuple[Tensor, Tensor]:
        y_anchor = y.detach().requires_grad_(True)
        latent_drift = self.compute_score_matching_drift(
            model=model,
            y=y_anchor,
            t=t,
            eps=eps,
        ).detach()
        x_anchor, pushed_drift = model.decode_jvp(
            y=y_anchor.detach(),
            tangent=latent_drift,
            apply_as_frozen=True,
        )
        return x_anchor.detach(), pushed_drift.detach()

    def compute_losses(
        self,
        *,
        model: LatentPushGenerationModel,
        batch: GenerativeBatch,
    ) -> LatentPushGenerationStepOutput:
        x = batch.data()
        batch_size = int(x.shape[0])
        attenuate = lambda tensor: attenuate_gradient(
            tensor,
            attenuation_factor=self.decoder_attenuation,
        )

        y_data = model.encode(x=x)
        y_data_target = y_data.detach()
        x_recon = attenuate(model.decode(y=y_data))

        z = self.draw_prior_latents(
            batch_size=batch_size,
            device=x.device,
            dtype=x.dtype,
        )
        z_cycle = model.encode(
            x=attenuate(model.decode(y=z)),
        )

        t = self.loss_config.sample_time(
            batch_size=batch_size,
            device=x.device,
            dtype=x.dtype,
        )
        alpha = self.loss_config.get_noise_weight(t=t)

        eps_denoise = torch.randn_like(y_data_target)
        y_t = (1.0 - t) * y_data_target + t * eps_denoise
        y_denoised = model.encode(
            x=attenuate(model.decode(y=y_t)),
            t=t,
        )

        x_score_target, score_matching_drift = self.compute_pushed_decoder_drift(
            model=model,
            y=y_data_target,
            t=t,
            eps=torch.randn_like(y_data_target),
        )
        score_matching_loss = F.mse_loss(
            model.decode(y=y_data_target),
            (x_score_target + score_matching_drift).detach(),
        )

        unweighted_loss_terms = {
            "reconstruction": F.mse_loss(x_recon, x),
            "prior_matching": self.compute_prior_matching_loss(y_data=y_data),
            "cycle_prior": F.mse_loss(z_cycle, z),
            "denoising": (alpha * (y_denoised - y_data_target).square()).mean(),
            "score_matching": score_matching_loss,
        }
        weighted_loss_terms = {
            "reconstruction": self.loss_config.reconstruction_weight
            * unweighted_loss_terms["reconstruction"],
            "prior_matching": self.loss_config.prior_matching_weight
            * unweighted_loss_terms["prior_matching"],
            "cycle_prior": self.loss_config.cycle_prior_weight
            * unweighted_loss_terms["cycle_prior"],
            "denoising": self.loss_config.denoising_weight
            * unweighted_loss_terms["denoising"],
            "score_matching": self.loss_config.score_matching_weight
            * unweighted_loss_terms["score_matching"],
        }
        total_loss = sum(weighted_loss_terms.values())

        return LatentPushGenerationStepOutput(
            total_loss=total_loss,
            loss_terms={"total": total_loss, **weighted_loss_terms},
            weighted_loss_terms=weighted_loss_terms,
            unweighted_loss_terms=unweighted_loss_terms,
        )
