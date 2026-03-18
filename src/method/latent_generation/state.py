from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Float
from pydantic import ConfigDict
from torch import Tensor

from src.data.base import GenerativeBatch
from src.method.base import MethodState, MethodStepOutput
from src.method.latent_generation.config import LatentGenerationLossConfig
from src.method.latent_generation.model import LatentGenerationModel


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

    @staticmethod
    def compute_prior_matching_loss(*, y_data: Tensor) -> Tensor:
        latent_dim = int(y_data.shape[-1])
        mean = y_data.mean(dim=0)
        centered = y_data - mean
        cov = centered.transpose(0, 1) @ centered
        cov = cov / max(int(y_data.shape[0]) - 1, 1)
        eye = torch.eye(
            latent_dim,
            device=y_data.device,
            dtype=y_data.dtype,
        )
        return mean.square().mean() + (cov - eye).square().mean()

    @staticmethod
    def compute_analytic_gaussian_denoiser(
        *,
        y_t: Float[Tensor, "batch latent_dim"],
        t: Float[Tensor, "batch 1"],
    ) -> Float[Tensor, "batch latent_dim"]:
        denoising_coef = (1.0 - t) / ((1.0 - t).square() + t.square())
        return denoising_coef * y_t

    def compute_score_mismatch_potential(
        self,
        *,
        model: LatentGenerationModel,
        y: Float[Tensor, "batch latent_dim"],
        t: Float[Tensor, "batch 1"],
        eps: Float[Tensor, "batch latent_dim"],
    ) -> Float[Tensor, "batch"]:
        y_t = (1.0 - t) * y + t * eps
        y_denoised = model.roundtrip(
            y=y_t,
            t=t,
            apply_as_frozen=True,
        )
        y_target = self.compute_analytic_gaussian_denoiser(
            y_t=y_t,
            t=t,
        )
        alpha = self.loss_config.get_noise_weight(t=t)
        return (
            alpha * (y_denoised - y_target).square().mean(dim=-1, keepdim=True)
        ).squeeze(-1)

    def compute_score_matching_drift(
        self,
        *,
        model: LatentGenerationModel,
        y: Float[Tensor, "batch latent_dim"],
        t: Float[Tensor, "batch 1"],
        eps: Float[Tensor, "batch latent_dim"],
    ) -> Float[Tensor, "batch latent_dim"]:
        score_mismatch_potential = self.compute_score_mismatch_potential(
            model=model,
            y=y,
            t=t,
            eps=eps,
        )
        return -torch.autograd.grad(
            score_mismatch_potential.sum(),
            y,
        )[0]

    def compute_losses(
        self,
        *,
        model: LatentGenerationModel,
        batch: GenerativeBatch,
    ) -> LatentGenerationStepOutput:
        x = batch.data()
        batch_size = int(x.shape[0])

        y_data = model.encode(x=x)
        y_data_target = y_data.detach()
        x_recon = model.decode(y=y_data)

        x_cycle_data = model.decode(y=y_data_target)
        y_cycle_data = model.encode(x=x_cycle_data)
        z = self.draw_prior_latents(
            batch_size=batch_size,
            device=x.device,
            dtype=x.dtype,
        )
        x_cycle_prior = model.decode(y=z)
        z_cycle = model.encode(x=x_cycle_prior)

        t = self.loss_config.sample_time(
            batch_size=batch_size,
            device=x.device,
            dtype=x.dtype,
        )
        alpha = self.loss_config.get_noise_weight(t=t)

        eps_denoise = torch.randn_like(y_data_target)
        y_t = (1.0 - t) * y_data_target + t * eps_denoise
        y_denoised = model.roundtrip(y=y_t, t=t)

        y_roundtrip = model.encode(
            x=x_cycle_data,
            apply_as_frozen=True,
        )
        eps_denoising_match = torch.randn_like(y_roundtrip)
        score_matching_drift = self.compute_score_matching_drift(
            model=model,
            y=y_roundtrip,
            t=t,
            eps=eps_denoising_match,
        )
        y_score_target = (y_roundtrip.detach() + score_matching_drift.detach()).detach()

        denoising_loss = (alpha * (y_denoised - y_data_target).square()).mean()
        denoising_match_loss = F.mse_loss(y_roundtrip, y_score_target)

        unweighted_loss_terms = {
            "reconstruction": F.mse_loss(x_recon, x),
            "prior_matching": self.compute_prior_matching_loss(y_data=y_data),
            "cycle_data": F.mse_loss(y_cycle_data, y_data_target),
            "cycle_prior": F.mse_loss(z_cycle, z),
            "denoising": denoising_loss,
            "denoising_match": denoising_match_loss,
        }
        weighted_loss_terms = {
            "reconstruction": self.loss_config.reconstruction_weight
            * unweighted_loss_terms["reconstruction"],
            "prior_matching": self.loss_config.prior_matching_weight
            * unweighted_loss_terms["prior_matching"],
            "cycle_data": self.loss_config.cycle_data_weight
            * unweighted_loss_terms["cycle_data"],
            "cycle_prior": self.loss_config.cycle_prior_weight
            * unweighted_loss_terms["cycle_prior"],
            "denoising": self.loss_config.denoising_weight
            * unweighted_loss_terms["denoising"],
            "denoising_match": self.loss_config.denoising_match_weight
            * unweighted_loss_terms["denoising_match"],
        }
        total_loss = sum(weighted_loss_terms.values())

        return LatentGenerationStepOutput(
            total_loss=total_loss,
            loss_terms={"total": total_loss, **weighted_loss_terms},
            weighted_loss_terms=weighted_loss_terms,
            unweighted_loss_terms=unweighted_loss_terms,
        )
