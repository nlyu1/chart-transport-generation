from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import einsum
from jaxtyping import Float
from torch import Tensor

from src.common.training import BaseLoss
from src.config.base import BaseConfig
from src.utils import clip_norm


class DeterministicChartTransportLossConfig(BaseConfig):
    transport_step_multiplier: float
    transport_step_cap: float

    num_time_samples: int
    t_range: tuple[float, float]
    antipodal_estimate: bool

    decoder_huber_delta: float
    encoder_transport_weight: float
    decoder_transport_weight: float

    def _sample_stratified_transport_times(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Float[Tensor, "batch time_samples"]:
        bin_edges = torch.linspace(
            self.t_range[0],
            self.t_range[1],
            self.num_time_samples + 1,
            device=device,
            dtype=dtype,
        )
        bin_starts = bin_edges[:-1]
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        return bin_starts.unsqueeze(0) + bin_widths.unsqueeze(0) * torch.rand(
            batch_size,
            self.num_time_samples,
            device=device,
            dtype=dtype,
        )

    @dataclass
    class Loss(BaseLoss):
        encoder: Float[Tensor, ""]
        decoder: Float[Tensor, ""]
        encoder_transport_weight: float
        decoder_transport_weight: float

        def sum(self):
            return (
                self.encoder * self.encoder_transport_weight
                + self.decoder * self.decoder_transport_weight
            )

    def zero(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Loss:
        zero = torch.zeros((), device=device, dtype=dtype)
        return self.Loss(
            encoder=zero,
            decoder=zero,
            encoder_transport_weight=self.encoder_transport_weight,
            decoder_transport_weight=self.decoder_transport_weight,
        )

    def _estimate_average_score(
        self,
        state,
        *,
        latent: Float[Tensor, "batch ..."],
    ) -> tuple[Float[Tensor, "batch ..."], Float[Tensor, "batch ..."]]:
        batch_size = latent.shape[0]
        device = latent.device

        noise_t = self._sample_stratified_transport_times(
            batch_size=batch_size,
            device=device,
            dtype=latent.dtype,
        )
        weighted_score_field = torch.zeros_like(latent)
        weighted_prior_score_field = torch.zeros_like(latent)
        with torch.no_grad():
            for j in range(self.num_time_samples):
                t = noise_t[:, j]
                epsilon = state.config.critic.epsilon_like(latent=latent)
                noised_latent = state.config.critic.apply_mixture(
                    latent=latent,
                    epsilon=epsilon,
                    t=t,
                )
                noise_estimates = state.model.critic(noised_latent, t=t)
                latent_prior_score = state.prior_config.analytic_score(
                    y_t=noised_latent,
                    t=t,
                )
                if self.antipodal_estimate:
                    antipodal_noised_latent = state.config.critic.apply_mixture(
                        latent=latent,
                        epsilon=-epsilon,
                        t=t,
                    )
                    antipodal_noise_estimates = state.model.critic(
                        antipodal_noised_latent,
                        t=t,
                    )
                    noise_estimates.add_(antipodal_noise_estimates).div_(2.0)
                    antipodal_prior_score = state.prior_config.analytic_score(
                        y_t=antipodal_noised_latent,
                        t=t,
                    )
                    latent_prior_score.add_(antipodal_prior_score).div_(2.0)
                latent_score_estimates = einsum(
                    -1.0 / t,
                    noise_estimates,
                    "b, b ... -> b ...",
                )
                noise_level_weights = (1.0 - t).pow(-1.0) / self.num_time_samples
                weighted_score_field.add_(
                    einsum(
                        latent_score_estimates,
                        noise_level_weights,
                        "b ..., b -> b ...",
                    )
                )
                weighted_prior_score_field.add_(
                    einsum(
                        latent_prior_score,
                        noise_level_weights,
                        "b ..., b -> b ...",
                    )
                )
        return weighted_score_field, weighted_prior_score_field

    def _estimate_transport_field(
        self,
        state,
        *,
        data_latent: Float[Tensor, "batch ..."],
    ) -> Float[Tensor, "batch ..."]:
        data_latent_score, data_latent_prior_score = self._estimate_average_score(
            state,
            latent=data_latent,
        )
        return data_latent_prior_score - data_latent_score

    def _scale_clip_transport_field(
        self,
        field: Float[Tensor, "batch ..."],
    ) -> Float[Tensor, "batch ..."]:
        return clip_norm(
            field * self.transport_step_multiplier,
            max_norm=self.transport_step_cap,
        )

    def _latent_transport_loss(
        self,
        *,
        latent: Float[Tensor, "batch ..."],
        transported_latent: Float[Tensor, "batch ..."],
    ) -> Float[Tensor, ""]:
        per_sample_squared_error = (
            latent - transported_latent.detach()
        ).reshape(latent.shape[0], -1).square().sum(dim=1)
        return per_sample_squared_error.mean()

    def apply(
        self,
        state,
        *,
        data_sample: Float[Tensor, "batch ..."],
        data_latent: Float[Tensor, "batch ..."],
    ) -> Loss:
        data_latent_transport_field = self._estimate_transport_field(
            state,
            data_latent=data_latent,
        )
        transported_data_latent = (
            data_latent + self._scale_clip_transport_field(data_latent_transport_field)
        ).detach()
        return self.Loss(
            encoder=self._latent_transport_loss(
                latent=data_latent,
                transported_latent=transported_data_latent,
            ),
            decoder=F.huber_loss(
                state.decode(latent=transported_data_latent.detach()),
                data_sample.detach(),
                reduction="mean",
                delta=self.decoder_huber_delta,
            ),
            encoder_transport_weight=self.encoder_transport_weight,
            decoder_transport_weight=self.decoder_transport_weight,
        )


__all__ = ["DeterministicChartTransportLossConfig"]
