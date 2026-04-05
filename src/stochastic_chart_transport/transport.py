from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import einsum
from jaxtyping import Float
from torch import Tensor

from src.common.training import BaseLoss
from src.config.base import BaseConfig
from src.utils import clip_norm


class StochasticChartTransportLossConfig(BaseConfig):
    transport_step_multiplier: float
    transport_step_cap: float

    num_time_samples: int
    """
    Approximate the actual transport-field integral across this many
    noise-spectrum time samples, with stratified uniform sampling within each bin.
    We do `num_time_samples` forward passes to estimate the drift field.
    """
    t_range: tuple[float, float]
    antipodal_estimate: bool
    decoder_transport_weight: float
    encoder_transport_weight: float
    data_transport_weight: float
    model_transport_weight: float
    decoder_huber_delta: float

    def _sample_stratified_transport_times(
        self, *, batch_size: int, device: torch.device, dtype: torch.dtype
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
            batch_size, self.num_time_samples, device=device, dtype=dtype
        )

    @dataclass
    class Loss(BaseLoss):
        decoder_data: Float[Tensor, ""]
        decoder_model: Float[Tensor, ""]
        encoder_data: Float[Tensor, ""]
        encoder_model: Float[Tensor, ""]

        def sum(self):
            return (
                self.encoder_data
                + self.encoder_model
                + self.decoder_data
                + self.decoder_model
            )

    def _estimate_transport_field(
        self,
        state,
        *,
        data_latent: Float[Tensor, "batch ..."],
        model_latent: Float[Tensor, "batch ..."],
    ) -> Float[Tensor, "double_batch ..."]:
        """
        Returns weighted score estimate under the critic
        """
        batch_size = data_latent.shape[0]
        device = data_latent.device

        noise_t: Float[Tensor, "b t"] = self._sample_stratified_transport_times(
            batch_size=batch_size, device=device, dtype=data_latent.dtype
        )
        noise_t = torch.cat([noise_t, noise_t])
        combined_latent = torch.cat([data_latent, model_latent])
        combined_transport_field = torch.zeros_like(combined_latent)
        combined_categorical = torch.zeros(
            (2 * batch_size,), dtype=torch.long, device=device
        )
        combined_categorical[batch_size:] = 1
        # Estimate the transport field
        with torch.no_grad():
            for j in range(self.num_time_samples):
                t = noise_t[:, j]

                epsilon = state.config.critic.epsilon_like(latent=combined_latent)
                noised_latent = state.config.critic.apply_mixture(
                    latent=combined_latent, epsilon=epsilon, t=t
                )
                # The actual score is -1.0 * noise_estimates / t
                noise_estimates = state.model.critic(
                    noised_latent, t=t, categorical=combined_categorical
                )
                analytic_prior_scores = state.prior_config.analytic_score(
                    y_t=noised_latent, t=t
                )
                if self.antipodal_estimate:
                    antipodal_noised_latent = state.config.critic.apply_mixture(
                        latent=combined_latent, epsilon=-epsilon, t=t
                    )
                    antipodal_noise_estimates = state.model.critic(
                        antipodal_noised_latent, t=t, categorical=combined_categorical
                    )
                    antipodal_prior_scores = state.prior_config.analytic_score(
                        y_t=antipodal_noised_latent, t=t
                    )
                    noise_estimates.add_(antipodal_noise_estimates).div_(2.0)
                    analytic_prior_scores.add_(antipodal_prior_scores).div_(2.0)

                # (1) Weighting to obtain score stimates
                # (2) pullback along the noise process
                latent_score_estimates = einsum(
                    -1.0 / t, noise_estimates, "b, b ... -> b ..."
                )
                noise_level_tranport_field = einsum(
                    1.0 - t,
                    analytic_prior_scores - latent_score_estimates,
                    "b, b ... -> b ...",
                )
                # (1) Bulk-KL weighting of the Wasserstein transport objective
                #   1 / (1-t)**2 corresponds to the uniform-velocity FM weight
                # (2) Average across the time samples
                noise_level_weights: Float[Tensor, "b"] = (1.0 - t).pow(
                    -2.0
                ) / self.num_time_samples
                combined_transport_field.add_(
                    einsum(
                        noise_level_tranport_field,
                        noise_level_weights,
                        "b ..., b -> b ...",
                    )
                )
        return combined_transport_field

    def _scale_clip_transport_field(
        self, field: Float[Tensor, "b ..."]
    ) -> Float[Tensor, "b ..."]:
        return clip_norm(
            field * self.transport_step_multiplier, max_norm=self.transport_step_cap
        )

    def apply(
        self,
        state,
        *,
        data_sample: Float[Tensor, "batch ..."],
        model_sample: Float[Tensor, "batch ..."],
        data_latent: Float[Tensor, "batch ..."],
        model_latent: Float[Tensor, "batch ..."],
    ) -> Loss:
        """
        Transport supervision reusing caller-provided current-step tensors.

        Caller contract:
        1. `data_latent` and `model_latent` must be fresh current-step latents
           with encoder gradients attached if `encoder_loss` should train the
           encoder.
        2. The transport field and transported latent target are always treated
           as detached targets.
        3. `model_sample` is only a decoder-loss target here; its value may be
           detached by the caller without changing transport decoder semantics.
        """
        # Rescale and clip
        combined_transport_field: Float[Tensor, "b ..."] = (
            self._scale_clip_transport_field(
                self._estimate_transport_field(
                    state,
                    data_latent=data_latent.detach(),
                    model_latent=model_latent.detach(),
                )
            )
        )
        transported_data_latent, transported_model_latent = (
            (torch.cat([data_latent, model_latent]) + combined_transport_field)
            .detach()
            .chunk(2, dim=0)
        )
        # We only supervise the data component, not the fiber component
        reconstructed_combined_sample, _ = state.decode(
            torch.cat([transported_data_latent, transported_model_latent], dim=0)
        )
        reconstructed_data_sample, reconstructed_model_sample = (
            reconstructed_combined_sample.chunk(2, dim=0)
        )

        decoder_data_loss = (
            F.huber_loss(
                reconstructed_data_sample,
                data_sample.detach(),
                delta=self.decoder_huber_delta,
                reduction="mean",
            )
            * self.decoder_transport_weight
            * self.data_transport_weight
        )
        decoder_model_loss = (
            F.huber_loss(
                reconstructed_model_sample,
                model_sample.detach(),
                delta=self.decoder_huber_delta,
                reduction="mean",
            )
            * self.decoder_transport_weight
            * self.model_transport_weight
        )
        # Transport step cap is responsible for well-conditioning the mse target
        encoder_data_loss = (
            F.mse_loss(
                data_latent,
                transported_data_latent,
                reduction="mean",
            )
            * self.encoder_transport_weight
            * self.data_transport_weight
        )
        encoder_model_loss = (
            F.mse_loss(
                model_latent,
                transported_model_latent,
                reduction="mean",
            )
            * self.encoder_transport_weight
            * self.model_transport_weight
        )
        return self.Loss(
            encoder_data=encoder_data_loss,
            encoder_model=encoder_model_loss,
            decoder_data=decoder_data_loss,
            decoder_model=decoder_model_loss,
        )
