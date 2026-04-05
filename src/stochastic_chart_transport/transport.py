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
    """
    Loss weighting & components
    """
    decoder_huber_delta: float
    forward_kl_weight: float
    reverse_kl_weight: float
    data_decoder_weight_multiplier: float  # Applied on top

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
        data_decoder: Float[Tensor, ""]
        data_encoder: Float[Tensor, ""]
        latent: Float[Tensor, ""]

        forward_kl_weight: float
        reverse_kl_weight: float
        data_decoder_weight_multiplier: float

        def sum(self):
            return (
                self.forward_kl_weight
                * (
                    self.data_encoder
                    + self.data_decoder_weight_multiplier * self.data_decoder
                )
                + self.reverse_kl_weight * self.latent
            )

    def _estimate_average_score(
        self,
        state,
        *,
        latent: Float[Tensor, "batch ..."],
        categorical: Float[Tensor, "batch ..."],
    ) -> Float[Tensor, "batch ..."]:
        """
        Given latent coordinates, estimates their scores at the given point
            using specified critic categorical
        """
        batch_size = latent.shape[0]
        device = latent.device

        noise_t: Float[Tensor, "b t"] = self._sample_stratified_transport_times(
            batch_size=batch_size, device=device, dtype=latent.dtype
        )
        weighted_score_field = torch.zeros_like(latent)
        with torch.no_grad():
            for j in range(self.num_time_samples):
                t = noise_t[:, j]
                epsilon = state.config.critic.epsilon_like(latent=latent)
                noised_latent = state.config.critic.apply_mixture(
                    latent=latent, epsilon=epsilon, t=t
                )
                # The actual score is -1.0 * noise_estimates / t
                noise_estimates = state.model.critic(
                    noised_latent, t=t, categorical=categorical
                )
                if self.antipodal_estimate:
                    antipodal_noised_latent = state.config.critic.apply_mixture(
                        latent=latent, epsilon=-epsilon, t=t
                    )
                    antipodal_noise_estimates = state.model.critic(
                        antipodal_noised_latent, t=t, categorical=categorical
                    )
                    noise_estimates.add_(antipodal_noise_estimates).div_(2.0)
                latent_score_estimates = einsum(
                    -1.0 / t, noise_estimates, "b, b ... -> b ..."
                )
                # (1) Bulk-KL weighting of the Wasserstein transport objective
                #   1 / (1-t)**2 corresponds to the uniform-velocity FM weight
                # (2) Average across the time samples
                noise_level_weights: Float[Tensor, "b"] = (1.0 - t).pow(
                    -2.0
                ) / self.num_time_samples
                weighted_score_field.add_(
                    einsum(
                        latent_score_estimates, noise_level_weights, "b ..., b -> b ..."
                    )
                )
        return weighted_score_field

    def _estimate_transport_fields(
        self,
        state,
        *,
        data_latent: Float[Tensor, "batch ..."],
        model_latent: Float[Tensor, "batch ..."],
    ) -> tuple[Float[Tensor, "batch ..."], Float[Tensor, "batch ..."]]:
        """
        Returns the non scaled-clipped transport field for each of data and model
        """
        combined_categorical = torch.zeros(
            (data_latent.shape[0] + model_latent.shape[0],),
            device=data_latent.device,
            dtype=torch.long,
        )
        combined_categorical[-model_latent.shape[0] :] = 1

        # Data-latent scores
        data_latent_data_score, data_latent_model_score = self._estimate_average_score(
            state,
            latent=torch.cat([data_latent, data_latent]),
            categorical=combined_categorical,
        ).chunk(2, dim=0)
        # Model-latent scores
        model_latent_data_score, model_latent_model_score = (
            self._estimate_average_score(
                state,
                latent=torch.cat([model_latent, model_latent]),
                categorical=combined_categorical,
            ).chunk(2, dim=0)
        )

        # Data latents should drift toward model latents
        data_latent_transport_field = data_latent_model_score - data_latent_data_score
        # Model latents should drift toward data latents
        model_latent_transport_field = (
            model_latent_data_score - model_latent_model_score
        )
        return data_latent_transport_field, model_latent_transport_field

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
        data_latent: Float[Tensor, "batch ..."],
        model_latent: Float[Tensor, "batch ..."],
    ) -> Loss:
        # Rescale and clip
        data_latent_transport_field, model_latent_transport_field = (
            self._estimate_transport_fields(
                state, data_latent=data_latent, model_latent=model_latent
            )
        )
        transported_data_latent, transported_model_latent = (
            (
                torch.cat([data_latent, model_latent])
                + self._scale_clip_transport_field(
                    torch.cat(
                        [data_latent_transport_field, model_latent_transport_field]
                    )
                )
            )
            .chunk(2, dim=0)
            .detach()
        )

        # Data-side: attract towards model latents.
        #    Separately update encoder and decoder
        data_encoder_loss = F.mse_loss(
            data_latent, transported_data_latent.detach(), reduction="mean"
        )
        data_decoder_loss = F.huber_loss(
            model_latent,
            transported_model_latent.detach(),
            reduction="mean",
            delta=self.decoder_huber_delta,
        )
        # Model-side: one loss updates both the encoder and decoder
        model_loss = F.mse_loss(
            model_latent, transported_model_latent.detach(), reduction="mean"
        )
        return self.Loss(
            data_encoder=data_encoder_loss,
            data_decoder=data_decoder_loss,
            latent=model_loss,
            forward_kl_weight=self.forward_kl_weight,
            reverse_kl_weight=self.reverse_kl_weight,
            data_decoder_weight_multiplier=self.data_decoder_weight_multiplier,
        )
