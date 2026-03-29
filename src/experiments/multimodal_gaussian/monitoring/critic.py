from __future__ import annotations

from typing import TYPE_CHECKING

from jaxtyping import Float, Int
import torch
from torch import Tensor

from src.data.mnist.monitoring import (
    build_latent_grid,
    project_latent_vectors_to_pca_plane,
    write_critic_monitor_artifacts,
)
from src.monitoring.configs import CriticMonitorConfig
from src.monitoring.utils import step_folder

if TYPE_CHECKING:
    from src.experiments.multimodal_gaussian.state import MultimodalTrainingRuntime


def _sample_mode_batch(
    *,
    rt: "MultimodalTrainingRuntime",
    batch_size_per_mode: int,
) -> tuple[Float[Tensor, "batch ambient_dim"], Int[Tensor, "batch"]]:
    samples = []
    labels = []
    for mode_id in range(rt.runtime_data_config.num_modes):
        samples.append(
            rt.runtime_data_config.sample_class(
                mode_id=mode_id,
                batch_size=batch_size_per_mode,
            )
        )
        labels.append(
            torch.full(
                (batch_size_per_mode,),
                fill_value=mode_id,
                device=rt.device,
                dtype=torch.long,
            )
        )
    return torch.cat(samples, dim=0), torch.cat(labels, dim=0)


def _critic_score_from_noise_prediction(
    *,
    predicted_noise: Float[Tensor, "batch latent_dim"],
    t: Float[Tensor, "batch"],
) -> Float[Tensor, "batch latent_dim"]:
    return -predicted_noise / t.unsqueeze(-1)


class GaussianCriticMonitorConfig(CriticMonitorConfig):
    transport_grid_resolution: int
    transport_num_time_samples: int

    def _transport_t_values(
        self,
        *,
        rt: "MultimodalTrainingRuntime",
    ) -> list[float]:
        transport_config = rt.tc.chart_transport_config.loss_config.transport_config
        t_min, t_max = transport_config.t_range
        return torch.linspace(
            t_min,
            t_max,
            self.transport_num_time_samples,
            dtype=torch.float32,
        ).tolist()

    def _sample_snapshot(
        self,
        *,
        rt: "MultimodalTrainingRuntime",
        clean_latents: Float[Tensor, "batch latent_dim"],
        t_value: float,
    ) -> tuple[
        Float[Tensor, "batch latent_dim"],
        Float[Tensor, "batch latent_dim"],
    ]:
        t = torch.full(
            (clean_latents.shape[0],),
            float(t_value),
            device=clean_latents.device,
            dtype=torch.float32,
        )
        noise = torch.randn_like(clean_latents)
        noised_latents = (
            (1.0 - t).unsqueeze(-1) * clean_latents + t.unsqueeze(-1) * noise
        )
        predicted_noise = rt.chart_transport_model.critic(noised_latents, t).float()
        data_score = _critic_score_from_noise_prediction(
            predicted_noise=predicted_noise,
            t=t,
        )
        return noised_latents.float(), data_score.float()

    def _estimate_clean_transport_field(
        self,
        *,
        rt: "MultimodalTrainingRuntime",
        clean_latents: Float[Tensor, "batch latent_dim"],
        t_values: list[float],
    ) -> Float[Tensor, "batch latent_dim"]:
        if len(t_values) == 0:
            raise ValueError("t_values must be non-empty")

        prior_config = rt.tc.chart_transport_config.prior_config
        transport_config = rt.tc.chart_transport_config.loss_config.transport_config

        transport_field = torch.zeros_like(clean_latents, dtype=torch.float32)
        for t_value in t_values:
            t = torch.full(
                (clean_latents.shape[0],),
                float(t_value),
                device=clean_latents.device,
                dtype=torch.float32,
            )
            pullback_weight = transport_config.kl_weight_schedule.pullback_weight(
                t.float(),
            ).unsqueeze(-1)
            noise = torch.randn_like(clean_latents)

            def evaluate_with_noise(
                *,
                sampled_noise: Float[Tensor, "batch latent_dim"],
            ) -> Float[Tensor, "batch latent_dim"]:
                noised_latents = (
                    (1.0 - t).unsqueeze(-1) * clean_latents
                    + t.unsqueeze(-1) * sampled_noise
                )
                predicted_noise = rt.chart_transport_model.critic(
                    noised_latents,
                    t,
                ).float()
                prior_score = prior_config.analytic_score(
                    noised_latents.float(),
                    t.float(),
                ).float()
                return pullback_weight * (
                    prior_score + predicted_noise / t.unsqueeze(-1)
                )

            transport_terms = evaluate_with_noise(sampled_noise=noise)
            if transport_config.antipodal_estimate:
                transport_terms = 0.5 * (
                    transport_terms + evaluate_with_noise(sampled_noise=-noise)
                )
            transport_field = transport_field + transport_terms

        return transport_field / len(t_values)

    def apply_to(
        self,
        *,
        rt: "MultimodalTrainingRuntime",
        step: int,
        stage: str,
    ) -> dict[str, float]:
        dense_samples, dense_labels = _sample_mode_batch(
            rt=rt,
            batch_size_per_mode=self.n_data_latents_per_mode,
        )
        vector_samples, vector_labels = _sample_mode_batch(
            rt=rt,
            batch_size_per_mode=self.n_vectors_per_mode,
        )
        dense_clean_latents = rt.chart_transport_model.encoder(dense_samples).float()
        vector_clean_latents = rt.chart_transport_model.encoder(vector_samples).float()

        score_snapshots = []
        for t_value in self.sample_t_values:
            cloud_latents, _ = self._sample_snapshot(
                rt=rt,
                clean_latents=dense_clean_latents,
                t_value=t_value,
            )
            arrow_latents, data_score = self._sample_snapshot(
                rt=rt,
                clean_latents=vector_clean_latents,
                t_value=t_value,
            )
            score_snapshots.append(
                (
                    t_value,
                    cloud_latents.detach().cpu().float(),
                    arrow_latents.detach().cpu().float(),
                    data_score.detach().cpu().float(),
                )
            )

        transport_t_values = self._transport_t_values(rt=rt)
        transport_field = self._estimate_clean_transport_field(
            rt=rt,
            clean_latents=vector_clean_latents,
            t_values=transport_t_values,
        )
        transport_grid_points, transport_grid_xs, transport_grid_ys = build_latent_grid(
            reference_points=dense_clean_latents,
            resolution=self.transport_grid_resolution,
        )
        transport_grid_field = self._estimate_clean_transport_field(
            rt=rt,
            clean_latents=transport_grid_points,
            t_values=transport_t_values,
        )
        transport_grid_projection = project_latent_vectors_to_pca_plane(
            reference_points=dense_clean_latents,
            vectors=transport_grid_field,
        ).norm(dim=-1).reshape(
            transport_grid_ys.shape[0],
            transport_grid_xs.shape[0],
        )

        write_critic_monitor_artifacts(
            output_folder=step_folder(run_folder=rt.tc.folder, step=step),
            dense_clean_latents=dense_clean_latents.detach().cpu().float(),
            dense_labels=dense_labels.detach().cpu().long(),
            vector_clean_latents=vector_clean_latents.detach().cpu().float(),
            vector_labels=vector_labels.detach().cpu().long(),
            score_snapshots=score_snapshots,
            transport_field=transport_field.detach().cpu().float(),
            transport_grid_xs=transport_grid_xs.detach().cpu().float(),
            transport_grid_ys=transport_grid_ys.detach().cpu().float(),
            transport_grid_projection=transport_grid_projection.detach().cpu().float(),
            num_contour_lines=self.num_contour_lines,
        )

        data_score_norms = [
            score_snapshot[3].reshape(score_snapshot[3].shape[0], -1).norm(dim=-1).mean()
            for score_snapshot in score_snapshots
        ]
        transport_field_norm = transport_field.norm(dim=-1)

        return {
            "critic_monitor_snapshot_score_norm_mean": torch.stack(
                data_score_norms,
            ).mean().item(),
            "critic_monitor_transport_norm_mean": transport_field_norm.mean().item(),
            "critic_monitor_transport_norm_max": transport_field_norm.max().item(),
            "critic_monitor_transport_t_min": min(transport_t_values),
            "critic_monitor_transport_t_max": max(transport_t_values),
        }


__all__ = ["GaussianCriticMonitorConfig"]
