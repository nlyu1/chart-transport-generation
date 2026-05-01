"""Improved Precision and Recall for generative models.

Implements the manifold-based estimator from Kynkaanniemi et al., NeurIPS 2019,
"Improved Precision and Recall Metric for Generative Models"
(https://arxiv.org/abs/1904.06991). For each point set the support manifold is
the union of L2 balls centred on every feature, with radius equal to the
distance to that point's ``nearest_k``-th nearest neighbour (excluding itself).
Precision is the fraction of generated samples covered by the data manifold;
recall is the fraction of data samples covered by the generated manifold.

The same pairwise-distance bookkeeping yields the density and coverage variants
of Naeem et al., ICML 2020 (https://arxiv.org/abs/2002.09797), which are
returned alongside precision/recall by :meth:`PrecisionRecallConfig.evaluate`.

Distances are computed in float32 (the ``a^2 + b^2 - 2 a.b`` formula is
catastrophically unstable in bfloat16) and chunked along the query axis so the
full ``len(data) x len(generated)`` matrix never has to live in memory.
"""

from __future__ import annotations

import torch
from jaxtyping import Float
from pydantic import Field
from torch import Tensor

from src.config.base import BaseConfig


def _pairwise_squared_distance_chunk(
    *,
    x_chunk: Float[Tensor, "n d"],
    y: Float[Tensor, "m d"],
    x_norm_sq: Float[Tensor, "n"],
    y_norm_sq: Float[Tensor, "m"],
) -> Float[Tensor, "n m"]:
    cross = x_chunk @ y.transpose(-2, -1)
    sq = x_norm_sq.unsqueeze(1) + y_norm_sq.unsqueeze(0) - 2.0 * cross
    return sq.clamp_min_(0.0)


def _kth_nearest_neighbour_radii_sq(
    *,
    features: Float[Tensor, "n d"],
    k: int,
    chunk_size: int,
) -> Float[Tensor, "n"]:
    """Squared distance from each point to its ``k``-th nearest neighbour.

    Self-distances are excluded by computing the ``k+1`` smallest entries per
    row (the smallest is the self-pair, modulo float roundoff already pinned to
    zero by ``clamp_min_``) and taking the entry at index ``k``.
    """
    n = features.shape[0]
    if n <= k:
        raise ValueError(
            f"need at least k+1={k + 1} samples to define a k={k}-th NN radius; got {n}"
        )
    norm_sq = features.pow(2).sum(dim=-1)
    radii_sq = torch.empty(n, device=features.device, dtype=features.dtype)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_dists = _pairwise_squared_distance_chunk(
            x_chunk=features[start:end],
            y=features,
            x_norm_sq=norm_sq[start:end],
            y_norm_sq=norm_sq,
        )
        topk_vals, _ = torch.topk(
            chunk_dists, k=k + 1, dim=-1, largest=False, sorted=True
        )
        radii_sq[start:end] = topk_vals[:, k]
    return radii_sq


def _ball_membership(
    *,
    query: Float[Tensor, "q d"],
    support: Float[Tensor, "s d"],
    support_radii_sq: Float[Tensor, "s"],
    chunk_size: int,
) -> tuple[Float[Tensor, "q"], Float[Tensor, "q"], Float[Tensor, "q"]]:
    """For each query point, summarise its position relative to support balls.

    Returns three tensors:

    - ``any_cover``: 1.0 if the query lies inside at least one support ball.
    - ``count_cover``: number of support balls containing the query (for density).
    - ``min_dist_sq``: squared distance to the nearest support point (for coverage).
    """
    q = query.shape[0]
    query_norm_sq = query.pow(2).sum(dim=-1)
    support_norm_sq = support.pow(2).sum(dim=-1)
    radii_row = support_radii_sq.unsqueeze(0)

    any_cover = torch.empty(q, device=query.device, dtype=query.dtype)
    count_cover = torch.empty(q, device=query.device, dtype=query.dtype)
    min_dist_sq = torch.empty(q, device=query.device, dtype=query.dtype)
    for start in range(0, q, chunk_size):
        end = min(start + chunk_size, q)
        dists = _pairwise_squared_distance_chunk(
            x_chunk=query[start:end],
            y=support,
            x_norm_sq=query_norm_sq[start:end],
            y_norm_sq=support_norm_sq,
        )
        within = dists <= radii_row
        any_cover[start:end] = within.any(dim=-1).to(query.dtype)
        count_cover[start:end] = within.sum(dim=-1).to(query.dtype)
        min_dist_sq[start:end] = dists.min(dim=-1).values
    return any_cover, count_cover, min_dist_sq


class PrecisionRecallConfig(BaseConfig):
    """Improved precision/recall (and density/coverage) on pre-extracted features.

    Inputs to :meth:`get_precision`, :meth:`get_recall`, and :meth:`evaluate`
    are batches of feature vectors -- typically the activations of a pretrained
    classifier (VGG, Inception, CLIP, ...). No normalisation is applied here;
    feed in whatever embedding the downstream evaluation specifies.

    ``get_precision`` and ``get_recall`` each perform the full pairwise-distance
    pipeline. Call :meth:`evaluate` instead when both numbers (or density and
    coverage) are wanted -- it returns the whole bundle from a single pass.
    """

    nearest_k: int = Field(default=3, ge=1)
    distance_chunk: int = Field(default=4096, ge=1)

    def get_precision(
        self,
        *,
        data: Float[Tensor, "n d"],
        generated: Float[Tensor, "m d"],
    ) -> float:
        return self.evaluate(data=data, generated=generated)["precision"]

    def get_recall(
        self,
        *,
        data: Float[Tensor, "n d"],
        generated: Float[Tensor, "m d"],
    ) -> float:
        return self.evaluate(data=data, generated=generated)["recall"]

    def evaluate(
        self,
        *,
        data: Float[Tensor, "n d"],
        generated: Float[Tensor, "m d"],
    ) -> dict[str, float]:
        if data.ndim != 2 or generated.ndim != 2:
            raise ValueError(
                f"data and generated must be 2D (batch, features); "
                f"got shapes {tuple(data.shape)} and {tuple(generated.shape)}"
            )
        if data.shape[-1] != generated.shape[-1]:
            raise ValueError(
                f"feature dim mismatch: data={data.shape[-1]} vs "
                f"generated={generated.shape[-1]}"
            )
        if data.device != generated.device:
            raise ValueError(
                f"data and generated must live on the same device; "
                f"got {data.device} and {generated.device}"
            )

        data_f = data.detach().to(torch.float32)
        generated_f = generated.detach().to(torch.float32)

        data_radii_sq = _kth_nearest_neighbour_radii_sq(
            features=data_f, k=self.nearest_k, chunk_size=self.distance_chunk
        )
        gen_radii_sq = _kth_nearest_neighbour_radii_sq(
            features=generated_f, k=self.nearest_k, chunk_size=self.distance_chunk
        )

        gen_in_data_any, gen_in_data_count, _ = _ball_membership(
            query=generated_f,
            support=data_f,
            support_radii_sq=data_radii_sq,
            chunk_size=self.distance_chunk,
        )
        data_in_gen_any, _, data_to_gen_min_sq = _ball_membership(
            query=data_f,
            support=generated_f,
            support_radii_sq=gen_radii_sq,
            chunk_size=self.distance_chunk,
        )

        precision = float(gen_in_data_any.mean().item())
        recall = float(data_in_gen_any.mean().item())
        density = float(gen_in_data_count.mean().item() / self.nearest_k)
        coverage = float(
            (data_to_gen_min_sq <= data_radii_sq).to(torch.float32).mean().item()
        )

        return {
            "precision": precision,
            "recall": recall,
            "density": density,
            "coverage": coverage,
        }
