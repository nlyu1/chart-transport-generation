from __future__ import annotations

from src.config.base import BaseConfig


class CriticLossConfig(BaseConfig):
    """
    Specify for the score matching critic's loss
    """

    loss_weight: float
    huber_delta: float


class ChartPretrainConfig(BaseConfig):
    """
    Specifies the losses to pretrain the chart.
    Prior to any critic training or transport,
    we pretrain the chart (i.e. initialize to a point on the manifold) by:
    1. Data + prior reconstruction loss
    2. Weakly anchor by asking each sample's latent mean to be zero
        and put softplus penalty on the latents.

    Concretely, the softplus weight looks like, for each sample:
    softplus_loss = softplus(sample norm - softplus radius)).
    """

    zero_mean_weight: float
    softplus_weight: float
    softplus_radius: float


__all__ = ["CriticLossConfig", "ChartPretrainConfig"]
