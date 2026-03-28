from __future__ import annotations

from src.chart_transport.field import BaseKLWeightSchedule
from src.config.base import BaseConfig


class TransportLossConfig(BaseConfig):
    """
    Specification for how the transport field is estimated and determines transport.
    """

    kl_weight_schedule: BaseKLWeightSchedule
    """Weight along the noise spectrum."""

    transport_step_size: float
    """Target transport is computed as transport_step_size * field."""

    num_time_samples: int
    """
    Approximate the actual transport-field integral across this many
    noise-spectrum time samples, with stratified uniform sampling within each bin.
    """

    t_range: tuple[float, float]
    """Closed training/estimation interval for transport times."""

    antipodal_estimate: bool
    """Whether to use an antipodal estimate for the field."""

    decoder_transport_weight: float
    encoder_transport_weight: float
    huber_delta: float
    """Weights for the transport losses."""


__all__ = ["TransportLossConfig"]
