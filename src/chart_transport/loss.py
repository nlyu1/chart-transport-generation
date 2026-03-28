from __future__ import annotations

from src.chart_transport.constraint import ManifoldConstraintConfig
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

    antipodal_estimate: bool
    """Whether to use an antipodal estimate for the field."""

    decoder_transport_weight: float
    encoder_transport_weight: float
    """Weights for the transport losses."""


class ChartTransportLossConfig(BaseConfig):
    """
    Specifies all the loss components:
    1. manifold constraint: dual variable or loss weighting
    2. transport: how to estimate the transport field, and loss weighting
    3. critic: just the weight of the loss
    """

    constraint_config: ManifoldConstraintConfig
    """Constraint for manifold invariants."""

    transport_config: TransportLossConfig
    """Specification for transport-field estimation."""

    critic_loss_weight: float
    """Weight of the overall loss."""

    update_chart_every_n_critic_steps: int
    """Potentially update the critic many steps before updating the chart."""


__all__ = ["TransportLossConfig", "ChartTransportLossConfig"]
