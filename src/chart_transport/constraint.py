from __future__ import annotations

from src.config.base import BaseConfig


class LagrangianConstraintConfig(BaseConfig):
    data_constraint_budget: float
    prior_constraint_budget: float
    dual_variable_lr: float


class LossConstraintConfig(BaseConfig):
    data_loss_weight: float
    prior_loss_weight: float


class ManifoldConstraintConfig(BaseConfig):
    huber_delta: float
    """Use huber loss to stabilize far-off-manifold conditions."""

    constraint_method: LagrangianConstraintConfig | LossConstraintConfig


__all__ = [
    "LagrangianConstraintConfig",
    "LossConstraintConfig",
    "ManifoldConstraintConfig",
]
