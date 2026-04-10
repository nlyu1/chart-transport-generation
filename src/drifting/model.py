from __future__ import annotations

from typing import Self

import torch
import torch.nn as nn
from jaxtyping import Float
from pydantic import model_validator
from torch import Tensor

from src.config.base import BaseConfig
from src.model.base import ModelConfig


class DriftingModelConfig(BaseConfig):
    generator: ModelConfig
    lr: float
    grad_clip_norm: float
    linear_weight_scale: float = 1.0
    zero_linear_bias: bool = False
    output_bias: Float[Tensor, "output_dim"] | None = None

    @classmethod
    def initialize(
        cls,
        *,
        generator: ModelConfig,
        lr: float,
        grad_clip_norm: float,
        linear_weight_scale: float = 1.0,
        zero_linear_bias: bool = False,
        output_bias: list[float] | None = None,
    ) -> Self:
        output_bias_tensor = None
        if output_bias is not None:
            output_bias_tensor = torch.tensor(output_bias, dtype=torch.float32)
        return cls(
            generator=generator,
            lr=lr,
            grad_clip_norm=grad_clip_norm,
            linear_weight_scale=linear_weight_scale,
            zero_linear_bias=zero_linear_bias,
            output_bias=output_bias_tensor,
        )

    @model_validator(mode="after")
    def _validate_config(self) -> Self:
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive")
        if self.linear_weight_scale <= 0.0:
            raise ValueError("linear_weight_scale must be positive")
        if self.output_bias is not None and self.output_bias.ndim != 1:
            raise ValueError("output_bias must be rank-1")
        return self

    def get_model(self) -> nn.Module:
        model = self.generator.get_model()
        self.initialize_parameters(model)
        return model

    def initialize_parameters(
        self,
        model: nn.Module,
    ) -> None:
        linear_layers: list[nn.Linear] = []
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    module.weight.mul_(self.linear_weight_scale)
                    if self.zero_linear_bias and module.bias is not None:
                        module.bias.zero_()
                    linear_layers.append(module)
            if self.output_bias is not None:
                if not linear_layers:
                    raise ValueError("output_bias requires at least one linear layer")
                output_layer = linear_layers[-1]
                if output_layer.bias is None:
                    raise ValueError("final linear layer must have a bias term")
                if output_layer.bias.shape != self.output_bias.shape:
                    raise ValueError(
                        "output_bias shape must match the final linear layer bias shape"
                    )
                output_layer.bias.copy_(
                    self.output_bias.to(
                        device=output_layer.bias.device,
                        dtype=output_layer.bias.dtype,
                    )
                )

    def get_optimizer(
        self,
        model: nn.Module,
    ) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), lr=self.lr)


__all__ = ["DriftingModelConfig"]
