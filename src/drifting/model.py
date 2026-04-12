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
    input_skip_weight: Float[Tensor, "output_dim input_dim"] | None = None
    input_skip_bias: Float[Tensor, "output_dim"] | None = None
    residual_output_scale: float = 1.0

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
        input_skip_weight: list[list[float]] | None = None,
        input_skip_bias: list[float] | None = None,
        residual_output_scale: float = 1.0,
    ) -> Self:
        output_bias_tensor = None
        if output_bias is not None:
            output_bias_tensor = torch.tensor(output_bias, dtype=torch.float32)
        input_skip_weight_tensor = None
        if input_skip_weight is not None:
            input_skip_weight_tensor = torch.tensor(
                input_skip_weight,
                dtype=torch.float32,
            )
        input_skip_bias_tensor = None
        if input_skip_bias is not None:
            input_skip_bias_tensor = torch.tensor(
                input_skip_bias,
                dtype=torch.float32,
            )
        return cls(
            generator=generator,
            lr=lr,
            grad_clip_norm=grad_clip_norm,
            linear_weight_scale=linear_weight_scale,
            zero_linear_bias=zero_linear_bias,
            output_bias=output_bias_tensor,
            input_skip_weight=input_skip_weight_tensor,
            input_skip_bias=input_skip_bias_tensor,
            residual_output_scale=residual_output_scale,
        )

    @model_validator(mode="after")
    def _validate_config(self) -> Self:
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive")
        if self.linear_weight_scale <= 0.0:
            raise ValueError("linear_weight_scale must be positive")
        if self.residual_output_scale <= 0.0:
            raise ValueError("residual_output_scale must be positive")
        if self.output_bias is not None and self.output_bias.ndim != 1:
            raise ValueError("output_bias must be rank-1")
        if self.input_skip_weight is not None and self.input_skip_weight.ndim != 2:
            raise ValueError("input_skip_weight must be rank-2")
        if self.input_skip_bias is not None and self.input_skip_bias.ndim != 1:
            raise ValueError("input_skip_bias must be rank-1")
        if (self.input_skip_weight is None) != (self.input_skip_bias is None):
            raise ValueError(
                "input_skip_weight and input_skip_bias must be provided together"
            )
        if (
            self.input_skip_weight is not None
            and self.output_bias is not None
            and self.input_skip_bias.shape != self.output_bias.shape
        ):
            raise ValueError("input_skip_bias and output_bias must share shape")
        return self

    def get_model(self) -> nn.Module:
        residual_model = self.generator.get_model()
        self.initialize_parameters(residual_model)
        if self.input_skip_weight is None:
            return residual_model
        return AffineResidualGenerator(
            residual_model=residual_model,
            input_skip_weight=self.input_skip_weight,
            input_skip_bias=self.input_skip_bias,
            residual_output_scale=self.residual_output_scale,
        )

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


class AffineResidualGenerator(nn.Module):
    def __init__(
        self,
        *,
        residual_model: nn.Module,
        input_skip_weight: Float[Tensor, "output_dim input_dim"],
        input_skip_bias: Float[Tensor, "output_dim"],
        residual_output_scale: float,
    ) -> None:
        super().__init__()
        self.residual_model = residual_model
        self.input_skip = nn.Linear(
            input_skip_weight.shape[1],
            input_skip_weight.shape[0],
            bias=True,
        )
        self.residual_output_scale = residual_output_scale
        with torch.no_grad():
            self.input_skip.weight.copy_(input_skip_weight)
            self.input_skip.bias.copy_(input_skip_bias)

    def forward(
        self,
        latent: Float[Tensor, "batch input_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        return self.input_skip(latent) + self.residual_output_scale * self.residual_model(
            latent
        )


__all__ = ["DriftingModelConfig"]
