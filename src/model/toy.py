from __future__ import annotations

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from src.config.base import ConfigMethodsMixin
from src.model.mlp import MLPConfig
from src.model.time_conditioning import TimeConditioningConfig


@pydantic_dataclass(
    kw_only=True,
    frozen=True,
    config=ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    ),
)
class ResidualStackConfig(ConfigMethodsMixin):
    input_projection_config: MLPConfig
    residual_block_config: MLPConfig
    output_projection_config: MLPConfig
    num_blocks: int

    def __post_init__(self) -> None:
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        state_dim = self.input_projection_config.output_dim
        if self.residual_block_config.input_dim != state_dim:
            raise ValueError(
                "residual_block_config.input_dim must match input_projection_config.output_dim"
            )
        if self.residual_block_config.output_dim != state_dim:
            raise ValueError(
                "residual_block_config.output_dim must match input_projection_config.output_dim"
            )
        if self.output_projection_config.input_dim != state_dim:
            raise ValueError(
                "output_projection_config.input_dim must match input_projection_config.output_dim"
            )


@pydantic_dataclass(
    kw_only=True,
    frozen=True,
    config=ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    ),
)
class TimeConditionedResidualStackConfig(ConfigMethodsMixin):
    input_projection_config: MLPConfig
    residual_block_config: MLPConfig
    output_projection_config: MLPConfig
    time_conditioning_config: TimeConditioningConfig
    num_blocks: int

    def __post_init__(self) -> None:
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        state_dim = self.input_projection_config.output_dim
        if self.residual_block_config.input_dim != state_dim:
            raise ValueError(
                "residual_block_config.input_dim must match input_projection_config.output_dim"
            )
        if self.residual_block_config.output_dim != state_dim:
            raise ValueError(
                "residual_block_config.output_dim must match input_projection_config.output_dim"
            )
        if self.output_projection_config.input_dim != state_dim:
            raise ValueError(
                "output_projection_config.input_dim must match input_projection_config.output_dim"
            )


@pydantic_dataclass(
    kw_only=True,
    frozen=True,
    config=ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    ),
)
class ToyLatentModelConfig(ConfigMethodsMixin):
    encoder_config: TimeConditionedResidualStackConfig
    decoder_config: ResidualStackConfig

    def __post_init__(self) -> None:
        if (
            self.encoder_config.output_projection_config.output_dim
            != self.decoder_config.input_projection_config.input_dim
        ):
            raise ValueError(
                "encoder output dim must match decoder input dim"
            )
        if (
            self.decoder_config.output_projection_config.output_dim
            != self.encoder_config.input_projection_config.input_dim
        ):
            raise ValueError(
                "decoder output dim must match encoder input dim"
            )
