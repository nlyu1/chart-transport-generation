from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, TypeVar

from src.config.base import BaseConfig


StructuredResponseT = TypeVar("StructuredResponseT", bound=BaseConfig)


class ModelClient(Protocol):
    def generate_structured(
        self,
        *,
        role: str,
        system_prompt: str,
        input_sections: Mapping[str, Any],
        output_type: type[StructuredResponseT],
    ) -> StructuredResponseT:
        ...


class DryRunModelClient:
    def __init__(
        self,
        *,
        responses: Mapping[str, object],
    ) -> None:
        self.responses = dict(responses)

    def generate_structured(
        self,
        *,
        role: str,
        system_prompt: str,
        input_sections: Mapping[str, Any],
        output_type: type[StructuredResponseT],
    ) -> StructuredResponseT:
        del system_prompt
        del input_sections
        if role not in self.responses:
            raise KeyError(f"No dry-run response configured for role: {role}")
        raw_response = self.responses[role]
        if isinstance(raw_response, output_type):
            return raw_response
        if isinstance(raw_response, Mapping):
            return output_type.model_validate(raw_response)
        raise TypeError(
            f"Dry-run response for role {role} must be {output_type.__name__} "
            "or a mapping."
        )


__all__ = ["DryRunModelClient", "ModelClient"]
