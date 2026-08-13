"""Strict provider-independent structured-generation contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar, runtime_checkable

from agentrig.capabilities.base import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityRequirements,
)
from agentrig.capabilities.text_generation import (
    GenerationUsage,
    ModelMetadata,
    TextGenerationFinishReason,
    TextGenerationRequest,
)
from agentrig.core._json import (
    JsonValue,
    freeze_json_object,
    freeze_json_value,
)
from agentrig.core._validation import require_trimmed_string
from agentrig.core.context import RunContext

OutputT = TypeVar("OutputT")


@dataclass(frozen=True, slots=True, kw_only=True)
class StructuredOutputSchema(Generic[OutputT]):
    """Immutable JSON schema plus the decoder that establishes typed output."""

    schema_id: str
    json_schema: Mapping[str, JsonValue]
    decoder: Callable[[JsonValue], OutputT] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        require_trimmed_string("structured output schema ID", self.schema_id)
        frozen_schema = freeze_json_object(
            "structured output JSON schema",
            self.json_schema,
        )
        if not frozen_schema:
            raise ValueError("structured output JSON schema must not be empty")
        object.__setattr__(self, "json_schema", frozen_schema)
        if not callable(self.decoder):
            raise TypeError("structured output schema decoder must be callable")

    def decode(self, value: JsonValue) -> OutputT:
        """Freeze provider JSON before decoding it into the declared type."""
        return self.decoder(
            freeze_json_value("structured generation output", value)
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class StructuredGenerationRequest(Generic[OutputT]):
    """Text or multimodal input paired with one strict output schema."""

    input: TextGenerationRequest = field(repr=False)
    output_schema: StructuredOutputSchema[OutputT] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.input, TextGenerationRequest):
            raise TypeError(
                "structured generation input must be a TextGenerationRequest"
            )
        if not isinstance(self.output_schema, StructuredOutputSchema):
            raise TypeError(
                "structured generation output_schema must be a "
                "StructuredOutputSchema"
            )

    @property
    def requirements(self) -> CapabilityRequirements:
        """Require strict output plus every feature needed by the input."""
        input_requirements = self.input.requirements
        return CapabilityRequirements(
            kind=CapabilityKind.STRUCTURED_GENERATION,
            features=frozenset(
                {
                    *input_requirements.features,
                    CapabilityFeature.STRUCTURED_OUTPUT,
                }
            ),
            minimum_limits=input_requirements.minimum_limits,
            allowed_data_retention=(
                input_requirements.allowed_data_retention
            ),
        )

    def require_supported_by(self, descriptor: CapabilityDescriptor) -> None:
        """Fail before provider execution if this request is unsupported."""
        self.requirements.require(descriptor)


@dataclass(frozen=True, slots=True, init=False)
class StructuredGenerationResult(Generic[OutputT]):
    """Schema-decoded output with its frozen JSON and generation metadata."""

    output: OutputT = field(repr=False)
    encoded_output: JsonValue = field(repr=False)
    schema_id: str
    usage: GenerationUsage
    model: ModelMetadata
    finish_reason: TextGenerationFinishReason

    def __init__(
        self,
        *,
        encoded_output: JsonValue,
        output_schema: StructuredOutputSchema[OutputT],
        usage: GenerationUsage,
        model: ModelMetadata,
        finish_reason: TextGenerationFinishReason,
    ) -> None:
        if not isinstance(output_schema, StructuredOutputSchema):
            raise TypeError(
                "structured generation result output_schema must be a "
                "StructuredOutputSchema"
            )
        if not isinstance(usage, GenerationUsage):
            raise TypeError(
                "structured generation result usage must be GenerationUsage"
            )
        if not isinstance(model, ModelMetadata):
            raise TypeError(
                "structured generation result model must be ModelMetadata"
            )
        if not isinstance(finish_reason, TextGenerationFinishReason):
            raise TypeError(
                "structured generation finish_reason must be a "
                "TextGenerationFinishReason"
            )

        frozen_output = freeze_json_value(
            "structured generation output",
            encoded_output,
        )
        decoded_output = output_schema.decode(frozen_output)
        object.__setattr__(self, "output", decoded_output)
        object.__setattr__(self, "encoded_output", frozen_output)
        object.__setattr__(self, "schema_id", output_schema.schema_id)
        object.__setattr__(self, "usage", usage)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "finish_reason", finish_reason)


@runtime_checkable
class StructuredGenerator(Protocol[OutputT]):
    """Generate one schema-decoded value through a portable implementation."""

    @property
    def descriptor(self) -> CapabilityDescriptor:
        """Return stable identity and supported optional features."""
        ...

    async def generate(
        self,
        request: StructuredGenerationRequest[OutputT],
        context: RunContext,
    ) -> StructuredGenerationResult[OutputT]:
        """Generate a strict result or raise a normalized failure."""
        ...
