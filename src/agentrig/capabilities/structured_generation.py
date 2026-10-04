"""Strict provider-independent structured-generation contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import re
from types import MappingProxyType
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
from agentrig.core._validation import freeze_string_map, require_trimmed_string
from agentrig.core.context import RunContext

OutputT = TypeVar("OutputT")
_DIAGNOSTIC_TOKEN = re.compile(r"[a-z][a-z0-9_.-]{0,95}")
_DIAGNOSTIC_LOCATION = re.compile(
    r"/(?:[A-Za-z0-9_.~-]+|\*)(?:/(?:[A-Za-z0-9_.~-]+|\*))*|/"
)


def _require_diagnostic_token(value: object) -> None:
    if type(value) is not str or _DIAGNOSTIC_TOKEN.fullmatch(value) is None:
        raise ValueError("structured decode diagnostic token is invalid")


@dataclass(frozen=True, slots=True, kw_only=True)
class StructuredDecodeDiagnostic:
    """One code-owned, content-free decoder code and location pair.

    Locations are fixed slash-delimited schema paths; ``*`` denotes any array
    member and ``/`` the root. Bounds and syntax do not establish privacy:
    callers must never construct this registry from provider content or IDs.
    """

    code: str
    location: str

    def __post_init__(self) -> None:
        _require_diagnostic_token(self.code)
        if (
            type(self.location) is not str
            or len(self.location) > 256
            or _DIAGNOSTIC_LOCATION.fullmatch(self.location) is None
        ):
            raise ValueError("structured decode diagnostic location is invalid")


class StructuredOutputDecodeError(ValueError):
    """Request an optional schema-owned diagnostic without exception text."""

    def __init__(self, diagnostic_key: str) -> None:
        _require_diagnostic_token(diagnostic_key)
        self.diagnostic_key = diagnostic_key
        super().__init__("structured output decoder rejected value")


@dataclass(frozen=True, slots=True, kw_only=True)
class StructuredOutputSchema(Generic[OutputT]):
    """Immutable JSON schema plus the decoder that establishes typed output."""

    schema_id: str
    json_schema: Mapping[str, JsonValue]
    decoder: Callable[[JsonValue], OutputT] = field(repr=False, compare=False)
    decode_diagnostics: Mapping[str, StructuredDecodeDiagnostic] = field(
        default_factory=dict, repr=False
    )

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
        if not isinstance(self.decode_diagnostics, Mapping):
            raise TypeError("structured decode diagnostics must be a mapping")
        if len(self.decode_diagnostics) > 256:
            raise ValueError("structured decode diagnostic registry is too large")
        copied: dict[str, StructuredDecodeDiagnostic] = {}
        for key, diagnostic in self.decode_diagnostics.items():
            _require_diagnostic_token(key)
            if type(diagnostic) is not StructuredDecodeDiagnostic:
                raise TypeError("structured decode diagnostic entry is invalid")
            copied[key] = diagnostic
        object.__setattr__(
            self, "decode_diagnostics", MappingProxyType(dict(sorted(copied.items())))
        )

    def decode(self, value: JsonValue) -> OutputT:
        """Freeze provider JSON before decoding it into the declared type."""
        return self.decoder(
            freeze_json_value("structured generation output", value)
        )

    def resolve_decode_diagnostic(
        self, error: BaseException
    ) -> StructuredDecodeDiagnostic | None:
        """Return only the trusted registered pair; ignore arbitrary errors.

        Exact-type checks avoid reading attributes on unexpected exceptions or
        subclasses. The exception key is never itself retained as metadata.
        """
        if type(error) is not StructuredOutputDecodeError:
            return None
        key = vars(error).get("diagnostic_key")
        if type(key) is not str or len(key) > 96:
            return None
        return self.decode_diagnostics.get(key)


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
    provider_metadata: Mapping[str, str]

    def __init__(
        self,
        *,
        encoded_output: JsonValue,
        output_schema: StructuredOutputSchema[OutputT],
        usage: GenerationUsage,
        model: ModelMetadata,
        finish_reason: TextGenerationFinishReason,
        provider_metadata: Mapping[str, str] | None = None,
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
        object.__setattr__(self, "provider_metadata", freeze_string_map(
            "structured generation provider metadata", provider_metadata or {}))


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
