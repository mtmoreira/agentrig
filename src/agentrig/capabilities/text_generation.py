"""Provider-independent text-generation request and result contracts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from agentrig.capabilities.base import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    CapabilityRequirements,
)
from agentrig.core._validation import require_trimmed_string
from agentrig.core.artifacts import ArtifactRef
from agentrig.core.context import RunContext


class TextMessageRole(StrEnum):
    """Portable conversation roles accepted by text generators."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class TextGenerationFinishReason(StrEnum):
    """Portable reasons a successful text generation stopped."""

    COMPLETED = "completed"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    OTHER = "other"


@dataclass(frozen=True, slots=True, kw_only=True)
class TextMessage:
    """One text or artifact-bearing message in a generation request."""

    role: TextMessageRole
    text: str | None = field(default=None, repr=False)
    artifacts: tuple[ArtifactRef, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.role, TextMessageRole):
            raise TypeError("text message role must be a TextMessageRole")
        if self.text is not None:
            _require_content_text("text message text", self.text)
        copied_artifacts = _copy_artifacts(
            "text message artifacts",
            self.artifacts,
        )
        if self.text is None and not copied_artifacts:
            raise ValueError("text message requires text or at least one artifact")
        object.__setattr__(self, "artifacts", copied_artifacts)


@dataclass(frozen=True, slots=True, kw_only=True)
class TextGenerationRequest:
    """Free-form or message-based text request with optional artifact inputs."""

    prompt: str | None = field(default=None, repr=False)
    messages: tuple[TextMessage, ...] = field(default=(), repr=False)
    input_artifacts: tuple[ArtifactRef, ...] = field(default=(), repr=False)
    max_output_tokens: int | None = None
    idempotency_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        copied_messages = tuple(self.messages)
        if any(not isinstance(item, TextMessage) for item in copied_messages):
            raise TypeError(
                "text generation messages must contain TextMessage values"
            )
        if (self.prompt is None) == (not copied_messages):
            raise ValueError(
                "text generation requires exactly one prompt or message sequence"
            )
        if self.prompt is not None:
            _require_content_text("text generation prompt", self.prompt)
        object.__setattr__(self, "messages", copied_messages)

        copied_artifacts = _copy_artifacts(
            "text generation input_artifacts",
            self.input_artifacts,
        )
        _require_unique_artifacts(
            (*copied_artifacts, *_message_artifacts(copied_messages)),
        )
        object.__setattr__(self, "input_artifacts", copied_artifacts)

        if self.max_output_tokens is not None and (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 0
        ):
            raise ValueError(
                "text generation max_output_tokens must be a positive integer"
            )
        if self.idempotency_key is not None:
            require_trimmed_string(
                "text generation idempotency key",
                self.idempotency_key,
            )

    @property
    def requirements(self) -> CapabilityRequirements:
        """Derive portable feature and limit requirements from this request."""
        features: list[CapabilityFeature] = []
        if self.messages:
            features.append(CapabilityFeature.MESSAGE_INPUT)
        artifact_count = len(self.input_artifacts) + len(
            _message_artifacts(self.messages)
        )
        limits: dict[CapabilityLimit, int] = {}
        if artifact_count:
            features.append(CapabilityFeature.MULTIMODAL_INPUT)
            limits[CapabilityLimit.MAX_INPUT_ARTIFACTS] = artifact_count
        if self.max_output_tokens is not None:
            limits[
                CapabilityLimit.MAX_OUTPUT_TOKENS
            ] = self.max_output_tokens
        if self.idempotency_key is not None:
            features.append(CapabilityFeature.IDEMPOTENCY_KEYS)
        return CapabilityRequirements(
            kind=CapabilityKind.TEXT_GENERATION,
            features=frozenset(features),
            minimum_limits=limits,
        )

    def require_supported_by(self, descriptor: CapabilityDescriptor) -> None:
        """Fail before provider execution if this request is unsupported."""
        self.requirements.require(descriptor)


@dataclass(frozen=True, slots=True, kw_only=True)
class GenerationUsage:
    """Portable token counts reported by one generation operation."""

    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        _require_optional_count("text generation input_tokens", self.input_tokens)
        _require_optional_count(
            "text generation output_tokens",
            self.output_tokens,
        )

    @property
    def total_tokens(self) -> int | None:
        """Return a total only when both component counts are available."""
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelMetadata:
    """Portable provider and model identity for one generated result."""

    provider: str
    model_id: str
    version: str | None = None

    def __post_init__(self) -> None:
        require_trimmed_string("generation model provider", self.provider)
        require_trimmed_string("generation model ID", self.model_id)
        if self.version is not None:
            require_trimmed_string("generation model version", self.version)


@dataclass(frozen=True, slots=True, kw_only=True)
class TextGenerationResult:
    """Successful text output with portable usage and model metadata."""

    text: str = field(repr=False)
    usage: GenerationUsage
    model: ModelMetadata
    finish_reason: TextGenerationFinishReason

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("generated text must be a string")
        if not isinstance(self.usage, GenerationUsage):
            raise TypeError(
                "text generation result usage must be GenerationUsage"
            )
        if not isinstance(self.model, ModelMetadata):
            raise TypeError(
                "text generation result model must be ModelMetadata"
            )
        if not isinstance(self.finish_reason, TextGenerationFinishReason):
            raise TypeError(
                "text generation finish_reason must be a "
                "TextGenerationFinishReason"
            )


@runtime_checkable
class TextGenerator(Protocol):
    """Generate text through one provider-independent implementation."""

    @property
    def descriptor(self) -> CapabilityDescriptor:
        """Return stable identity and supported optional features."""
        ...

    async def generate(
        self,
        request: TextGenerationRequest,
        context: RunContext,
    ) -> TextGenerationResult:
        """Generate one normalized text result or raise a normalized failure."""
        ...


def _copy_artifacts(
    field_name: str,
    artifacts: Iterable[ArtifactRef],
) -> tuple[ArtifactRef, ...]:
    copied = tuple(artifacts)
    if any(not isinstance(item, ArtifactRef) for item in copied):
        raise TypeError(f"{field_name} must contain ArtifactRef values")
    _require_unique_artifacts(copied)
    return copied


def _message_artifacts(
    messages: Iterable[TextMessage],
) -> tuple[ArtifactRef, ...]:
    return tuple(
        artifact
        for message in messages
        for artifact in message.artifacts
    )


def _require_unique_artifacts(artifacts: Iterable[ArtifactRef]) -> None:
    artifact_ids = tuple(item.artifact_id for item in artifacts)
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError(
            "text generation input artifacts must have unique artifact IDs"
        )


def _require_optional_count(field_name: str, value: int | None) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 0
    ):
        raise ValueError(f"{field_name} must be a non-negative integer or None")


def _require_content_text(field_name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must contain non-whitespace text")
    return value
