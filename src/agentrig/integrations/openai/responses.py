"""Portable structured multimodal adapter for OpenAI Responses."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, Protocol, TypeVar, runtime_checkable

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    DataRetention,
    GenerationUsage,
    ModelMetadata,
    StructuredGenerationRequest,
    StructuredGenerationResult,
    TextGenerationFinishReason,
    TextMessageRole,
)
from agentrig.core import (
    ArtifactRef,
    ArtifactResolver,
    DeadlineExceeded,
    ResolvedArtifact,
    RunContext,
)
from agentrig.core._json import JsonValue, freeze_json_object
from agentrig.core._validation import require_trimmed_string
from agentrig.core.errors import AgentRigError, Failure, FailureKind

OPENAI_RESPONSES_SDK_VERSION = "2.47.0"
OPENAI_RESPONSES_MAX_INPUT_IMAGES = 10
OPENAI_RESPONSES_MAX_IMAGE_BYTES = 20 * 1024 * 1024
OPENAI_RESPONSES_MAX_OUTPUT_TOKENS = 32_768

OPENAI_RESPONSES_STRUCTURED_CAPABILITY = CapabilityDescriptor(
    capability_id="openai.responses.structured_generation",
    version=OPENAI_RESPONSES_SDK_VERSION,
    kind=CapabilityKind.STRUCTURED_GENERATION,
    features=frozenset(
        {
            CapabilityFeature.MESSAGE_INPUT,
            CapabilityFeature.MULTIMODAL_INPUT,
            CapabilityFeature.CANCELLATION,
            CapabilityFeature.STRUCTURED_OUTPUT,
            CapabilityFeature.USAGE_REPORTING,
        }
    ),
    limits={
        CapabilityLimit.MAX_INPUT_ARTIFACTS: OPENAI_RESPONSES_MAX_INPUT_IMAGES,
        CapabilityLimit.MAX_OUTPUT_TOKENS: OPENAI_RESPONSES_MAX_OUTPUT_TOKENS,
    },
    data_retention=DataRetention.PROVIDER_MANAGED,
)

OutputT = TypeVar("OutputT")
_SCHEMA_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")
_IMAGE_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


@dataclass(frozen=True, slots=True, kw_only=True)
class OpenAIResponsesImage:
    """One bounded image detached from its application storage location."""

    media_type: str
    content: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if self.media_type not in _IMAGE_MEDIA_TYPES:
            raise ValueError("OpenAI Responses image media type is unsupported")
        if not isinstance(self.content, bytes) or not self.content:
            raise ValueError("OpenAI Responses image content must be nonempty bytes")
        if len(self.content) > OPENAI_RESPONSES_MAX_IMAGE_BYTES:
            raise ValueError("OpenAI Responses image exceeds the byte limit")


@dataclass(frozen=True, slots=True, kw_only=True)
class OpenAIResponsesMessage:
    """Private text and image content for one Responses input message."""

    role: TextMessageRole
    text: str | None = field(default=None, repr=False)
    images: tuple[OpenAIResponsesImage, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.role, TextMessageRole):
            raise TypeError("OpenAI Responses message role is invalid")
        if self.text is not None:
            require_trimmed_string("OpenAI Responses message text", self.text)
        images = tuple(self.images)
        if any(not isinstance(image, OpenAIResponsesImage) for image in images):
            raise TypeError("OpenAI Responses message images are invalid")
        if self.text is None and not images:
            raise ValueError("OpenAI Responses message requires content")
        if images and self.role is not TextMessageRole.USER:
            raise ValueError("OpenAI Responses images require a user message")
        object.__setattr__(self, "images", images)


@dataclass(frozen=True, slots=True, kw_only=True)
class OpenAIResponsesRequest:
    """One strict, stateless, tool-free Responses request."""

    model: str
    messages: tuple[OpenAIResponsesMessage, ...] = field(repr=False)
    schema_name: str
    output_schema: Mapping[str, JsonValue] = field(repr=False)
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        require_trimmed_string("OpenAI Responses model", self.model)
        messages = tuple(self.messages)
        if not messages:
            raise ValueError("OpenAI Responses request requires messages")
        if any(not isinstance(message, OpenAIResponsesMessage) for message in messages):
            raise TypeError("OpenAI Responses request messages are invalid")
        if _SCHEMA_NAME_PATTERN.fullmatch(self.schema_name) is None:
            raise ValueError("OpenAI Responses schema name is invalid")
        schema = freeze_json_object(
            "OpenAI Responses output schema",
            self.output_schema,
        )
        if not schema:
            raise ValueError("OpenAI Responses output schema must not be empty")
        if self.max_output_tokens is not None and (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens < 1
        ):
            raise ValueError("OpenAI Responses max output tokens must be positive")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "output_schema", schema)


class OpenAIResponsesStatus(StrEnum):
    """Safe terminal status detached from an SDK response."""

    COMPLETED = "completed"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True, kw_only=True)
class OpenAIResponsesResult:
    """Strict JSON text and portable metadata from one SDK response."""

    output_text: str = field(repr=False)
    model: str
    status: OpenAIResponsesStatus
    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        require_trimmed_string("OpenAI Responses output", self.output_text)
        require_trimmed_string("OpenAI Responses response model", self.model)
        if not isinstance(self.status, OpenAIResponsesStatus):
            raise TypeError("OpenAI Responses status is invalid")
        for name, value in (
            ("input", self.input_tokens),
            ("output", self.output_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"OpenAI Responses {name} tokens are invalid")


@runtime_checkable
class OpenAIResponsesClient(Protocol):
    """Minimal injected client seam implemented by the optional SDK bridge."""

    async def create(
        self,
        request: OpenAIResponsesRequest,
    ) -> OpenAIResponsesResult: ...

    async def close(self) -> None: ...


@runtime_checkable
class OpenAIResponsesAuthenticationSource(Protocol):
    """Resolve an application-owned API key only at client creation."""

    def resolve_api_key(self) -> str: ...


@runtime_checkable
class OpenAIResponsesClientFactory(Protocol):
    """Create a short-lived client after portable request preflight."""

    def create(self) -> OpenAIResponsesClient: ...


class OpenAIResponsesStructuredGenerator(Generic[OutputT]):
    """Resolve image artifacts and execute one strict Responses call."""

    def __init__(
        self,
        *,
        client_factory: OpenAIResponsesClientFactory,
        artifact_resolver: ArtifactResolver,
        model: str,
    ) -> None:
        if not isinstance(client_factory, OpenAIResponsesClientFactory):
            raise TypeError("OpenAI Responses factory is invalid")
        if not isinstance(artifact_resolver, ArtifactResolver):
            raise TypeError("OpenAI Responses artifact resolver is invalid")
        require_trimmed_string("OpenAI Responses model", model)
        self._client_factory = client_factory
        self._artifact_resolver = artifact_resolver
        self._model = model

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return OPENAI_RESPONSES_STRUCTURED_CAPABILITY

    async def generate(
        self,
        request: StructuredGenerationRequest[OutputT],
        context: RunContext,
    ) -> StructuredGenerationResult[OutputT]:
        if not isinstance(request, StructuredGenerationRequest):
            raise TypeError("OpenAI Responses request is invalid")
        if not isinstance(context, RunContext):
            raise TypeError("OpenAI Responses context is invalid")
        _check_constraints(context)
        request.require_supported_by(self.descriptor)
        messages = await self._resolve_messages(request, context)
        _check_constraints(context)
        client = self._client_factory.create()
        try:
            response = await _call_with_constraints(
                client,
                OpenAIResponsesRequest(
                    model=self._model,
                    messages=messages,
                    schema_name=_schema_name(request.output_schema.schema_id),
                    output_schema=dict(request.output_schema.json_schema),
                    max_output_tokens=request.input.max_output_tokens,
                ),
                context,
            )
        finally:
            try:
                await client.close()
            except Exception:
                pass
        try:
            encoded = json.loads(response.output_text)
            return StructuredGenerationResult(
                encoded_output=encoded,
                output_schema=request.output_schema,
                usage=GenerationUsage(
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                ),
                model=ModelMetadata(provider="openai", model_id=response.model),
                finish_reason=(
                    TextGenerationFinishReason.COMPLETED
                    if response.status is OpenAIResponsesStatus.COMPLETED
                    else TextGenerationFinishReason.LENGTH
                ),
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            raise AgentRigError(
                Failure(
                    kind=FailureKind.PERMANENT_PROVIDER,
                    message="OpenAI Responses returned invalid structured output",
                    code="openai.responses.invalid_output",
                )
            ) from None

    async def _resolve_messages(
        self,
        request: StructuredGenerationRequest[OutputT],
        context: RunContext,
    ) -> tuple[OpenAIResponsesMessage, ...]:
        source = request.input
        if source.prompt is not None:
            images = await self._resolve_images(source.input_artifacts, context)
            return (
                OpenAIResponsesMessage(
                    role=TextMessageRole.USER,
                    text=source.prompt,
                    images=images,
                ),
            )
        resolved: list[OpenAIResponsesMessage] = []
        for message in source.messages:
            images = await self._resolve_images(message.artifacts, context)
            resolved.append(
                OpenAIResponsesMessage(
                    role=message.role,
                    text=message.text,
                    images=images,
                )
            )
        if source.input_artifacts:
            images = await self._resolve_images(source.input_artifacts, context)
            resolved.append(
                OpenAIResponsesMessage(
                    role=TextMessageRole.USER,
                    images=images,
                )
            )
        return tuple(resolved)

    async def _resolve_images(
        self,
        artifacts: tuple[ArtifactRef, ...],
        context: RunContext,
    ) -> tuple[OpenAIResponsesImage, ...]:
        images: list[OpenAIResponsesImage] = []
        for artifact in artifacts:
            _check_constraints(context)
            resolved = await _resolve_with_constraints(
                self._artifact_resolver,
                artifact,
                context,
            )
            if (
                not isinstance(resolved, ResolvedArtifact)
                or resolved.artifact != artifact
            ):
                raise AgentRigError(
                    Failure(
                        kind=FailureKind.INVALID_INPUT,
                        message="artifact resolver returned mismatched content",
                        code="artifact.resolution_mismatch",
                    )
                )
            images.append(
                OpenAIResponsesImage(
                    media_type=resolved.artifact.media_type,
                    content=resolved.content,
                )
            )
        return tuple(images)


def _schema_name(schema_id: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]", "_", schema_id)[:64]
    if not name:
        raise ValueError("structured output schema ID cannot form a provider name")
    return name


def _check_constraints(context: RunContext) -> None:
    context.cancellation.raise_if_cancelled()
    if context.deadline is not None:
        context.deadline.raise_if_expired(context.clock)


async def _call_with_constraints(
    client: OpenAIResponsesClient,
    request: OpenAIResponsesRequest,
    context: RunContext,
) -> OpenAIResponsesResult:
    call_task = asyncio.create_task(client.create(request))
    cancellation_task = asyncio.create_task(context.cancellation.wait_cancelled())
    timeout: float | None = None
    if context.deadline is not None:
        timeout = context.deadline.remaining_seconds(context.clock)
    try:
        done, _ = await asyncio.wait(
            {call_task, cancellation_task},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if call_task in done:
            result = call_task.result()
            if not isinstance(result, OpenAIResponsesResult):
                raise TypeError("OpenAI Responses client returned an invalid result")
            return result
        call_task.cancel()
        await asyncio.gather(call_task, return_exceptions=True)
        if cancellation_task in done:
            context.cancellation.raise_if_cancelled()
        if context.deadline is not None:
            raise DeadlineExceeded(context.deadline)
        raise AssertionError("OpenAI Responses constraint wait ended unexpectedly")
    finally:
        cancellation_task.cancel()
        if not call_task.done():
            call_task.cancel()
        await asyncio.gather(call_task, cancellation_task, return_exceptions=True)


async def _resolve_with_constraints(
    resolver: ArtifactResolver,
    artifact: ArtifactRef,
    context: RunContext,
) -> ResolvedArtifact:
    resolution_task = asyncio.create_task(resolver.resolve(artifact))
    cancellation_task = asyncio.create_task(context.cancellation.wait_cancelled())
    timeout: float | None = None
    if context.deadline is not None:
        timeout = context.deadline.remaining_seconds(context.clock)
    try:
        done, _ = await asyncio.wait(
            {resolution_task, cancellation_task},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if resolution_task in done:
            return resolution_task.result()
        resolution_task.cancel()
        await asyncio.gather(resolution_task, return_exceptions=True)
        if cancellation_task in done:
            context.cancellation.raise_if_cancelled()
        if context.deadline is not None:
            raise DeadlineExceeded(context.deadline)
        raise AssertionError("artifact resolution constraint wait ended unexpectedly")
    finally:
        cancellation_task.cancel()
        if not resolution_task.done():
            resolution_task.cancel()
        await asyncio.gather(
            resolution_task,
            cancellation_task,
            return_exceptions=True,
        )
