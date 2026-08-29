"""Provider-neutral OpenAI image client seam and capability adapter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    DataRetention,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageInputRole,
    ImageUsage,
    ModelMetadata,
)
from agentrig.core._validation import require_trimmed_string
from agentrig.core.artifacts import ArtifactRef, ArtifactResolver
from agentrig.core.context import RunContext
from agentrig.core.errors import AgentRigError, Failure, FailureKind

OPENAI_IMAGE_SDK_VERSION = "2.47.0"
OPENAI_IMAGE_MAX_INPUTS = 16


class OpenAIImageOperation(StrEnum):
    GENERATE = "generate"
    EDIT = "edit"


@dataclass(frozen=True, slots=True, kw_only=True)
class OpenAIImageSource:
    role: ImageInputRole
    media_type: str
    content: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.role, ImageInputRole):
            raise TypeError("OpenAI image source role must be ImageInputRole")
        require_trimmed_string("OpenAI image source media type", self.media_type)
        if not self.media_type.partition(";")[0].startswith("image/"):
            raise ValueError("OpenAI image source must use an image media type")
        if not isinstance(self.content, bytes) or not self.content:
            raise ValueError("OpenAI image source content must be nonempty bytes")


@dataclass(frozen=True, slots=True, kw_only=True)
class OpenAIImageRequest:
    operation: OpenAIImageOperation
    model: str
    prompt: str = field(repr=False)
    width: int
    height: int
    output_media_type: str
    sources: tuple[OpenAIImageSource, ...] = field(default=(), repr=False)
    idempotency_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.operation, OpenAIImageOperation):
            raise TypeError("OpenAI image operation is invalid")
        require_trimmed_string("OpenAI image model", self.model)
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("OpenAI image prompt must contain text")
        for name, value in (("width", self.width), ("height", self.height)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"OpenAI image {name} must be positive")
        require_trimmed_string(
            "OpenAI image output media type", self.output_media_type
        )
        sources = tuple(self.sources)
        if any(not isinstance(item, OpenAIImageSource) for item in sources):
            raise TypeError("OpenAI image sources must contain source values")
        if len(sources) > OPENAI_IMAGE_MAX_INPUTS:
            raise ValueError("OpenAI image input count exceeds the portable limit")
        roles = tuple(item.role for item in sources)
        if self.operation is OpenAIImageOperation.GENERATE and sources:
            raise ValueError("OpenAI image generation cannot contain edit sources")
        if self.operation is OpenAIImageOperation.EDIT:
            if roles.count(ImageInputRole.EDIT_BASE) != 1:
                raise ValueError("OpenAI image edit requires exactly one edit base")
            if roles.count(ImageInputRole.EDIT_MASK) > 1:
                raise ValueError("OpenAI image edit accepts at most one edit mask")
        object.__setattr__(self, "sources", sources)
        if self.idempotency_key is not None:
            require_trimmed_string(
                "OpenAI image idempotency key", self.idempotency_key
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class OpenAIImageResult:
    content: bytes = field(repr=False)
    media_type: str
    model: str
    usage: ImageUsage = ImageUsage()

    def __post_init__(self) -> None:
        if not isinstance(self.content, bytes) or not self.content:
            raise ValueError("OpenAI image result content must be nonempty bytes")
        require_trimmed_string("OpenAI image result media type", self.media_type)
        if not self.media_type.partition(";")[0].startswith("image/"):
            raise ValueError("OpenAI image result must use an image media type")
        require_trimmed_string("OpenAI image result model", self.model)
        if not isinstance(self.usage, ImageUsage):
            raise TypeError("OpenAI image result usage must be ImageUsage")


@runtime_checkable
class OpenAIImageClient(Protocol):
    async def create(self, request: OpenAIImageRequest) -> OpenAIImageResult: ...
    async def close(self) -> None: ...


@runtime_checkable
class OpenAIImageClientFactory(Protocol):
    def create(self) -> OpenAIImageClient: ...


@runtime_checkable
class ImageArtifactPublisher(Protocol):
    async def publish(
        self,
        *,
        request: ImageGenerationRequest,
        content: bytes,
        media_type: str,
        context: RunContext,
    ) -> ArtifactRef: ...


OPENAI_IMAGE_CAPABILITY = CapabilityDescriptor(
    capability_id="openai.image",
    version="1",
    kind=CapabilityKind.IMAGE_GENERATION,
    features=frozenset(
        {
            CapabilityFeature.REFERENCE_IMAGES,
            CapabilityFeature.IMAGE_EDITING,
            CapabilityFeature.MASKS,
            CapabilityFeature.IDEMPOTENCY_KEYS,
        }
    ),
    limits={
        CapabilityLimit.MAX_IMAGE_INPUTS: OPENAI_IMAGE_MAX_INPUTS,
        CapabilityLimit.MAX_REFERENCE_IMAGES: OPENAI_IMAGE_MAX_INPUTS,
    },
    data_retention=DataRetention.PROVIDER_MANAGED,
)


class OpenAIImageGenerator:
    """Resolve portable inputs and invoke one injected OpenAI image client."""

    descriptor = OPENAI_IMAGE_CAPABILITY

    def __init__(
        self,
        *,
        client_factory: OpenAIImageClientFactory,
        artifact_resolver: ArtifactResolver,
        artifact_publisher: ImageArtifactPublisher,
        model: str,
    ) -> None:
        if not isinstance(client_factory, OpenAIImageClientFactory):
            raise TypeError("OpenAI image client factory is invalid")
        if not isinstance(artifact_resolver, ArtifactResolver):
            raise TypeError("OpenAI image artifact resolver is invalid")
        if not isinstance(artifact_publisher, ImageArtifactPublisher):
            raise TypeError("OpenAI image artifact publisher is invalid")
        require_trimmed_string("OpenAI image generator model", model)
        self._client_factory = client_factory
        self._resolver = artifact_resolver
        self._publisher = artifact_publisher
        self._model = model

    async def generate(
        self,
        request: ImageGenerationRequest,
        context: RunContext,
    ) -> ImageGenerationResult:
        if not isinstance(request, ImageGenerationRequest):
            raise TypeError("OpenAI image request must be ImageGenerationRequest")
        if not isinstance(context, RunContext):
            raise TypeError("OpenAI image context must be RunContext")
        request.require_supported_by(self.descriptor)
        _check_constraints(context)
        sources = await self._resolve_sources(request, context)
        operation = (
            OpenAIImageOperation.EDIT if sources else OpenAIImageOperation.GENERATE
        )
        translated = OpenAIImageRequest(
            operation=operation,
            model=self._model,
            prompt=request.specification.prompt,
            width=request.specification.size.width,
            height=request.specification.size.height,
            output_media_type=request.specification.output_media_type,
            sources=sources,
            idempotency_key=request.idempotency_key,
        )
        client = None
        try:
            _check_constraints(context)
            client = self._client_factory.create()
            result = await client.create(translated)
        except AgentRigError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception:
            raise AgentRigError(
                Failure(
                    kind=FailureKind.TRANSIENT_PROVIDER,
                    message="OpenAI image request failed",
                    code="openai.image.request_failed",
                )
            ) from None
        finally:
            if client is not None:
                try:
                    await client.close()
                except Exception:
                    pass
        _check_constraints(context)
        if result.media_type != request.specification.output_media_type:
            raise AgentRigError(
                Failure(
                    kind=FailureKind.PERMANENT_PROVIDER,
                    message="OpenAI image result media type was unexpected",
                    code="openai.image.media_type_mismatch",
                )
            )
        try:
            artifact = await self._publisher.publish(
                request=request,
                content=result.content,
                media_type=result.media_type,
                context=context,
            )
        except AgentRigError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception:
            raise AgentRigError(
                Failure(
                    kind=FailureKind.UNEXPECTED,
                    message="OpenAI image artifact could not be published",
                    code="openai.image.publish_failed",
                )
            ) from None
        return ImageGenerationResult(
            request=request,
            image=artifact,
            model=ModelMetadata(provider="openai", model_id=result.model),
            usage=result.usage,
        )

    async def _resolve_sources(
        self, request: ImageGenerationRequest, context: RunContext
    ) -> tuple[OpenAIImageSource, ...]:
        role_artifacts = _role_artifacts(request)
        resolved: list[OpenAIImageSource] = []
        for role, artifact in role_artifacts:
            _check_constraints(context)
            item = await self._resolver.resolve(artifact)
            if item.artifact != artifact:
                raise AgentRigError(
                    Failure(
                        kind=FailureKind.INVALID_INPUT,
                        message="resolved image artifact identity changed",
                        code="openai.image.resolved_identity_mismatch",
                    )
                )
            resolved.append(
                OpenAIImageSource(
                    role=role,
                    media_type=artifact.media_type,
                    content=item.content,
                )
            )
        return tuple(resolved)


def _role_artifacts(
    request: ImageGenerationRequest,
) -> tuple[tuple[ImageInputRole, ArtifactRef], ...]:
    if request.inputs:
        roles = tuple(item.role for item in request.inputs)
        if ImageInputRole.EDIT_BASE not in roles:
            raise AgentRigError(
                Failure(
                    kind=FailureKind.INVALID_INPUT,
                    message="OpenAI image inputs require an explicit edit base",
                    code="openai.image.edit_base_required",
                )
            )
        return tuple((item.role, item.artifact) for item in request.inputs)
    if not request.reference_images:
        return ()
    result: list[tuple[ImageInputRole, ArtifactRef]] = [
        (ImageInputRole.EDIT_BASE, request.reference_images[0])
    ]
    result.extend(
        (ImageInputRole.COMPOSITION_REFERENCE, item)
        for item in request.reference_images[1:]
    )
    if request.mask is not None:
        result.append((ImageInputRole.EDIT_MASK, request.mask))
    return tuple(result)


def _check_constraints(context: RunContext) -> None:
    context.cancellation.raise_if_cancelled()
    if context.deadline is not None:
        context.deadline.raise_if_expired(context.clock)
