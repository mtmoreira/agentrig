"""Provider-neutral ComfyUI image client seam and capability adapter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import Protocol, runtime_checkable

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    DataRetention,
    ImageGenerationRequest,
    ImageArtifactPublisher,
    ImageGenerationResult,
    ImageInputFidelity,
    ImageInputRole,
    ImageOperation,
    ImageQuality,
    ImageUsage,
    ModelMetadata,
)
from agentrig.core import ArtifactRef, ArtifactResolver, ResolvedArtifact, RunContext
from agentrig.core._validation import require_trimmed_string
from agentrig.core.errors import AgentRigError, Failure, FailureKind

COMFYUI_IMAGE_MAX_INPUTS = 16


class ComfyUIImageOperation(StrEnum):
    GENERATE = "generate"
    REFERENCE_GENERATE = "reference_generate"
    EDIT = "edit"


@dataclass(frozen=True, slots=True, kw_only=True)
class ComfyUIImageSource:
    role: ImageInputRole
    media_type: str
    content: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.role, ImageInputRole):
            raise TypeError("ComfyUI image source role must be ImageInputRole")
        require_trimmed_string("ComfyUI image source media type", self.media_type)
        if not self.media_type.partition(";")[0].startswith("image/"):
            raise ValueError("ComfyUI image source must use an image media type")
        if not isinstance(self.content, bytes) or not self.content:
            raise ValueError("ComfyUI image source content must be nonempty bytes")


@dataclass(frozen=True, slots=True, kw_only=True)
class ComfyUIImageRequest:
    operation: ComfyUIImageOperation
    model: str
    prompt: str = field(repr=False)
    width: int
    height: int
    output_media_type: str
    quality: ImageQuality = ImageQuality.AUTO
    input_fidelity: ImageInputFidelity | None = None
    sources: tuple[ComfyUIImageSource, ...] = field(default=(), repr=False)
    idempotency_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.operation, ComfyUIImageOperation):
            raise TypeError("ComfyUI image operation is invalid")
        require_trimmed_string("ComfyUI image model", self.model)
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("ComfyUI image prompt must contain text")
        for name, value in (("width", self.width), ("height", self.height)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"ComfyUI image {name} must be positive")
        require_trimmed_string("ComfyUI output media type", self.output_media_type)
        if self.output_media_type != "image/png":
            raise ValueError("ComfyUI image output must be image/png")
        if not isinstance(self.quality, ImageQuality):
            raise TypeError("ComfyUI image quality is invalid")
        if self.input_fidelity is not None and not isinstance(
            self.input_fidelity, ImageInputFidelity
        ):
            raise TypeError("ComfyUI image input fidelity is invalid")
        if self.operation is ComfyUIImageOperation.GENERATE and self.input_fidelity is not None:
            raise ValueError("ComfyUI source-free generation cannot set input fidelity")
        sources = tuple(self.sources)
        if any(not isinstance(item, ComfyUIImageSource) for item in sources):
            raise TypeError("ComfyUI image sources must contain source values")
        if len(sources) > COMFYUI_IMAGE_MAX_INPUTS:
            raise ValueError("ComfyUI image input count exceeds the portable limit")
        roles = tuple(item.role for item in sources)
        if self.operation is ComfyUIImageOperation.GENERATE and sources:
            raise ValueError("ComfyUI source-free generation cannot contain inputs")
        if self.operation is ComfyUIImageOperation.REFERENCE_GENERATE:
            if not sources or any(
                role in (ImageInputRole.EDIT_BASE, ImageInputRole.EDIT_MASK)
                for role in roles
            ):
                raise ValueError("ComfyUI reference generation requires references")
        if self.operation is ComfyUIImageOperation.EDIT:
            if roles.count(ImageInputRole.EDIT_BASE) != 1:
                raise ValueError("ComfyUI editing requires exactly one edit base")
            if roles.count(ImageInputRole.EDIT_MASK) > 1:
                raise ValueError("ComfyUI editing accepts at most one edit mask")
        object.__setattr__(self, "sources", sources)
        if self.idempotency_key is not None:
            require_trimmed_string("ComfyUI image idempotency key", self.idempotency_key)


@dataclass(frozen=True, slots=True, kw_only=True)
class ComfyUIImageResult:
    content: bytes = field(repr=False)
    media_type: str
    model: str
    workflow_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, bytes) or not self.content:
            raise ValueError("ComfyUI image result content must be nonempty bytes")
        if self.media_type != "image/png":
            raise ValueError("ComfyUI image result must be image/png")
        require_trimmed_string("ComfyUI image result model", self.model)
        if self.workflow_id is not None:
            require_trimmed_string("ComfyUI workflow ID", self.workflow_id)


@runtime_checkable
class ComfyUIImageClient(Protocol):
    async def create(self, request: ComfyUIImageRequest) -> ComfyUIImageResult: ...
    async def close(self) -> None: ...


@runtime_checkable
class ComfyUIImageClientFactory(Protocol):
    def create(self) -> ComfyUIImageClient: ...


COMFYUI_IMAGE_CAPABILITY = CapabilityDescriptor(
    capability_id="comfyui.image",
    version="1",
    kind=CapabilityKind.IMAGE_GENERATION,
    features=frozenset({
        CapabilityFeature.REFERENCE_IMAGES,
        CapabilityFeature.IMAGE_EDITING,
        CapabilityFeature.IMAGE_MODEL_SELECTION,
        CapabilityFeature.IMAGE_QUALITY,
        CapabilityFeature.IMAGE_INPUT_FIDELITY,
        CapabilityFeature.IDEMPOTENCY_KEYS,
    }),
    limits={
        CapabilityLimit.MAX_IMAGE_INPUTS: COMFYUI_IMAGE_MAX_INPUTS,
        CapabilityLimit.MAX_REFERENCE_IMAGES: COMFYUI_IMAGE_MAX_INPUTS,
    },
    # ComfyUI owns its input/output directories; cleanup depends on the
    # caller-managed server and workflow installation.
    data_retention=DataRetention.UNKNOWN,
)


class ComfyUIImageGenerator:
    """Resolve portable image inputs and invoke one injected ComfyUI client."""

    descriptor = COMFYUI_IMAGE_CAPABILITY

    def __init__(
        self,
        *,
        client_factory: ComfyUIImageClientFactory,
        artifact_resolver: ArtifactResolver,
        artifact_publisher: ImageArtifactPublisher,
        model: str,
    ) -> None:
        if not isinstance(client_factory, ComfyUIImageClientFactory):
            raise TypeError("ComfyUI image client factory is invalid")
        if not isinstance(artifact_resolver, ArtifactResolver):
            raise TypeError("ComfyUI image artifact resolver is invalid")
        if not isinstance(artifact_publisher, ImageArtifactPublisher):
            raise TypeError("ComfyUI image artifact publisher is invalid")
        require_trimmed_string("ComfyUI image generator model", model)
        self._client_factory = client_factory
        self._resolver = artifact_resolver
        self._publisher = artifact_publisher
        self._model = model

    async def generate(
        self, request: ImageGenerationRequest, context: RunContext
    ) -> ImageGenerationResult:
        if not isinstance(request, ImageGenerationRequest):
            raise TypeError("ComfyUI image request must be ImageGenerationRequest")
        if not isinstance(context, RunContext):
            raise TypeError("ComfyUI image context must be RunContext")
        request.require_supported_by(self.descriptor)
        _check_constraints(context)
        sources = await self._resolve_sources(request, context)
        settings = request.settings
        operation = (
            ComfyUIImageOperation(settings.operation.value)
            if settings is not None
            else (ComfyUIImageOperation.EDIT if sources else ComfyUIImageOperation.GENERATE)
        )
        translated = ComfyUIImageRequest(
            operation=operation,
            model=settings.model if settings is not None else self._model,
            prompt=_role_bound_prompt(request.specification.prompt, sources),
            width=request.specification.size.width,
            height=request.specification.size.height,
            output_media_type=request.specification.output_media_type,
            quality=settings.quality if settings is not None else ImageQuality.AUTO,
            input_fidelity=settings.input_fidelity if settings is not None else None,
            sources=sources,
            idempotency_key=request.idempotency_key,
        )
        capture = context.private_trace_capture
        if capture is not None:
            capture.record(context, kind="provider.request", metadata={
                "provider": "comfyui", "model": translated.model,
            }, content={
                "operation": translated.operation.value,
                "prompt": translated.prompt,
                "width": translated.width, "height": translated.height,
                "quality": translated.quality.value,
                "source_roles": [source.role.value for source in translated.sources],
                "source_sha256": [sha256(source.content).hexdigest()
                                  for source in translated.sources],
            })
        client: ComfyUIImageClient | None = None
        try:
            client = self._client_factory.create()
            result = await client.create(translated)
            if capture is not None:
                capture.record(context, kind="provider.response", metadata={
                    "provider": "comfyui", "model": result.model,
                }, content={
                    "workflow_id": result.workflow_id,
                    "media_type": result.media_type,
                    "output_sha256": sha256(result.content).hexdigest(),
                })
        except AgentRigError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise AgentRigError(Failure(
                kind=FailureKind.TRANSIENT_PROVIDER,
                message="ComfyUI image request failed",
                code="comfyui.image.request_failed",
                metadata={"exception_type": type(error).__qualname__},
            )) from None
        finally:
            if client is not None:
                try:
                    await client.close()
                except Exception:
                    pass
        _check_constraints(context)
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
            raise AgentRigError(Failure(
                kind=FailureKind.UNEXPECTED,
                message="ComfyUI image artifact could not be published",
                code="comfyui.image.publish_failed",
            )) from None
        return ImageGenerationResult(
            request=request,
            image=artifact,
            model=ModelMetadata(provider="comfyui", model_id=result.model),
            usage=ImageUsage(cost=0.0, currency="USD", input_images=len(sources), output_images=1),
        )

    async def _resolve_sources(
        self, request: ImageGenerationRequest, context: RunContext
    ) -> tuple[ComfyUIImageSource, ...]:
        pairs = _role_artifacts(request)
        result: list[ComfyUIImageSource] = []
        for role, artifact in pairs:
            _check_constraints(context)
            resolved = await self._resolver.resolve(artifact)
            if not isinstance(resolved, ResolvedArtifact) or resolved.artifact != artifact:
                raise AgentRigError(Failure(
                    kind=FailureKind.INVALID_INPUT,
                    message="resolved image artifact identity changed",
                    code="comfyui.image.resolved_identity_mismatch",
                ))
            result.append(ComfyUIImageSource(
                role=role,
                media_type=artifact.media_type,
                content=resolved.content,
            ))
        return tuple(result)


def _role_artifacts(request: ImageGenerationRequest) -> tuple[tuple[ImageInputRole, ArtifactRef], ...]:
    if request.inputs:
        return tuple((item.role, item.artifact) for item in request.inputs)
    if not request.reference_images:
        return ()
    result = [(ImageInputRole.EDIT_BASE, request.reference_images[0])]
    result.extend(
        (ImageInputRole.COMPOSITION_REFERENCE, item)
        for item in request.reference_images[1:]
    )
    if request.mask is not None:
        result.append((ImageInputRole.EDIT_MASK, request.mask))
    return tuple(result)


def _role_bound_prompt(prompt: str, sources: tuple[ComfyUIImageSource, ...]) -> str:
    if not sources:
        return prompt
    declarations = "\n".join(
        f"{index}. {source.role.value}" for index, source in enumerate(sources, 1)
    )
    return (
        "Ordered source-image roles (the local workflow receives images in this exact order):\n"
        + declarations + "\n\n" + prompt
    )


def _check_constraints(context: RunContext) -> None:
    try:
        context.cancellation.raise_if_cancelled()
        if context.deadline is not None:
            context.deadline.raise_if_expired(context.clock)
    except AgentRigError:
        raise
    except Exception as error:
        from agentrig.core import normalize_exception
        raise AgentRigError(normalize_exception(error)) from None
