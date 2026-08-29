"""Provider-independent image-generation and editing contracts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
import math
from typing import Protocol, runtime_checkable

from agentrig.capabilities.base import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    CapabilityRequirements,
)
from agentrig.capabilities.text_generation import ModelMetadata
from agentrig.core._validation import require_trimmed_string
from agentrig.core.artifacts import ArtifactId, ArtifactRef
from agentrig.core.context import RunContext


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageSize:
    """Positive pixel dimensions for one generated image canvas."""

    width: int
    height: int

    def __post_init__(self) -> None:
        _require_positive_integer("image width", self.width)
        _require_positive_integer("image height", self.height)


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageRegion:
    """One rectangular region in output-canvas pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        _require_non_negative_integer("image region x", self.x)
        _require_non_negative_integer("image region y", self.y)
        _require_positive_integer("image region width", self.width)
        _require_positive_integer("image region height", self.height)

    def fits_within(self, size: ImageSize) -> bool:
        """Whether this rectangle is wholly inside the given image canvas."""
        if not isinstance(size, ImageSize):
            raise TypeError("image region size must be an ImageSize")
        return (
            self.x + self.width <= size.width
            and self.y + self.height <= size.height
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageSpecification:
    """Private creative direction and required generated image shape."""

    prompt: str = field(repr=False)
    size: ImageSize
    output_media_type: str = "image/png"

    def __post_init__(self) -> None:
        _require_content_text("image generation prompt", self.prompt)
        if not isinstance(self.size, ImageSize):
            raise TypeError("image specification size must be an ImageSize")
        _require_image_media_type(
            "image specification output_media_type",
            self.output_media_type,
        )


class ImageInputRole(StrEnum):
    """Portable semantic role of one source image."""

    EDIT_BASE = "edit_base"
    EDIT_MASK = "edit_mask"
    IDENTITY_REFERENCE = "identity_reference"
    STYLE_REFERENCE = "style_reference"
    COMPOSITION_REFERENCE = "composition_reference"


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageInput:
    """One explicitly role-bound image input."""

    role: ImageInputRole
    artifact: ArtifactRef = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.role, ImageInputRole):
            raise TypeError("image input role must be an ImageInputRole")
        if not isinstance(self.artifact, ArtifactRef):
            raise TypeError("image input artifact must be an ArtifactRef")
        _require_image_artifact("image input artifact", self.artifact)


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageUsage:
    """Portable image usage; unavailable provider values remain unknown."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    input_images: int | None = None
    output_images: int | None = None
    cost: float | None = None
    currency: str | None = None

    def __post_init__(self) -> None:
        _require_optional_count("image usage input_tokens", self.input_tokens)
        _require_optional_count("image usage output_tokens", self.output_tokens)
        _require_optional_count("image usage input_images", self.input_images)
        _require_optional_count("image usage output_images", self.output_images)
        if self.cost is not None:
            if (
                isinstance(self.cost, bool)
                or not isinstance(self.cost, (int, float))
                or not math.isfinite(self.cost)
                or self.cost < 0
            ):
                raise ValueError("image usage cost must be finite and non-negative")
        if (self.cost is None) != (self.currency is None):
            raise ValueError("image usage cost and currency must be reported together")
        if self.currency is not None:
            require_trimmed_string("image usage currency", self.currency)
            if len(self.currency) != 3 or self.currency.upper() != self.currency:
                raise ValueError("image usage currency must be three uppercase letters")

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageGenerationRequest:
    """A generation specification with optional reference and edit inputs."""

    specification: ImageSpecification = field(repr=False)
    inputs: tuple[ImageInput, ...] = field(default=(), repr=False)
    reference_images: tuple[ArtifactRef, ...] = field(
        default=(),
        repr=False,
    )
    mask: ArtifactRef | None = field(default=None, repr=False)
    regions: tuple[ImageRegion, ...] = ()
    idempotency_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.specification, ImageSpecification):
            raise TypeError(
                "image generation specification must be an ImageSpecification"
            )
        copied_inputs = tuple(self.inputs)
        if any(not isinstance(item, ImageInput) for item in copied_inputs):
            raise TypeError("image generation inputs must contain ImageInput values")
        object.__setattr__(self, "inputs", copied_inputs)

        copied_references = _copy_image_artifacts(
            "image generation reference_images",
            self.reference_images,
        )
        object.__setattr__(self, "reference_images", copied_references)

        if copied_inputs and (copied_references or self.mask is not None):
            raise ValueError(
                "explicit image inputs cannot be mixed with legacy references or mask"
            )

        if self.mask is not None:
            if not isinstance(self.mask, ArtifactRef):
                raise TypeError(
                    "image generation mask must be an ArtifactRef or None"
                )
            _require_image_artifact("image generation mask", self.mask)

        source_artifacts = tuple(item.artifact for item in copied_inputs) or (
            *copied_references,
            *((self.mask,) if self.mask is not None else ()),
        )
        _require_unique_artifacts(source_artifacts)

        roles = tuple(item.role for item in copied_inputs)
        if roles.count(ImageInputRole.EDIT_BASE) > 1:
            raise ValueError("image generation requires at most one edit base")
        if roles.count(ImageInputRole.EDIT_MASK) > 1:
            raise ValueError("image generation requires at most one edit mask")
        if ImageInputRole.EDIT_MASK in roles and ImageInputRole.EDIT_BASE not in roles:
            raise ValueError("image edit mask requires an explicit edit base")

        copied_regions = tuple(self.regions)
        if any(not isinstance(item, ImageRegion) for item in copied_regions):
            raise TypeError(
                "image generation regions must contain ImageRegion values"
            )
        if len(copied_regions) != len(set(copied_regions)):
            raise ValueError("image generation regions must not contain duplicates")
        if any(
            not region.fits_within(self.specification.size)
            for region in copied_regions
        ):
            raise ValueError(
                "image generation regions must fit within the output canvas"
            )
        object.__setattr__(self, "regions", copied_regions)

        has_edit_base = ImageInputRole.EDIT_BASE in roles
        if copied_regions and copied_inputs and not has_edit_base:
            raise ValueError("image generation regions require an explicit edit base")
        if (self.mask is not None or copied_regions) and not (
            copied_references or has_edit_base
        ):
            raise ValueError(
                "image generation masks and regions require a reference image"
            )
        if self.idempotency_key is not None:
            require_trimmed_string(
                "image generation idempotency key",
                self.idempotency_key,
            )

    @property
    def source_artifact_ids(self) -> tuple[ArtifactId, ...]:
        """Return the complete ordered lineage required of the output."""
        if self.inputs:
            return tuple(item.artifact.artifact_id for item in self.inputs)
        mask_ids = (
            (self.mask.artifact_id,) if self.mask is not None else ()
        )
        return tuple(
            artifact.artifact_id for artifact in self.reference_images
        ) + mask_ids

    @property
    def requirements(self) -> CapabilityRequirements:
        """Derive portable feature and reference-count requirements."""
        features: list[CapabilityFeature] = []
        limits: dict[CapabilityLimit, int] = {}
        if self.inputs:
            features.append(CapabilityFeature.REFERENCE_IMAGES)
            limits[CapabilityLimit.MAX_IMAGE_INPUTS] = len(self.inputs)
            if any(
                item.role in (ImageInputRole.EDIT_BASE, ImageInputRole.EDIT_MASK)
                for item in self.inputs
            ):
                features.append(CapabilityFeature.IMAGE_EDITING)
            if any(item.role is ImageInputRole.EDIT_MASK for item in self.inputs):
                features.append(CapabilityFeature.MASKS)
        if self.reference_images:
            features.append(CapabilityFeature.REFERENCE_IMAGES)
            limits[CapabilityLimit.MAX_REFERENCE_IMAGES] = len(
                self.reference_images
            )
        if self.mask is not None:
            features.append(CapabilityFeature.MASKS)
        if self.regions:
            features.append(CapabilityFeature.REGIONS)
        if self.idempotency_key is not None:
            features.append(CapabilityFeature.IDEMPOTENCY_KEYS)
        return CapabilityRequirements(
            kind=CapabilityKind.IMAGE_GENERATION,
            features=frozenset(features),
            minimum_limits=limits,
        )

    def require_supported_by(self, descriptor: CapabilityDescriptor) -> None:
        """Fail before provider execution if this request is unsupported."""
        self.requirements.require(descriptor)


@dataclass(frozen=True, slots=True, init=False)
class ImageGenerationResult:
    """One generated image artifact with verified source lineage."""

    image: ArtifactRef = field(repr=False)
    model: ModelMetadata
    usage: ImageUsage

    def __init__(
        self,
        *,
        request: ImageGenerationRequest,
        image: ArtifactRef,
        model: ModelMetadata,
        usage: ImageUsage | None = None,
    ) -> None:
        if not isinstance(request, ImageGenerationRequest):
            raise TypeError(
                "image generation result request must be an "
                "ImageGenerationRequest"
            )
        if not isinstance(image, ArtifactRef):
            raise TypeError(
                "image generation result image must be an ArtifactRef"
            )
        if image.kind != "image":
            raise ValueError(
                "image generation result artifact kind must be image"
            )
        _require_image_artifact("image generation result artifact", image)
        if image.media_type != request.specification.output_media_type:
            raise ValueError(
                "image generation result media type must match the specification"
            )
        if request.inputs and (
            image.input_artifact_ids != request.source_artifact_ids
        ):
            raise ValueError(
                "explicit image result lineage must exactly match ordered inputs"
            )
        missing_lineage = set(request.source_artifact_ids).difference(
            image.input_artifact_ids
        )
        if not request.inputs and missing_lineage:
            raise ValueError(
                "image generation result lineage must include every source "
                "artifact"
            )
        if not isinstance(model, ModelMetadata):
            raise TypeError(
                "image generation result model must be ModelMetadata"
            )
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "model", model)
        effective_usage = usage if usage is not None else ImageUsage()
        if not isinstance(effective_usage, ImageUsage):
            raise TypeError("image generation result usage must be ImageUsage")
        object.__setattr__(self, "usage", effective_usage)


@runtime_checkable
class ImageGenerator(Protocol):
    """Generate one image through a provider-independent implementation."""

    @property
    def descriptor(self) -> CapabilityDescriptor:
        """Return stable identity and supported optional features."""
        ...

    async def generate(
        self,
        request: ImageGenerationRequest,
        context: RunContext,
    ) -> ImageGenerationResult:
        """Generate one image result or raise a normalized failure."""
        ...


def _copy_image_artifacts(
    field_name: str,
    artifacts: Iterable[ArtifactRef],
) -> tuple[ArtifactRef, ...]:
    copied = tuple(artifacts)
    if any(not isinstance(item, ArtifactRef) for item in copied):
        raise TypeError(f"{field_name} must contain ArtifactRef values")
    for artifact in copied:
        _require_image_artifact(field_name, artifact)
    _require_unique_artifacts(copied)
    return copied


def _require_unique_artifacts(artifacts: Iterable[ArtifactRef]) -> None:
    artifact_ids = tuple(artifact.artifact_id for artifact in artifacts)
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError(
            "image generation source artifacts must have unique artifact IDs"
        )


def _require_image_artifact(field_name: str, artifact: ArtifactRef) -> None:
    if not artifact.media_type.partition(";")[0].startswith("image/"):
        raise ValueError(f"{field_name} must use an image media type")


def _require_image_media_type(field_name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    require_trimmed_string(field_name, value)
    essence = value.partition(";")[0]
    parts = essence.split("/")
    if (
        len(parts) != 2
        or parts[0] != "image"
        or not parts[1]
        or any(character.isspace() for character in essence)
    ):
        raise ValueError(f"{field_name} must contain an image media type")
    return value


def _require_positive_integer(field_name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_non_negative_integer(field_name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_optional_count(field_name: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer or None")


def _require_content_text(field_name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must contain non-whitespace text")
    return value
