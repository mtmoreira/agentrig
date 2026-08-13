"""Portable capability identity, feature, limit, and retention contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from agentrig.core._validation import require_trimmed_string


class CapabilityKind(StrEnum):
    """Stable categories of provider-independent capabilities."""

    TEXT_GENERATION = "text_generation"
    STRUCTURED_GENERATION = "structured_generation"
    CODING = "coding"
    IMAGE_GENERATION = "image_generation"
    SEARCH = "search"
    RETRIEVAL = "retrieval"
    TOOL = "tool"


class CapabilityFeature(StrEnum):
    """Portable optional features advertised by capability implementations."""

    MESSAGE_INPUT = "message_input"
    MULTIMODAL_INPUT = "multimodal_input"
    STREAMING = "streaming"
    CANCELLATION = "cancellation"
    STRUCTURED_OUTPUT = "structured_output"
    SESSION_CONTINUATION = "session_continuation"
    APPROVAL_REQUESTS = "approval_requests"
    TOOL_USE = "tool_use"
    REFERENCE_IMAGES = "reference_images"
    MASKS = "masks"
    REGIONS = "regions"
    CITATIONS = "citations"
    IDEMPOTENCY_KEYS = "idempotency_keys"


class CapabilityLimit(StrEnum):
    """Portable bounded quantities that requirements can compare."""

    MAX_INPUT_ARTIFACTS = "max_input_artifacts"
    MAX_CHANGED_FILES = "max_changed_files"
    MAX_OUTPUT_TOKENS = "max_output_tokens"
    MAX_REFERENCE_IMAGES = "max_reference_images"
    MAX_RESULTS = "max_results"
    MAX_TOOL_CALLS = "max_tool_calls"


class DataRetention(StrEnum):
    """Implementation-declared handling of submitted capability inputs."""

    NOT_RETAINED = "not_retained"
    TRANSIENT = "transient"
    PROVIDER_MANAGED = "provider_managed"
    UNKNOWN = "unknown"


_ALL_RETENTION_POLICIES = frozenset(DataRetention)


@dataclass(frozen=True, slots=True, kw_only=True)
class CapabilityDescriptor:
    """Stable implementation identity and portable support characteristics."""

    capability_id: str
    version: str
    kind: CapabilityKind
    features: frozenset[CapabilityFeature] = frozenset()
    limits: Mapping[CapabilityLimit, int] = field(default_factory=dict)
    data_retention: DataRetention = DataRetention.UNKNOWN

    def __post_init__(self) -> None:
        require_trimmed_string("capability ID", self.capability_id)
        require_trimmed_string("capability version", self.version)
        if not isinstance(self.kind, CapabilityKind):
            raise TypeError("capability kind must be a CapabilityKind")
        object.__setattr__(
            self,
            "features",
            _freeze_features("capability features", self.features),
        )
        object.__setattr__(
            self,
            "limits",
            _freeze_limits("capability limits", self.limits),
        )
        if not isinstance(self.data_retention, DataRetention):
            raise TypeError(
                "capability data_retention must be a DataRetention"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class CapabilityRequirements:
    """Portable requirements checked before invoking one implementation."""

    kind: CapabilityKind
    features: frozenset[CapabilityFeature] = frozenset()
    minimum_limits: Mapping[CapabilityLimit, int] = field(default_factory=dict)
    allowed_data_retention: frozenset[DataRetention] = _ALL_RETENTION_POLICIES

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CapabilityKind):
            raise TypeError("capability requirement kind must be a CapabilityKind")
        object.__setattr__(
            self,
            "features",
            _freeze_features("required capability features", self.features),
        )
        object.__setattr__(
            self,
            "minimum_limits",
            _freeze_limits(
                "required capability limits",
                self.minimum_limits,
            ),
        )
        allowed_retention = _freeze_retention(
            self.allowed_data_retention,
        )
        if not allowed_retention:
            raise ValueError(
                "allowed capability data retention must not be empty"
            )
        object.__setattr__(
            self,
            "allowed_data_retention",
            allowed_retention,
        )

    def unmet_by(self, descriptor: CapabilityDescriptor) -> tuple[str, ...]:
        """Return stable reasons this descriptor cannot satisfy the request."""
        if not isinstance(descriptor, CapabilityDescriptor):
            raise TypeError(
                "capability requirements must check a CapabilityDescriptor"
            )
        unmet: list[str] = []
        if descriptor.kind is not self.kind:
            unmet.append(f"kind:{self.kind.value}")

        missing_features = self.features - descriptor.features
        unmet.extend(
            f"feature:{feature.value}"
            for feature in sorted(missing_features, key=lambda item: item.value)
        )

        for limit, minimum in sorted(
            self.minimum_limits.items(),
            key=lambda item: item[0].value,
        ):
            available = descriptor.limits.get(limit)
            if available is None or available < minimum:
                unmet.append(f"limit:{limit.value}>={minimum}")

        if descriptor.data_retention not in self.allowed_data_retention:
            unmet.append(
                f"data_retention:{descriptor.data_retention.value}"
            )
        return tuple(unmet)

    def require(self, descriptor: CapabilityDescriptor) -> None:
        """Raise before execution when portable requirements are not met."""
        unmet = self.unmet_by(descriptor)
        if unmet:
            raise ValueError(
                "capability does not satisfy requirements: " + ", ".join(unmet)
            )


def _freeze_features(
    field_name: str,
    features: Iterable[CapabilityFeature],
) -> frozenset[CapabilityFeature]:
    copied = tuple(features)
    if any(not isinstance(item, CapabilityFeature) for item in copied):
        raise TypeError(f"{field_name} must contain CapabilityFeature values")
    if len(copied) != len(set(copied)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return frozenset(copied)


def _freeze_limits(
    field_name: str,
    limits: Mapping[CapabilityLimit, int],
) -> Mapping[CapabilityLimit, int]:
    copied: dict[CapabilityLimit, int] = {}
    for limit, value in limits.items():
        if not isinstance(limit, CapabilityLimit):
            raise TypeError(f"{field_name} keys must be CapabilityLimit values")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} values must be positive integers")
        copied[limit] = value
    return MappingProxyType(copied)


def _freeze_retention(
    values: Iterable[DataRetention],
) -> frozenset[DataRetention]:
    copied = tuple(values)
    if any(not isinstance(item, DataRetention) for item in copied):
        raise TypeError(
            "allowed capability data retention must contain DataRetention values"
        )
    if len(copied) != len(set(copied)):
        raise ValueError(
            "allowed capability data retention must not contain duplicates"
        )
    return frozenset(copied)
