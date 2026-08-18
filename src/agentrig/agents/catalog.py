"""Immutable application-scoped autonomous-runtime registrations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from agentrig.agents.runtime import AgentRuntime
from agentrig.capabilities.base import (
    CapabilityDescriptor,
    CapabilityKind,
    CapabilityRequirements,
)
from agentrig.core._validation import require_trimmed_string
from agentrig.core.errors import AgentRigError, Failure, FailureKind


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentRuntimeRegistration:
    """Bind one safe application identity to one portable runtime."""

    binding_id: str
    descriptor: CapabilityDescriptor
    runtime: AgentRuntime = field(repr=False)
    enabled: bool = True

    def __post_init__(self) -> None:
        require_trimmed_string("agent runtime binding ID", self.binding_id)
        if not isinstance(self.descriptor, CapabilityDescriptor):
            raise TypeError(
                "agent runtime registration descriptor must be a "
                "CapabilityDescriptor"
            )
        if self.descriptor.kind is not CapabilityKind.AGENT_RUNTIME:
            raise ValueError(
                "agent runtime registration requires an agent-runtime "
                "capability"
            )
        if not isinstance(self.runtime, AgentRuntime):
            raise TypeError(
                "agent runtime registration runtime must be an AgentRuntime"
            )
        if not isinstance(self.enabled, bool):
            raise TypeError("agent runtime registration enabled must be a bool")


@dataclass(frozen=True, slots=True, init=False)
class AgentRuntimeCatalog:
    """Resolve exact application bindings without provider initialization."""

    _registrations: tuple[AgentRuntimeRegistration, ...] = field(repr=False)
    _by_binding_id: Mapping[str, AgentRuntimeRegistration] = field(repr=False)

    def __init__(
        self,
        registrations: Iterable[AgentRuntimeRegistration],
    ) -> None:
        copied = tuple(registrations)
        by_binding_id: dict[str, AgentRuntimeRegistration] = {}
        for registration in copied:
            if not isinstance(registration, AgentRuntimeRegistration):
                raise TypeError(
                    "agent runtime catalog requires "
                    "AgentRuntimeRegistration values"
                )
            if registration.binding_id in by_binding_id:
                raise ValueError(
                    "agent runtime catalog binding IDs must be unique"
                )
            by_binding_id[registration.binding_id] = registration

        object.__setattr__(self, "_registrations", copied)
        object.__setattr__(
            self,
            "_by_binding_id",
            MappingProxyType(by_binding_id),
        )

    @property
    def registrations(self) -> tuple[AgentRuntimeRegistration, ...]:
        """Return registrations in deterministic construction order."""
        return self._registrations

    @property
    def binding_ids(self) -> tuple[str, ...]:
        """Return binding identities in deterministic construction order."""
        return tuple(
            registration.binding_id for registration in self._registrations
        )

    def resolve(
        self,
        binding_id: str,
        requirements: CapabilityRequirements,
    ) -> AgentRuntime:
        """Return one exact compatible runtime or raise a normalized failure."""
        require_trimmed_string("agent runtime binding ID", binding_id)
        if not isinstance(requirements, CapabilityRequirements):
            raise TypeError(
                "agent runtime requirements must be CapabilityRequirements"
            )
        if requirements.kind is not CapabilityKind.AGENT_RUNTIME:
            raise ValueError(
                "agent runtime resolution requires agent-runtime requirements"
            )

        registration = self._by_binding_id.get(binding_id)
        if registration is None:
            raise _resolution_error(
                code="agent_runtime.binding_unknown",
                message="agent runtime binding is unavailable",
            )
        if not registration.enabled:
            raise _resolution_error(
                code="agent_runtime.binding_disabled",
                message="agent runtime binding is unavailable",
            )

        unmet = requirements.unmet_by(registration.descriptor)
        if unmet:
            raise _resolution_error(
                code="agent_runtime.binding_incompatible",
                message="agent runtime binding is incompatible",
                metadata={"unmet_requirements": ",".join(unmet)},
            )
        return registration.runtime


def _resolution_error(
    *,
    code: str,
    message: str,
    metadata: Mapping[str, str] | None = None,
) -> AgentRigError:
    return AgentRigError(
        Failure(
            kind=FailureKind.INVALID_INPUT,
            message=message,
            code=code,
            metadata=metadata if metadata is not None else {},
        )
    )
