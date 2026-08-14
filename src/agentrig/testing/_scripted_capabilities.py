"""Internal validation shared by scripted capability implementations."""

from __future__ import annotations

from agentrig.capabilities import CapabilityDescriptor, CapabilityKind
from agentrig.core.context import RunContext
from agentrig.core.errors import Failure, FailureKind


def require_descriptor_kind(
    descriptor: CapabilityDescriptor,
    kind: CapabilityKind,
    label: str,
) -> None:
    """Require one descriptor of the expected portable capability kind."""
    if not isinstance(descriptor, CapabilityDescriptor):
        raise TypeError(f"{label} descriptor must be a CapabilityDescriptor")
    if descriptor.kind is not kind:
        raise ValueError(f"{label} descriptor must use the {kind.value} kind")


def require_context(context: RunContext, label: str) -> None:
    """Require a real execution context before touching scripted state."""
    if not isinstance(context, RunContext):
        raise TypeError(f"{label} context must be a RunContext")


def check_constraints(context: RunContext) -> None:
    """Enforce cancellation and deadline before consuming scripted state."""
    context.cancellation.raise_if_cancelled()
    if context.deadline is not None:
        context.deadline.raise_if_expired(context.clock)


def exhaustion_failure(
    descriptor: CapabilityDescriptor,
    *,
    code: str,
    message: str,
) -> Failure:
    """Create one safe failure for a depleted scripted capability."""
    return Failure(
        kind=FailureKind.UNEXPECTED,
        message=message,
        code=code,
        metadata={
            "capability_id": descriptor.capability_id,
            "capability_version": descriptor.version,
        },
    )
