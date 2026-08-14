"""Internal probes shared by portable capability contract suites."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityKind,
    CapabilityRequirements,
)
from agentrig.core.cancellation import RunCancelled
from agentrig.core.context import RunContext

InvocationCount = Callable[[], int]


def validate_contract_suite(
    *,
    label: str,
    descriptor: CapabilityDescriptor,
    expected_kind: CapabilityKind,
    supported_requirements: CapabilityRequirements,
    unsupported_requirements: CapabilityRequirements,
    context: RunContext,
    cancelled_context: RunContext,
    invocation_count: InvocationCount,
) -> None:
    """Validate fixtures before a reusable capability probe can run."""
    validate_contract_context(
        label=label,
        descriptor=descriptor,
        expected_kind=expected_kind,
        context=context,
        cancelled_context=cancelled_context,
        invocation_count=invocation_count,
    )
    if supported_requirements.unmet_by(descriptor):
        raise ValueError(
            f"{label} contract supported input is not supported"
        )
    if not unsupported_requirements.unmet_by(descriptor):
        raise ValueError(
            f"{label} contract unsupported input must be unsupported"
        )


def validate_contract_context(
    *,
    label: str,
    descriptor: CapabilityDescriptor,
    expected_kind: CapabilityKind,
    context: RunContext,
    cancelled_context: RunContext,
    invocation_count: InvocationCount,
) -> None:
    """Validate shared descriptor, context, and counter fixtures."""
    if not isinstance(descriptor, CapabilityDescriptor):
        raise TypeError(
            f"{label} contract descriptor must be a CapabilityDescriptor"
        )
    if descriptor.kind is not expected_kind:
        raise ValueError(
            f"{label} contract descriptor has the wrong capability kind"
        )
    if not isinstance(context, RunContext):
        raise TypeError(f"{label} contract context must be a RunContext")
    if not isinstance(cancelled_context, RunContext):
        raise TypeError(
            f"{label} contract cancelled_context must be a RunContext"
        )
    if not cancelled_context.cancellation.is_cancelled:
        raise ValueError(
            f"{label} contract cancelled_context must already be cancelled"
        )
    if not callable(invocation_count):
        raise TypeError(
            f"{label} contract invocation_count must be callable"
        )
    read_invocation_count(label, invocation_count)


async def verify_preflight_does_not_invoke(
    *,
    label: str,
    operation: Callable[[], Awaitable[object]],
    invocation_count: InvocationCount,
) -> None:
    """Require unsupported input to fail before the implementation boundary."""
    before = read_invocation_count(label, invocation_count)
    try:
        await operation()
    except ValueError:
        pass
    else:
        raise AssertionError(
            f"unsupported {label} input did not fail during preflight"
        )
    after = read_invocation_count(label, invocation_count)
    if after != before:
        raise AssertionError(
            f"unsupported {label} input invoked the implementation"
        )


async def verify_cancellation_does_not_invoke(
    *,
    label: str,
    operation: Callable[[], Awaitable[object]],
    invocation_count: InvocationCount,
) -> None:
    """Require an already-cancelled input to stop before implementation."""
    before = read_invocation_count(label, invocation_count)
    try:
        await operation()
    except RunCancelled:
        pass
    else:
        raise AssertionError(
            f"cancelled {label} input did not raise RunCancelled"
        )
    after = read_invocation_count(label, invocation_count)
    if after != before:
        raise AssertionError(
            f"cancelled {label} input invoked the implementation"
        )


def read_invocation_count(
    label: str,
    invocation_count: InvocationCount,
) -> int:
    """Read one non-negative implementation-boundary counter."""
    count = invocation_count()
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise TypeError(
            f"{label} contract invocation_count must return a "
            "non-negative integer"
        )
    return count
