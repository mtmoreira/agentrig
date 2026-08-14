"""Reusable contract probes for portable generation implementations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityKind,
    CapabilityRequirements,
    StructuredGenerationRequest,
    StructuredGenerationResult,
    StructuredGenerator,
    TextGenerationRequest,
    TextGenerationResult,
    TextGenerator,
)
from agentrig.core.cancellation import RunCancelled
from agentrig.core.context import RunContext

OutputT = TypeVar("OutputT")
InvocationCount = Callable[[], int]


@dataclass(frozen=True, slots=True, kw_only=True)
class TextGeneratorContractSuite:
    """Portable checks shared by text fakes and injected-client integrations.

    ``invocation_count`` observes the implementation boundary so the suite can
    prove that unsupported and already-cancelled requests never reach it.
    """

    generator: TextGenerator = field(repr=False, compare=False)
    supported_request: TextGenerationRequest = field(repr=False)
    unsupported_request: TextGenerationRequest = field(repr=False)
    context: RunContext = field(repr=False)
    cancelled_context: RunContext = field(repr=False)
    invocation_count: InvocationCount = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.generator, TextGenerator):
            raise TypeError("text contract generator must satisfy TextGenerator")
        if not isinstance(self.supported_request, TextGenerationRequest):
            raise TypeError(
                "text contract supported_request must be a "
                "TextGenerationRequest"
            )
        if not isinstance(self.unsupported_request, TextGenerationRequest):
            raise TypeError(
                "text contract unsupported_request must be a "
                "TextGenerationRequest"
            )
        _validate_suite(
            descriptor=self.generator.descriptor,
            expected_kind=CapabilityKind.TEXT_GENERATION,
            supported_requirements=self.supported_request.requirements,
            unsupported_requirements=self.unsupported_request.requirements,
            context=self.context,
            cancelled_context=self.cancelled_context,
            invocation_count=self.invocation_count,
        )

    async def verify(self) -> TextGenerationResult:
        """Run the shared success, preflight, and cancellation checks."""
        result = await self.generator.generate(
            self.supported_request,
            self.context,
        )
        if not isinstance(result, TextGenerationResult):
            raise AssertionError(
                "text generator returned a non-TextGenerationResult value"
            )

        await _verify_preflight_does_not_invoke(
            operation=lambda: self.generator.generate(
                self.unsupported_request,
                self.context,
            ),
            invocation_count=self.invocation_count,
        )
        await _verify_cancellation_does_not_invoke(
            operation=lambda: self.generator.generate(
                self.supported_request,
                self.cancelled_context,
            ),
            invocation_count=self.invocation_count,
        )
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class StructuredGeneratorContractSuite(Generic[OutputT]):
    """Portable checks shared by structured generation implementations."""

    generator: StructuredGenerator[OutputT] = field(repr=False, compare=False)
    supported_request: StructuredGenerationRequest[OutputT] = field(repr=False)
    unsupported_request: StructuredGenerationRequest[OutputT] = field(
        repr=False
    )
    context: RunContext = field(repr=False)
    cancelled_context: RunContext = field(repr=False)
    invocation_count: InvocationCount = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.generator, StructuredGenerator):
            raise TypeError(
                "structured contract generator must satisfy StructuredGenerator"
            )
        if not isinstance(
            self.supported_request,
            StructuredGenerationRequest,
        ):
            raise TypeError(
                "structured contract supported_request must be a "
                "StructuredGenerationRequest"
            )
        if not isinstance(
            self.unsupported_request,
            StructuredGenerationRequest,
        ):
            raise TypeError(
                "structured contract unsupported_request must be a "
                "StructuredGenerationRequest"
            )
        _validate_suite(
            descriptor=self.generator.descriptor,
            expected_kind=CapabilityKind.STRUCTURED_GENERATION,
            supported_requirements=self.supported_request.requirements,
            unsupported_requirements=self.unsupported_request.requirements,
            context=self.context,
            cancelled_context=self.cancelled_context,
            invocation_count=self.invocation_count,
        )

    async def verify(self) -> StructuredGenerationResult[OutputT]:
        """Run shared success, schema, preflight, and cancellation checks."""
        result = await self.generator.generate(
            self.supported_request,
            self.context,
        )
        if not isinstance(result, StructuredGenerationResult):
            raise AssertionError(
                "structured generator returned a non-StructuredGenerationResult"
            )
        if result.schema_id != self.supported_request.output_schema.schema_id:
            raise AssertionError(
                "structured generator result does not use the requested schema"
            )

        await _verify_preflight_does_not_invoke(
            operation=lambda: self.generator.generate(
                self.unsupported_request,
                self.context,
            ),
            invocation_count=self.invocation_count,
        )
        await _verify_cancellation_does_not_invoke(
            operation=lambda: self.generator.generate(
                self.supported_request,
                self.cancelled_context,
            ),
            invocation_count=self.invocation_count,
        )
        return result


def _validate_suite(
    *,
    descriptor: CapabilityDescriptor,
    expected_kind: CapabilityKind,
    supported_requirements: CapabilityRequirements,
    unsupported_requirements: CapabilityRequirements,
    context: RunContext,
    cancelled_context: RunContext,
    invocation_count: InvocationCount,
) -> None:
    if not isinstance(descriptor, CapabilityDescriptor):
        raise TypeError(
            "generation contract descriptor must be a CapabilityDescriptor"
        )
    if descriptor.kind is not expected_kind:
        raise ValueError(
            "generation contract descriptor has the wrong capability kind"
        )
    if not isinstance(context, RunContext):
        raise TypeError("generation contract context must be a RunContext")
    if not isinstance(cancelled_context, RunContext):
        raise TypeError(
            "generation contract cancelled_context must be a RunContext"
        )
    if not cancelled_context.cancellation.is_cancelled:
        raise ValueError(
            "generation contract cancelled_context must already be cancelled"
        )
    if not callable(invocation_count):
        raise TypeError("generation contract invocation_count must be callable")
    if supported_requirements.unmet_by(descriptor):
        raise ValueError(
            "generation contract supported_request is not supported"
        )
    if not unsupported_requirements.unmet_by(descriptor):
        raise ValueError(
            "generation contract unsupported_request must be unsupported"
        )
    _read_invocation_count(invocation_count)


async def _verify_preflight_does_not_invoke(
    *,
    operation: Callable[[], Awaitable[object]],
    invocation_count: InvocationCount,
) -> None:
    before = _read_invocation_count(invocation_count)
    try:
        await operation()
    except ValueError:
        pass
    else:
        raise AssertionError(
            "unsupported generation request did not fail during preflight"
        )
    after = _read_invocation_count(invocation_count)
    if after != before:
        raise AssertionError(
            "unsupported generation request invoked the implementation"
        )


async def _verify_cancellation_does_not_invoke(
    *,
    operation: Callable[[], Awaitable[object]],
    invocation_count: InvocationCount,
) -> None:
    before = _read_invocation_count(invocation_count)
    try:
        await operation()
    except RunCancelled:
        pass
    else:
        raise AssertionError(
            "cancelled generation request did not raise RunCancelled"
        )
    after = _read_invocation_count(invocation_count)
    if after != before:
        raise AssertionError("cancelled generation request invoked implementation")


def _read_invocation_count(invocation_count: InvocationCount) -> int:
    count = invocation_count()
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise TypeError(
            "generation contract invocation_count must return a "
            "non-negative integer"
        )
    return count
