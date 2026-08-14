"""Reusable contract probes for portable generation implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from agentrig.capabilities import (
    CapabilityKind,
    StructuredGenerationRequest,
    StructuredGenerationResult,
    StructuredGenerator,
    TextGenerationRequest,
    TextGenerationResult,
    TextGenerator,
)
from agentrig.core.context import RunContext
from agentrig.testing._capability_contracts import (
    InvocationCount,
    validate_contract_suite,
    verify_cancellation_does_not_invoke,
    verify_preflight_does_not_invoke,
)

OutputT = TypeVar("OutputT")


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
        validate_contract_suite(
            label="generation",
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

        await verify_preflight_does_not_invoke(
            label="generation",
            operation=lambda: self.generator.generate(
                self.unsupported_request,
                self.context,
            ),
            invocation_count=self.invocation_count,
        )
        await verify_cancellation_does_not_invoke(
            label="generation",
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
        validate_contract_suite(
            label="generation",
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

        await verify_preflight_does_not_invoke(
            label="generation",
            operation=lambda: self.generator.generate(
                self.unsupported_request,
                self.context,
            ),
            invocation_count=self.invocation_count,
        )
        await verify_cancellation_does_not_invoke(
            label="generation",
            operation=lambda: self.generator.generate(
                self.supported_request,
                self.cancelled_context,
            ),
            invocation_count=self.invocation_count,
        )
        return result
