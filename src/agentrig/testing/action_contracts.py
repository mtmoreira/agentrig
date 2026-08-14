"""Reusable contract probes for coding and image implementations."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentrig.capabilities import (
    CapabilityKind,
    CodingAgent,
    CodingResult,
    CodingTask,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageGenerator,
)
from agentrig.core.context import RunContext
from agentrig.testing._capability_contracts import (
    InvocationCount,
    validate_contract_suite,
    verify_cancellation_does_not_invoke,
    verify_preflight_does_not_invoke,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CodingAgentContractSuite:
    """Portable checks shared by coding fakes and provider integrations."""

    agent: CodingAgent = field(repr=False, compare=False)
    supported_task: CodingTask = field(repr=False)
    unsupported_task: CodingTask = field(repr=False)
    context: RunContext = field(repr=False)
    cancelled_context: RunContext = field(repr=False)
    invocation_count: InvocationCount = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.agent, CodingAgent):
            raise TypeError(
                "coding contract agent must satisfy CodingAgent"
            )
        if not isinstance(self.supported_task, CodingTask):
            raise TypeError(
                "coding contract supported_task must be a CodingTask"
            )
        if not isinstance(self.unsupported_task, CodingTask):
            raise TypeError(
                "coding contract unsupported_task must be a CodingTask"
            )
        validate_contract_suite(
            label="coding",
            descriptor=self.agent.descriptor,
            expected_kind=CapabilityKind.CODING,
            supported_requirements=(
                self.supported_task.capability_requirements
            ),
            unsupported_requirements=(
                self.unsupported_task.capability_requirements
            ),
            context=self.context,
            cancelled_context=self.cancelled_context,
            invocation_count=self.invocation_count,
        )

    async def verify(self) -> CodingResult:
        """Run shared success, binding, preflight, and cancellation checks."""
        result = await self.agent.execute(
            self.supported_task,
            self.context,
        )
        if not isinstance(result, CodingResult):
            raise AssertionError(
                "coding agent returned a non-CodingResult value"
            )
        if result.task_id != self.supported_task.task_id:
            raise AssertionError(
                "coding result is not bound to the requested task"
            )
        if result.workspace_id != self.supported_task.workspace.workspace_id:
            raise AssertionError(
                "coding result is not bound to the authorized workspace"
            )

        await verify_preflight_does_not_invoke(
            label="coding",
            operation=lambda: self.agent.execute(
                self.unsupported_task,
                self.context,
            ),
            invocation_count=self.invocation_count,
        )
        await verify_cancellation_does_not_invoke(
            label="coding",
            operation=lambda: self.agent.execute(
                self.supported_task,
                self.cancelled_context,
            ),
            invocation_count=self.invocation_count,
        )
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageGeneratorContractSuite:
    """Portable checks shared by image fakes and provider integrations."""

    generator: ImageGenerator = field(repr=False, compare=False)
    supported_request: ImageGenerationRequest = field(repr=False)
    unsupported_request: ImageGenerationRequest = field(repr=False)
    context: RunContext = field(repr=False)
    cancelled_context: RunContext = field(repr=False)
    invocation_count: InvocationCount = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.generator, ImageGenerator):
            raise TypeError(
                "image contract generator must satisfy ImageGenerator"
            )
        if not isinstance(self.supported_request, ImageGenerationRequest):
            raise TypeError(
                "image contract supported_request must be an "
                "ImageGenerationRequest"
            )
        if not isinstance(self.unsupported_request, ImageGenerationRequest):
            raise TypeError(
                "image contract unsupported_request must be an "
                "ImageGenerationRequest"
            )
        validate_contract_suite(
            label="image",
            descriptor=self.generator.descriptor,
            expected_kind=CapabilityKind.IMAGE_GENERATION,
            supported_requirements=self.supported_request.requirements,
            unsupported_requirements=self.unsupported_request.requirements,
            context=self.context,
            cancelled_context=self.cancelled_context,
            invocation_count=self.invocation_count,
        )

    async def verify(self) -> ImageGenerationResult:
        """Run shared result, lineage, preflight, and cancellation checks."""
        result = await self.generator.generate(
            self.supported_request,
            self.context,
        )
        if not isinstance(result, ImageGenerationResult):
            raise AssertionError(
                "image generator returned a non-ImageGenerationResult value"
            )
        if (
            result.image.media_type
            != self.supported_request.specification.output_media_type
        ):
            raise AssertionError(
                "image result does not use the requested media type"
            )
        if not set(self.supported_request.source_artifact_ids).issubset(
            result.image.input_artifact_ids
        ):
            raise AssertionError(
                "image result does not preserve requested source lineage"
            )

        await verify_preflight_does_not_invoke(
            label="image",
            operation=lambda: self.generator.generate(
                self.unsupported_request,
                self.context,
            ),
            invocation_count=self.invocation_count,
        )
        await verify_cancellation_does_not_invoke(
            label="image",
            operation=lambda: self.generator.generate(
                self.supported_request,
                self.cancelled_context,
            ),
            invocation_count=self.invocation_count,
        )
        return result
