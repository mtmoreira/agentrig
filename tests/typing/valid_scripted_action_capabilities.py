"""Positive fixture for scripted coding and image contract suites."""

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityKind,
    ChangedFileEvidence,
    CodingAgent,
    CodingChangeKind,
    CodingTask,
    CodingValidationStatus,
    ImageGenerationRequest,
    ImageGenerator,
    ModelMetadata,
    ValidationEvidence,
)
from agentrig.core import ArtifactId, RunContext
from agentrig.testing import (
    CodingAgentContractSuite,
    ImageGeneratorContractSuite,
    ScriptedCodingAgent,
    ScriptedCodingScenario,
    ScriptedImageGeneration,
    ScriptedImageGenerator,
)

coding_fake = ScriptedCodingAgent(
    descriptor=CapabilityDescriptor(
        capability_id="scripted.coding",
        version="1",
        kind=CapabilityKind.CODING,
    ),
    outcomes=(
        ScriptedCodingScenario(
            changed_files=(
                ChangedFileEvidence(
                    path="src/example.py",
                    change_kind=CodingChangeKind.MODIFIED,
                ),
            ),
            validations=(
                ValidationEvidence(
                    validation_id="unit",
                    status=CodingValidationStatus.PASSED,
                    summary="Unit tests passed.",
                ),
            ),
        ),
    ),
)
coding: CodingAgent = coding_fake

image_fake = ScriptedImageGenerator(
    descriptor=CapabilityDescriptor(
        capability_id="scripted.image",
        version="1",
        kind=CapabilityKind.IMAGE_GENERATION,
    ),
    outcomes=(
        ScriptedImageGeneration(
            artifact_id=ArtifactId("generated"),
            workspace_path="outputs/generated.png",
            model=ModelMetadata(provider="scripted", model_id="image-1"),
        ),
    ),
)
image: ImageGenerator = image_fake


def coding_suite(
    supported_task: CodingTask,
    unsupported_task: CodingTask,
    context: RunContext,
    cancelled_context: RunContext,
) -> CodingAgentContractSuite:
    return CodingAgentContractSuite(
        agent=coding,
        supported_task=supported_task,
        unsupported_task=unsupported_task,
        context=context,
        cancelled_context=cancelled_context,
        invocation_count=lambda: len(coding_fake.calls),
    )


def image_suite(
    supported_request: ImageGenerationRequest,
    unsupported_request: ImageGenerationRequest,
    context: RunContext,
    cancelled_context: RunContext,
) -> ImageGeneratorContractSuite:
    return ImageGeneratorContractSuite(
        generator=image,
        supported_request=supported_request,
        unsupported_request=unsupported_request,
        context=context,
        cancelled_context=cancelled_context,
        invocation_count=lambda: len(image_fake.calls),
    )
