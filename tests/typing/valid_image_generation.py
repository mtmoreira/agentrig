"""Positive fixture for typed image-generation execution."""

from dataclasses import dataclass

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityKind,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageGenerator,
    ImageSize,
    ImageSpecification,
    ModelMetadata,
)
from agentrig.core import ArtifactId, ArtifactRef, RunContext, RunId


@dataclass(frozen=True)
class DeterministicImageGenerator:
    descriptor: CapabilityDescriptor

    async def generate(
        self,
        request: ImageGenerationRequest,
        context: RunContext,
    ) -> ImageGenerationResult:
        request.require_supported_by(self.descriptor)
        return ImageGenerationResult(
            request=request,
            image=ArtifactRef(
                artifact_id=ArtifactId("generated-image"),
                kind="image",
                media_type=request.specification.output_media_type,
                producer_run_id=context.run_id,
                workspace_path="outputs/generated.png",
                input_artifact_ids=request.source_artifact_ids,
            ),
            model=ModelMetadata(provider="example", model_id="image-1"),
        )


generator: ImageGenerator = DeterministicImageGenerator(
    descriptor=CapabilityDescriptor(
        capability_id="example.image",
        version="1",
        kind=CapabilityKind.IMAGE_GENERATION,
    )
)
request = ImageGenerationRequest(
    specification=ImageSpecification(
        prompt="Create a quiet landscape.",
        size=ImageSize(width=1024, height=1024),
    )
)


async def generate_image(context: RunContext) -> ArtifactRef:
    result = await generator.generate(request, context)
    return result.image
