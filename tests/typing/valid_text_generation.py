"""Positive fixture for typed provider-independent text generation."""

from dataclasses import dataclass

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    DataRetention,
    GenerationUsage,
    ModelMetadata,
    TextGenerationFinishReason,
    TextGenerationRequest,
    TextGenerationResult,
    TextGenerator,
)
from agentrig.core import RunContext


@dataclass(frozen=True)
class EchoGenerator:
    descriptor: CapabilityDescriptor

    async def generate(
        self,
        request: TextGenerationRequest,
        context: RunContext,
    ) -> TextGenerationResult:
        del context
        request.require_supported_by(self.descriptor)
        if request.prompt is None:
            raise ValueError("typing fixture expects a free-form prompt")
        return TextGenerationResult(
            text=request.prompt,
            usage=GenerationUsage(input_tokens=1, output_tokens=1),
            model=ModelMetadata(provider="example", model_id="echo-1"),
            finish_reason=TextGenerationFinishReason.COMPLETED,
        )


generator: TextGenerator = EchoGenerator(
    descriptor=CapabilityDescriptor(
        capability_id="example.echo",
        version="1",
        kind=CapabilityKind.TEXT_GENERATION,
        features=frozenset({CapabilityFeature.CANCELLATION}),
        limits={CapabilityLimit.MAX_OUTPUT_TOKENS: 1024},
        data_retention=DataRetention.NOT_RETAINED,
    )
)


async def generate_text(
    request: TextGenerationRequest,
    context: RunContext,
) -> str:
    return (await generator.generate(request, context)).text
