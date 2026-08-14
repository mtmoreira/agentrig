"""Positive fixture for scripted generation fakes and contract suites."""

from collections.abc import Mapping
from dataclasses import dataclass

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    GenerationUsage,
    ModelMetadata,
    StructuredGenerationRequest,
    StructuredGenerator,
    StructuredOutputSchema,
    TextGenerationFinishReason,
    TextGenerationRequest,
    TextGenerationResult,
    TextGenerator,
)
from agentrig.core import JsonValue, RunContext
from agentrig.testing import (
    ScriptedStructuredGeneration,
    ScriptedStructuredGenerator,
    ScriptedTextGenerator,
    StructuredGeneratorContractSuite,
    TextGeneratorContractSuite,
)


@dataclass(frozen=True, slots=True)
class Answer:
    value: str


def decode_answer(value: JsonValue) -> Answer:
    if not isinstance(value, Mapping):
        raise ValueError("answer must be an object")
    raw_value = value.get("value")
    if not isinstance(raw_value, str):
        raise ValueError("answer must contain a string value")
    return Answer(value=raw_value)


metadata = ModelMetadata(provider="scripted", model_id="test")
text_result = TextGenerationResult(
    text="complete",
    usage=GenerationUsage(),
    model=metadata,
    finish_reason=TextGenerationFinishReason.COMPLETED,
)
text: TextGenerator = ScriptedTextGenerator(
    descriptor=CapabilityDescriptor(
        capability_id="scripted.text",
        version="1",
        kind=CapabilityKind.TEXT_GENERATION,
    ),
    outcomes=(text_result,),
)

schema = StructuredOutputSchema(
    schema_id="answer.v1",
    json_schema={"type": "object"},
    decoder=decode_answer,
)
structured_fake = ScriptedStructuredGenerator[Answer](
    descriptor=CapabilityDescriptor(
        capability_id="scripted.structured",
        version="1",
        kind=CapabilityKind.STRUCTURED_GENERATION,
        features=frozenset({CapabilityFeature.STRUCTURED_OUTPUT}),
    ),
    outcomes=(
        ScriptedStructuredGeneration(
            encoded_output={"value": "complete"},
            usage=GenerationUsage(),
            model=metadata,
            finish_reason=TextGenerationFinishReason.COMPLETED,
        ),
    ),
)
structured: StructuredGenerator[Answer] = structured_fake


def text_suite(
    context: RunContext,
    cancelled_context: RunContext,
) -> TextGeneratorContractSuite:
    return TextGeneratorContractSuite(
        generator=text,
        supported_request=TextGenerationRequest(prompt="supported"),
        unsupported_request=TextGenerationRequest(
            max_output_tokens=1,
            prompt="unsupported",
        ),
        context=context,
        cancelled_context=cancelled_context,
        invocation_count=lambda: 0,
    )


def structured_suite(
    context: RunContext,
    cancelled_context: RunContext,
) -> StructuredGeneratorContractSuite[Answer]:
    request = StructuredGenerationRequest(
        input=TextGenerationRequest(prompt="supported"),
        output_schema=schema,
    )
    return StructuredGeneratorContractSuite(
        generator=structured_fake,
        supported_request=request,
        unsupported_request=StructuredGenerationRequest(
            input=TextGenerationRequest(
                prompt="unsupported",
                max_output_tokens=1,
            ),
            output_schema=schema,
        ),
        context=context,
        cancelled_context=cancelled_context,
        invocation_count=lambda: 0,
    )
