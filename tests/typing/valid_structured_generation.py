"""Positive fixture for typed strict structured generation."""

from collections.abc import Mapping
from dataclasses import dataclass

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    DataRetention,
    GenerationUsage,
    ModelMetadata,
    StructuredDecodeDiagnostic,
    StructuredGenerationRequest,
    StructuredGenerationResult,
    StructuredGenerator,
    StructuredOutputDecodeError,
    StructuredOutputSchema,
    TextGenerationFinishReason,
)
from agentrig.core import JsonValue, RunContext


@dataclass(frozen=True)
class Answer:
    text: str


def decode_answer(value: JsonValue) -> Answer:
    if not isinstance(value, Mapping):
        raise StructuredOutputDecodeError("answer_object")
    text = value.get("text")
    if not isinstance(text, str):
        raise StructuredOutputDecodeError("answer_text")
    return Answer(text=text)


schema = StructuredOutputSchema(
    schema_id="example.answer.v1",
    json_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    },
    decoder=decode_answer,
    decode_diagnostics={
        "answer_object": StructuredDecodeDiagnostic(code="object_required", location="/"),
        "answer_text": StructuredDecodeDiagnostic(code="text_required", location="/text"),
    },
)

diagnostic: StructuredDecodeDiagnostic | None = schema.resolve_decode_diagnostic(
    StructuredOutputDecodeError("answer_text")
)


@dataclass(frozen=True)
class AnswerGenerator:
    descriptor: CapabilityDescriptor

    async def generate(
        self,
        request: StructuredGenerationRequest[Answer],
        context: RunContext,
    ) -> StructuredGenerationResult[Answer]:
        del context
        request.require_supported_by(self.descriptor)
        return StructuredGenerationResult(
            encoded_output={"text": "complete"},
            output_schema=request.output_schema,
            usage=GenerationUsage(input_tokens=2, output_tokens=1),
            model=ModelMetadata(provider="example", model_id="structured-1"),
            finish_reason=TextGenerationFinishReason.COMPLETED,
        )


generator: StructuredGenerator[Answer] = AnswerGenerator(
    descriptor=CapabilityDescriptor(
        capability_id="example.answer",
        version="1",
        kind=CapabilityKind.STRUCTURED_GENERATION,
        features=frozenset({CapabilityFeature.STRUCTURED_OUTPUT}),
        data_retention=DataRetention.NOT_RETAINED,
    )
)


async def generate_answer(
    request: StructuredGenerationRequest[Answer],
    context: RunContext,
) -> Answer:
    return (await generator.generate(request, context)).output
