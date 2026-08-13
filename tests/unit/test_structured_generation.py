from __future__ import annotations

import asyncio
import unittest
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    DataRetention,
    GenerationUsage,
    ModelMetadata,
    StructuredGenerationRequest,
    StructuredGenerationResult,
    StructuredGenerator,
    StructuredOutputSchema,
    TextGenerationFinishReason,
    TextGenerationRequest,
    TextMessage,
    TextMessageRole,
)
from agentrig.core import (
    ArtifactId,
    ArtifactRef,
    CancellationSource,
    JsonValue,
    RunContext,
    RunId,
)


@dataclass(frozen=True, slots=True)
class StoryOutline:
    title: str
    beats: tuple[str, ...]


def decode_outline(value: JsonValue) -> StoryOutline:
    if not isinstance(value, Mapping) or set(value) != {"title", "beats"}:
        raise ValueError("outline must contain exactly title and beats")
    title = value["title"]
    beats = value["beats"]
    if not isinstance(title, str):
        raise ValueError("outline title must be a string")
    if not isinstance(beats, tuple) or any(
        not isinstance(item, str) for item in beats
    ):
        raise ValueError("outline beats must be an array of strings")
    return StoryOutline(title=title, beats=beats)


def create_schema() -> StructuredOutputSchema[StoryOutline]:
    return StructuredOutputSchema(
        schema_id="story.outline.v1",
        json_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "beats": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["title", "beats"],
            "additionalProperties": False,
        },
        decoder=decode_outline,
    )


def create_descriptor(
    *,
    features: frozenset[CapabilityFeature] | None = None,
    limits: dict[CapabilityLimit, int] | None = None,
) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id="example.structured",
        version="1",
        kind=CapabilityKind.STRUCTURED_GENERATION,
        features=(
            features
            if features is not None
            else frozenset({CapabilityFeature.STRUCTURED_OUTPUT})
        ),
        limits=limits if limits is not None else {},
        data_retention=DataRetention.NOT_RETAINED,
    )


def create_result(
    encoded_output: JsonValue,
    *,
    schema: StructuredOutputSchema[StoryOutline] | None = None,
) -> StructuredGenerationResult[StoryOutline]:
    return StructuredGenerationResult(
        encoded_output=encoded_output,
        output_schema=schema if schema is not None else create_schema(),
        usage=GenerationUsage(input_tokens=12, output_tokens=8),
        model=ModelMetadata(provider="example", model_id="structured-1"),
        finish_reason=TextGenerationFinishReason.COMPLETED,
    )


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 14, 3, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


@dataclass
class SequentialRunIdGenerator:
    next_value: int = 1

    def generate(self) -> RunId:
        run_id = RunId(f"run-{self.next_value}")
        self.next_value += 1
        return run_id


def create_context() -> RunContext:
    return RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialRunIdGenerator(),
        cancellation=CancellationSource().token,
    )


def create_image() -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId("image-1"),
        kind="image",
        media_type="image/png",
        producer_run_id=RunId("run-producer"),
        workspace_path="inputs/reference.png",
    )


@dataclass
class ScriptedStructuredGenerator:
    descriptor: CapabilityDescriptor
    outputs: tuple[JsonValue, ...]
    call_count: int = 0

    async def generate(
        self,
        request: StructuredGenerationRequest[StoryOutline],
        context: RunContext,
    ) -> StructuredGenerationResult[StoryOutline]:
        request.require_supported_by(self.descriptor)
        context.cancellation.raise_if_cancelled()
        if context.deadline is not None:
            context.deadline.raise_if_expired(context.clock)
        output = self.outputs[min(self.call_count, len(self.outputs) - 1)]
        self.call_count += 1
        return create_result(output, schema=request.output_schema)


async def generate_typed(
    generator: StructuredGenerator[StoryOutline],
    request: StructuredGenerationRequest[StoryOutline],
    context: RunContext,
) -> StructuredGenerationResult[StoryOutline]:
    return await generator.generate(request, context)


class StructuredOutputSchemaTest(unittest.TestCase):
    def test_copies_schema_and_decodes_frozen_json(self) -> None:
        json_schema: dict[str, JsonValue] = {
            "type": "object",
            "required": ["title", "beats"],
        }
        schema = StructuredOutputSchema(
            schema_id="story.outline.v1",
            json_schema=json_schema,
            decoder=decode_outline,
        )
        json_schema["type"] = "string"

        result = schema.decode(
            {"title": "Opening", "beats": ["Arrival", "Discovery"]}
        )

        self.assertEqual(schema.json_schema["type"], "object")
        self.assertEqual(
            result,
            StoryOutline("Opening", ("Arrival", "Discovery")),
        )
        with self.assertRaises(TypeError):
            schema.json_schema["type"] = "array"  # type: ignore[index]

    def test_rejects_invalid_schema_identity_body_or_decoder(self) -> None:
        with self.assertRaises(ValueError):
            StructuredOutputSchema(
                schema_id=" padded ",
                json_schema={"type": "object"},
                decoder=decode_outline,
            )
        with self.assertRaises(ValueError):
            StructuredOutputSchema(
                schema_id="story.outline.v1",
                json_schema={},
                decoder=decode_outline,
            )
        with self.assertRaises(ValueError):
            StructuredOutputSchema(
                schema_id="story.outline.v1",
                json_schema={"invalid": object()},  # type: ignore[dict-item]
                decoder=decode_outline,
            )
        with self.assertRaises(TypeError):
            StructuredOutputSchema(
                schema_id="story.outline.v1",
                json_schema={"type": "object"},
                decoder="invalid",  # type: ignore[arg-type]
            )


class StructuredGenerationRequestTest(unittest.TestCase):
    def test_requires_strict_output_and_reuses_input_requirements(self) -> None:
        request = StructuredGenerationRequest(
            input=TextGenerationRequest(
                messages=(
                    TextMessage(
                        role=TextMessageRole.USER,
                        text="Outline this image.",
                        artifacts=(create_image(),),
                    ),
                ),
                max_output_tokens=120,
            ),
            output_schema=create_schema(),
        )

        self.assertEqual(
            request.requirements.features,
            frozenset(
                {
                    CapabilityFeature.MESSAGE_INPUT,
                    CapabilityFeature.MULTIMODAL_INPUT,
                    CapabilityFeature.STRUCTURED_OUTPUT,
                }
            ),
        )
        self.assertEqual(
            request.requirements.minimum_limits,
            {
                CapabilityLimit.MAX_INPUT_ARTIFACTS: 1,
                CapabilityLimit.MAX_OUTPUT_TOKENS: 120,
            },
        )
        self.assertNotIn("Outline this image", repr(request))
        request.require_supported_by(
            create_descriptor(
                features=request.requirements.features,
                limits={
                    CapabilityLimit.MAX_INPUT_ARTIFACTS: 1,
                    CapabilityLimit.MAX_OUTPUT_TOKENS: 120,
                },
            )
        )

    def test_rejects_invalid_request_values_or_unsupported_features(self) -> None:
        with self.assertRaises(TypeError):
            StructuredGenerationRequest(
                input="invalid",  # type: ignore[arg-type]
                output_schema=create_schema(),
            )
        with self.assertRaises(TypeError):
            StructuredGenerationRequest(
                input=TextGenerationRequest(prompt="outline"),
                output_schema="invalid",  # type: ignore[arg-type]
            )

        request = StructuredGenerationRequest(
            input=TextGenerationRequest(prompt="outline"),
            output_schema=create_schema(),
        )
        with self.assertRaisesRegex(ValueError, "structured_output"):
            request.require_supported_by(create_descriptor(features=frozenset()))


class StructuredGenerationResultTest(unittest.TestCase):
    def test_decodes_typed_output_and_freezes_encoded_provider_json(self) -> None:
        encoded: dict[str, JsonValue] = {
            "title": "Opening",
            "beats": ["Arrival", "Discovery"],
        }

        result = create_result(encoded)
        encoded["title"] = "Changed"

        self.assertEqual(
            result.output,
            StoryOutline("Opening", ("Arrival", "Discovery")),
        )
        self.assertEqual(result.encoded_output["title"], "Opening")
        self.assertEqual(result.schema_id, "story.outline.v1")
        self.assertEqual(result.usage.total_tokens, 20)
        self.assertNotIn("Opening", repr(result))
        with self.assertRaises(TypeError):
            result.encoded_output["title"] = "Other"  # type: ignore[index]

    def test_schema_decoder_rejects_malformed_or_extra_provider_output(
        self,
    ) -> None:
        invalid_outputs: tuple[JsonValue, ...] = (
            {"title": "Opening"},
            {"title": "Opening", "beats": [1]},
            {"title": "Opening", "beats": [], "private": "unexpected"},
        )
        for output in invalid_outputs:
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    create_result(output)

    def test_rejects_invalid_result_metadata(self) -> None:
        valid_output = {"title": "Opening", "beats": []}
        with self.assertRaises(TypeError):
            StructuredGenerationResult(
                encoded_output=valid_output,
                output_schema="invalid",  # type: ignore[arg-type]
                usage=GenerationUsage(),
                model=ModelMetadata(provider="example", model_id="model"),
                finish_reason=TextGenerationFinishReason.COMPLETED,
            )
        with self.assertRaises(TypeError):
            StructuredGenerationResult(
                encoded_output=valid_output,
                output_schema=create_schema(),
                usage="invalid",  # type: ignore[arg-type]
                model=ModelMetadata(provider="example", model_id="model"),
                finish_reason=TextGenerationFinishReason.COMPLETED,
            )


class StructuredGeneratorContractTest(unittest.TestCase):
    def test_protocol_supports_a_typed_strict_generator(self) -> None:
        generator = ScriptedStructuredGenerator(
            descriptor=create_descriptor(),
            outputs=(
                {"title": "Opening", "beats": ["Arrival"]},
            ),
        )
        request = StructuredGenerationRequest(
            input=TextGenerationRequest(prompt="Create an outline."),
            output_schema=create_schema(),
        )

        result = asyncio.run(
            generate_typed(generator, request, create_context())
        )

        self.assertIsInstance(generator, StructuredGenerator)
        self.assertEqual(result.output.title, "Opening")
        self.assertEqual(generator.call_count, 1)

    def test_preflight_failure_does_not_consume_scripted_output(self) -> None:
        generator = ScriptedStructuredGenerator(
            descriptor=create_descriptor(features=frozenset()),
            outputs=(
                {"title": "Opening", "beats": ["Arrival"]},
            ),
        )
        request = StructuredGenerationRequest(
            input=TextGenerationRequest(prompt="Create an outline."),
            output_schema=create_schema(),
        )

        with self.assertRaisesRegex(ValueError, "structured_output"):
            asyncio.run(generator.generate(request, create_context()))

        self.assertEqual(generator.call_count, 0)


if __name__ == "__main__":
    unittest.main()
