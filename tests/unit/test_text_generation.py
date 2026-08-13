from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass
from datetime import UTC, datetime

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    CapabilityRequirements,
    DataRetention,
    GenerationUsage,
    ModelMetadata,
    TextGenerationFinishReason,
    TextGenerationRequest,
    TextGenerationResult,
    TextGenerator,
    TextMessage,
    TextMessageRole,
)
from agentrig.core import (
    ArtifactId,
    ArtifactRef,
    CancellationSource,
    RunContext,
    RunId,
)


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 14, 2, 0, tzinfo=UTC)

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


def create_artifact(
    artifact_id: str,
    *,
    media_type: str = "image/png",
) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(artifact_id),
        kind="image",
        media_type=media_type,
        producer_run_id=RunId("run-producer"),
        workspace_path=f"inputs/{artifact_id}.bin",
    )


def create_descriptor(
    *,
    features: frozenset[CapabilityFeature] = frozenset(),
    limits: dict[CapabilityLimit, int] | None = None,
    retention: DataRetention = DataRetention.NOT_RETAINED,
) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id="example.text",
        version="1",
        kind=CapabilityKind.TEXT_GENERATION,
        features=features,
        limits=limits if limits is not None else {},
        data_retention=retention,
    )


def create_result(text: str = "complete") -> TextGenerationResult:
    return TextGenerationResult(
        text=text,
        usage=GenerationUsage(input_tokens=4, output_tokens=2),
        model=ModelMetadata(
            provider="example",
            model_id="text-1",
            version="2026-08-14",
        ),
        finish_reason=TextGenerationFinishReason.COMPLETED,
    )


@dataclass(frozen=True)
class EchoTextGenerator:
    descriptor: CapabilityDescriptor

    async def generate(
        self,
        request: TextGenerationRequest,
        context: RunContext,
    ) -> TextGenerationResult:
        request.require_supported_by(self.descriptor)
        context.cancellation.raise_if_cancelled()
        if context.deadline is not None:
            context.deadline.raise_if_expired(context.clock)
        if request.prompt is not None:
            text = request.prompt
        else:
            text = next(
                message.text
                for message in reversed(request.messages)
                if message.text is not None
            )
        return create_result(text)


async def generate_typed(
    generator: TextGenerator,
    request: TextGenerationRequest,
    context: RunContext,
) -> TextGenerationResult:
    return await generator.generate(request, context)


class CapabilityDescriptorTest(unittest.TestCase):
    def test_vocabulary_has_stable_wire_values(self) -> None:
        self.assertEqual(
            tuple(item.value for item in CapabilityKind),
            (
                "text_generation",
                "structured_generation",
                "coding",
                "image_generation",
                "search",
                "retrieval",
                "tool",
            ),
        )
        self.assertEqual(
            tuple(item.value for item in CapabilityLimit),
            (
                "max_input_artifacts",
                "max_changed_files",
                "max_output_tokens",
                "max_reference_images",
                "max_results",
                "max_tool_calls",
            ),
        )
        self.assertEqual(
            tuple(item.value for item in CapabilityFeature),
            (
                "message_input",
                "multimodal_input",
                "streaming",
                "cancellation",
                "structured_output",
                "session_continuation",
                "approval_requests",
                "tool_use",
                "reference_images",
                "masks",
                "regions",
                "citations",
                "idempotency_keys",
            ),
        )
        self.assertEqual(
            tuple(item.value for item in DataRetention),
            (
                "not_retained",
                "transient",
                "provider_managed",
                "unknown",
            ),
        )

    def test_copies_features_limits_and_retention_characteristics(self) -> None:
        features = {
            CapabilityFeature.MESSAGE_INPUT,
            CapabilityFeature.CANCELLATION,
        }
        limits = {CapabilityLimit.MAX_OUTPUT_TOKENS: 4096}

        descriptor = CapabilityDescriptor(
            capability_id="example.text",
            version="2",
            kind=CapabilityKind.TEXT_GENERATION,
            features=features,  # type: ignore[arg-type]
            limits=limits,
            data_retention=DataRetention.TRANSIENT,
        )
        features.clear()
        limits[CapabilityLimit.MAX_OUTPUT_TOKENS] = 1

        self.assertEqual(
            descriptor.features,
            frozenset(
                {
                    CapabilityFeature.MESSAGE_INPUT,
                    CapabilityFeature.CANCELLATION,
                }
            ),
        )
        self.assertEqual(
            descriptor.limits[CapabilityLimit.MAX_OUTPUT_TOKENS],
            4096,
        )
        self.assertEqual(descriptor.data_retention, DataRetention.TRANSIENT)
        with self.assertRaises(TypeError):
            descriptor.limits[
                CapabilityLimit.MAX_OUTPUT_TOKENS
            ] = 2  # type: ignore[index]

    def test_requirements_report_stable_unmet_features_limits_and_retention(
        self,
    ) -> None:
        requirements = CapabilityRequirements(
            kind=CapabilityKind.TEXT_GENERATION,
            features=frozenset(
                {
                    CapabilityFeature.MESSAGE_INPUT,
                    CapabilityFeature.MULTIMODAL_INPUT,
                }
            ),
            minimum_limits={
                CapabilityLimit.MAX_INPUT_ARTIFACTS: 2,
                CapabilityLimit.MAX_OUTPUT_TOKENS: 100,
            },
            allowed_data_retention=frozenset(
                {DataRetention.NOT_RETAINED}
            ),
        )
        descriptor = create_descriptor(
            features=frozenset({CapabilityFeature.MESSAGE_INPUT}),
            limits={
                CapabilityLimit.MAX_INPUT_ARTIFACTS: 1,
                CapabilityLimit.MAX_OUTPUT_TOKENS: 100,
            },
            retention=DataRetention.PROVIDER_MANAGED,
        )

        self.assertEqual(
            requirements.unmet_by(descriptor),
            (
                "feature:multimodal_input",
                "limit:max_input_artifacts>=2",
                "data_retention:provider_managed",
            ),
        )
        with self.assertRaisesRegex(ValueError, "multimodal_input"):
            requirements.require(descriptor)

    def test_rejects_invalid_descriptor_and_requirement_values(self) -> None:
        with self.assertRaises(ValueError):
            CapabilityDescriptor(
                capability_id=" padded ",
                version="1",
                kind=CapabilityKind.TEXT_GENERATION,
            )
        with self.assertRaises(TypeError):
            CapabilityDescriptor(
                capability_id="text",
                version="1",
                kind="text_generation",  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            create_descriptor(
                features=frozenset({"streaming"}),  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            CapabilityDescriptor(
                capability_id="text",
                version="1",
                kind=CapabilityKind.TEXT_GENERATION,
                features=(
                    CapabilityFeature.STREAMING,
                    CapabilityFeature.STREAMING,
                ),  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            create_descriptor(
                limits={CapabilityLimit.MAX_OUTPUT_TOKENS: 0},
            )
        with self.assertRaises(TypeError):
            CapabilityRequirements(
                kind=CapabilityKind.TEXT_GENERATION,
                allowed_data_retention=frozenset(
                    {"unknown"}  # type: ignore[arg-type]
                ),
            )
        with self.assertRaises(ValueError):
            CapabilityRequirements(
                kind=CapabilityKind.TEXT_GENERATION,
                allowed_data_retention=frozenset(),
            )


class TextGenerationRequestTest(unittest.TestCase):
    def test_message_and_finish_vocabularies_have_stable_wire_values(self) -> None:
        self.assertEqual(
            tuple(item.value for item in TextMessageRole),
            ("system", "user", "assistant"),
        )
        self.assertEqual(
            tuple(item.value for item in TextGenerationFinishReason),
            ("completed", "length", "content_filter", "other"),
        )

    def test_free_form_request_is_private_and_requires_only_declared_limits(
        self,
    ) -> None:
        request = TextGenerationRequest(
            prompt="  private prompt\n",
            max_output_tokens=32,
        )

        self.assertNotIn("private prompt", repr(request))
        self.assertEqual(request.prompt, "  private prompt\n")
        self.assertEqual(
            request.requirements.features,
            frozenset(),
        )
        self.assertEqual(
            request.requirements.minimum_limits,
            {CapabilityLimit.MAX_OUTPUT_TOKENS: 32},
        )
        request.require_supported_by(
            create_descriptor(
                limits={CapabilityLimit.MAX_OUTPUT_TOKENS: 64},
            )
        )

    def test_message_and_artifact_inputs_derive_portable_requirements(self) -> None:
        image = create_artifact("image-1")
        audio = create_artifact("audio-1", media_type="audio/wav")
        messages = [
            TextMessage(role=TextMessageRole.SYSTEM, text="Be concise."),
            TextMessage(
                role=TextMessageRole.USER,
                text="Describe the inputs.",
                artifacts=(image,),
            ),
        ]
        input_artifacts = [audio]
        request = TextGenerationRequest(
            messages=messages,  # type: ignore[arg-type]
            input_artifacts=input_artifacts,  # type: ignore[arg-type]
            max_output_tokens=80,
            idempotency_key="request-1",
        )
        messages.clear()
        input_artifacts.clear()

        self.assertEqual(len(request.messages), 2)
        self.assertEqual(request.input_artifacts, (audio,))
        self.assertEqual(
            request.requirements.features,
            frozenset(
                {
                    CapabilityFeature.MESSAGE_INPUT,
                    CapabilityFeature.MULTIMODAL_INPUT,
                    CapabilityFeature.IDEMPOTENCY_KEYS,
                }
            ),
        )
        self.assertEqual(
            request.requirements.minimum_limits,
            {
                CapabilityLimit.MAX_INPUT_ARTIFACTS: 2,
                CapabilityLimit.MAX_OUTPUT_TOKENS: 80,
            },
        )
        request.require_supported_by(
            create_descriptor(
                features=request.requirements.features,
                limits={
                    CapabilityLimit.MAX_INPUT_ARTIFACTS: 2,
                    CapabilityLimit.MAX_OUTPUT_TOKENS: 80,
                },
            )
        )

    def test_rejects_ambiguous_invalid_or_duplicate_inputs(self) -> None:
        image = create_artifact("image-1")
        invalid_requests = (
            {},
            {
                "prompt": "hello",
                "messages": (
                    TextMessage(role=TextMessageRole.USER, text="hello"),
                ),
            },
            {"prompt": " \n "},
            {"prompt": "hello", "max_output_tokens": 0},
            {"prompt": "hello", "idempotency_key": " padded "},
            {
                "prompt": "hello",
                "input_artifacts": (image, image),
            },
            {
                "messages": (
                    TextMessage(
                        role=TextMessageRole.USER,
                        text="hello",
                        artifacts=(image,),
                    ),
                ),
                "input_artifacts": (image,),
            },
        )
        for values in invalid_requests:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    TextGenerationRequest(**values)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            TextMessage(role=TextMessageRole.USER)
        with self.assertRaises(TypeError):
            TextMessage(
                role="user",  # type: ignore[arg-type]
                text="hello",
            )
        with self.assertRaises(TypeError):
            TextGenerationRequest(
                messages=("invalid",),  # type: ignore[arg-type]
            )


class TextGenerationResultTest(unittest.TestCase):
    def test_preserves_usage_model_and_finish_metadata(self) -> None:
        result = create_result("private output")

        self.assertEqual(result.usage.total_tokens, 6)
        self.assertEqual(result.model.provider, "example")
        self.assertEqual(result.model.model_id, "text-1")
        self.assertEqual(
            result.finish_reason,
            TextGenerationFinishReason.COMPLETED,
        )
        self.assertNotIn("private output", repr(result))
        self.assertIsNone(
            GenerationUsage(input_tokens=3).total_tokens
        )
        filtered = TextGenerationResult(
            text="",
            usage=GenerationUsage(),
            model=ModelMetadata(provider="example", model_id="text-1"),
            finish_reason=TextGenerationFinishReason.CONTENT_FILTER,
        )
        self.assertEqual(filtered.text, "")

    def test_rejects_invalid_usage_model_and_result_values(self) -> None:
        with self.assertRaises(ValueError):
            GenerationUsage(input_tokens=-1)
        with self.assertRaises(ValueError):
            ModelMetadata(provider="", model_id="text-1")
        with self.assertRaises(TypeError):
            TextGenerationResult(
                text=object(),  # type: ignore[arg-type]
                usage=GenerationUsage(),
                model=ModelMetadata(provider="example", model_id="text-1"),
                finish_reason=TextGenerationFinishReason.COMPLETED,
            )
        with self.assertRaises(TypeError):
            TextGenerationResult(
                text="complete",
                usage="invalid",  # type: ignore[arg-type]
                model=ModelMetadata(provider="example", model_id="text-1"),
                finish_reason=TextGenerationFinishReason.COMPLETED,
            )


class TextGeneratorContractTest(unittest.TestCase):
    def test_protocol_supports_a_typed_provider_neutral_generator(self) -> None:
        generator = EchoTextGenerator(
            descriptor=create_descriptor(
                limits={CapabilityLimit.MAX_OUTPUT_TOKENS: 32},
            )
        )
        request = TextGenerationRequest(
            prompt="hello",
            max_output_tokens=32,
        )

        result = asyncio.run(
            generate_typed(generator, request, create_context())
        )

        self.assertIsInstance(generator, TextGenerator)
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.usage.total_tokens, 6)

    def test_generator_preflights_unsupported_requests_before_execution(
        self,
    ) -> None:
        generator = EchoTextGenerator(descriptor=create_descriptor())
        request = TextGenerationRequest(
            messages=(TextMessage(role=TextMessageRole.USER, text="hello"),),
        )

        with self.assertRaisesRegex(ValueError, "message_input"):
            asyncio.run(generator.generate(request, create_context()))


if __name__ == "__main__":
    unittest.main()
