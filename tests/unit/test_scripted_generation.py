from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import unittest

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    DataRetention,
    GenerationUsage,
    ModelMetadata,
    StructuredGenerationRequest,
    StructuredGenerator,
    StructuredOutputSchema,
    TextGenerationFinishReason,
    TextGenerationRequest,
    TextGenerationResult,
    TextGenerator,
    TextMessage,
    TextMessageRole,
)
from agentrig.core import (
    AgentRigError,
    CancellationSource,
    Deadline,
    DeadlineExceeded,
    Failure,
    FailureKind,
    JsonValue,
    RunCancelled,
    RunContext,
    RunId,
)
from agentrig.testing import (
    ScriptedStructuredGeneration,
    ScriptedStructuredGenerator,
    ScriptedTextGenerator,
    StructuredGeneratorContractSuite,
    TextGeneratorContractSuite,
)


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 15, 6, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


@dataclass
class SequentialRunIdGenerator:
    next_value: int = 1

    def generate(self) -> RunId:
        run_id = RunId(f"run-{self.next_value}")
        self.next_value += 1
        return run_id


def create_context(
    source: CancellationSource | None = None,
    *,
    deadline: Deadline | None = None,
) -> RunContext:
    owned_source = source if source is not None else CancellationSource()
    return RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialRunIdGenerator(),
        cancellation=owned_source.token,
        deadline=deadline,
    )


def cancelled_context() -> RunContext:
    source = CancellationSource()
    source.cancel("contract cancellation")
    return create_context(source)


def text_descriptor() -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id="scripted.text",
        version="1",
        kind=CapabilityKind.TEXT_GENERATION,
        limits={CapabilityLimit.MAX_OUTPUT_TOKENS: 32},
        data_retention=DataRetention.NOT_RETAINED,
    )


def structured_descriptor() -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id="scripted.structured",
        version="1",
        kind=CapabilityKind.STRUCTURED_GENERATION,
        features=frozenset({CapabilityFeature.STRUCTURED_OUTPUT}),
        limits={CapabilityLimit.MAX_OUTPUT_TOKENS: 32},
        data_retention=DataRetention.NOT_RETAINED,
    )


def text_result(text: str = "complete") -> TextGenerationResult:
    return TextGenerationResult(
        text=text,
        usage=GenerationUsage(input_tokens=4, output_tokens=2),
        model=ModelMetadata(provider="scripted", model_id="text-1"),
        finish_reason=TextGenerationFinishReason.COMPLETED,
    )


@dataclass(frozen=True, slots=True)
class Outline:
    title: str


def decode_outline(value: JsonValue) -> Outline:
    if not isinstance(value, Mapping) or set(value) != {"title"}:
        raise ValueError("outline must contain exactly one title")
    title = value["title"]
    if not isinstance(title, str):
        raise ValueError("outline title must be a string")
    return Outline(title=title)


def outline_schema() -> StructuredOutputSchema[Outline]:
    return StructuredOutputSchema(
        schema_id="example.outline.v1",
        json_schema={
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
            "additionalProperties": False,
        },
        decoder=decode_outline,
    )


def structured_outcome(
    title: str = "Opening",
) -> ScriptedStructuredGeneration:
    return ScriptedStructuredGeneration(
        encoded_output={"title": title},
        usage=GenerationUsage(input_tokens=8, output_tokens=3),
        model=ModelMetadata(provider="scripted", model_id="structured-1"),
        finish_reason=TextGenerationFinishReason.COMPLETED,
    )


def supported_text_request() -> TextGenerationRequest:
    return TextGenerationRequest(prompt="Draft", max_output_tokens=32)


def unsupported_text_request() -> TextGenerationRequest:
    return TextGenerationRequest(
        messages=(TextMessage(role=TextMessageRole.USER, text="Draft"),),
        max_output_tokens=32,
    )


def supported_structured_request() -> StructuredGenerationRequest[Outline]:
    return StructuredGenerationRequest(
        input=TextGenerationRequest(prompt="Outline", max_output_tokens=32),
        output_schema=outline_schema(),
    )


def unsupported_structured_request() -> StructuredGenerationRequest[Outline]:
    return StructuredGenerationRequest(
        input=TextGenerationRequest(
            messages=(TextMessage(role=TextMessageRole.USER, text="Outline"),),
            max_output_tokens=32,
        ),
        output_schema=outline_schema(),
    )


def provider_failure() -> Failure:
    return Failure(
        kind=FailureKind.TRANSIENT_PROVIDER,
        message="scripted provider is temporarily unavailable",
        code="provider.busy",
    )


class ScriptedTextGeneratorTest(unittest.TestCase):
    def test_returns_results_and_failures_in_order_with_stable_calls(self) -> None:
        result = text_result("first")
        failure = provider_failure()
        generator = ScriptedTextGenerator(
            descriptor=text_descriptor(),
            outcomes=(result, failure),
        )
        context = create_context()

        first = asyncio.run(generator.generate(supported_text_request(), context))
        snapshot = generator.calls
        with self.assertRaises(AgentRigError) as raised:
            asyncio.run(generator.generate(supported_text_request(), context))

        self.assertIs(first, result)
        self.assertIs(raised.exception.failure, failure)
        self.assertEqual(tuple(call.index for call in snapshot), (0,))
        self.assertEqual(tuple(call.index for call in generator.calls), (0, 1))
        self.assertIs(generator.calls[0].context, context)
        self.assertTrue(generator.is_exhausted)

    def test_exhaustion_is_sanitized_and_repeat_last_is_unbounded(self) -> None:
        result = text_result()
        exhausted = ScriptedTextGenerator(
            descriptor=text_descriptor(),
            outcomes=(result,),
        )
        context = create_context()
        asyncio.run(exhausted.generate(supported_text_request(), context))

        with self.assertRaises(AgentRigError) as raised:
            asyncio.run(exhausted.generate(supported_text_request(), context))

        self.assertEqual(
            raised.exception.failure.code,
            "scripted_text_generator.exhausted",
        )
        self.assertEqual(
            raised.exception.failure.metadata,
            {
                "capability_id": "scripted.text",
                "capability_version": "1",
            },
        )
        repeating = ScriptedTextGenerator(
            descriptor=text_descriptor(),
            outcomes=(result,),
            repeat_last=True,
        )
        repeated = tuple(
            asyncio.run(repeating.generate(supported_text_request(), context))
            for _ in range(3)
        )
        self.assertEqual(repeated, (result, result, result))
        self.assertFalse(repeating.is_exhausted)

    def test_preflight_and_constraints_do_not_consume_outcomes(self) -> None:
        generator = ScriptedTextGenerator(
            descriptor=text_descriptor(),
            outcomes=(text_result(),),
        )
        with self.assertRaisesRegex(ValueError, "message_input"):
            asyncio.run(
                generator.generate(unsupported_text_request(), create_context())
            )

        source = CancellationSource()
        source.cancel("caller stopped")
        with self.assertRaises(RunCancelled):
            asyncio.run(
                generator.generate(supported_text_request(), create_context(source))
            )

        deadline = Deadline(
            expires_at=FixedClock().now(),
            monotonic_deadline=FixedClock().monotonic(),
        )
        with self.assertRaises(DeadlineExceeded):
            asyncio.run(
                generator.generate(
                    supported_text_request(),
                    create_context(deadline=deadline),
                )
            )

        self.assertEqual(generator.calls, ())
        self.assertFalse(generator.is_exhausted)


class ScriptedStructuredGeneratorTest(unittest.TestCase):
    def test_decodes_copied_json_through_each_requested_schema(self) -> None:
        encoded: dict[str, JsonValue] = {"title": "Opening"}
        scenario = ScriptedStructuredGeneration(
            encoded_output=encoded,
            usage=GenerationUsage(input_tokens=8, output_tokens=3),
            model=ModelMetadata(provider="scripted", model_id="structured-1"),
            finish_reason=TextGenerationFinishReason.COMPLETED,
        )
        encoded["title"] = "Changed"
        generator = ScriptedStructuredGenerator[Outline](
            descriptor=structured_descriptor(),
            outcomes=(scenario,),
        )
        request = supported_structured_request()
        context = create_context()

        result = asyncio.run(generator.generate(request, context))

        self.assertEqual(result.output, Outline(title="Opening"))
        self.assertEqual(result.schema_id, request.output_schema.schema_id)
        self.assertIsInstance(result.encoded_output, Mapping)
        if not isinstance(result.encoded_output, Mapping):
            raise AssertionError("structured output must be an object")
        self.assertEqual(result.encoded_output["title"], "Opening")
        self.assertEqual(len(generator.calls), 1)
        self.assertIs(generator.calls[0].request, request)

    def test_failure_exhaustion_and_repeat_last_match_text_semantics(self) -> None:
        failure = provider_failure()
        generator = ScriptedStructuredGenerator[Outline](
            descriptor=structured_descriptor(),
            outcomes=(failure,),
        )
        with self.assertRaises(AgentRigError) as raised:
            asyncio.run(
                generator.generate(
                    supported_structured_request(),
                    create_context(),
                )
            )
        self.assertIs(raised.exception.failure, failure)
        self.assertTrue(generator.is_exhausted)

        with self.assertRaises(AgentRigError) as exhausted:
            asyncio.run(
                generator.generate(
                    supported_structured_request(),
                    create_context(),
                )
            )
        self.assertEqual(
            exhausted.exception.failure.code,
            "scripted_structured_generator.exhausted",
        )

        repeating = ScriptedStructuredGenerator[Outline](
            descriptor=structured_descriptor(),
            outcomes=(structured_outcome(),),
            repeat_last=True,
        )
        for _ in range(3):
            result = asyncio.run(
                repeating.generate(
                    supported_structured_request(),
                    create_context(),
                )
            )
            self.assertEqual(result.output.title, "Opening")
        self.assertFalse(repeating.is_exhausted)


class GenerationContractSuiteTest(unittest.TestCase):
    def test_text_suite_verifies_shared_portable_semantics(self) -> None:
        generator = ScriptedTextGenerator(
            descriptor=text_descriptor(),
            outcomes=(text_result(),),
        )
        suite = TextGeneratorContractSuite(
            generator=generator,
            supported_request=supported_text_request(),
            unsupported_request=unsupported_text_request(),
            context=create_context(),
            cancelled_context=cancelled_context(),
            invocation_count=lambda: len(generator.calls),
        )

        result = asyncio.run(suite.verify())

        self.assertIsInstance(generator, TextGenerator)
        self.assertEqual(result.text, "complete")
        self.assertEqual(len(generator.calls), 1)

    def test_structured_suite_verifies_shared_portable_semantics(self) -> None:
        generator = ScriptedStructuredGenerator[Outline](
            descriptor=structured_descriptor(),
            outcomes=(structured_outcome(),),
        )
        suite = StructuredGeneratorContractSuite(
            generator=generator,
            supported_request=supported_structured_request(),
            unsupported_request=unsupported_structured_request(),
            context=create_context(),
            cancelled_context=cancelled_context(),
            invocation_count=lambda: len(generator.calls),
        )

        result = asyncio.run(suite.verify())

        self.assertIsInstance(generator, StructuredGenerator)
        self.assertEqual(result.output, Outline(title="Opening"))
        self.assertEqual(len(generator.calls), 1)

    def test_rejects_invalid_contract_fixture_configuration(self) -> None:
        generator = ScriptedTextGenerator(
            descriptor=text_descriptor(),
            outcomes=(text_result(),),
        )
        with self.assertRaisesRegex(ValueError, "must already be cancelled"):
            TextGeneratorContractSuite(
                generator=generator,
                supported_request=supported_text_request(),
                unsupported_request=unsupported_text_request(),
                context=create_context(),
                cancelled_context=create_context(),
                invocation_count=lambda: len(generator.calls),
            )
        with self.assertRaisesRegex(ValueError, "must be unsupported"):
            TextGeneratorContractSuite(
                generator=generator,
                supported_request=supported_text_request(),
                unsupported_request=supported_text_request(),
                context=create_context(),
                cancelled_context=cancelled_context(),
                invocation_count=lambda: len(generator.calls),
            )

    def test_suite_detects_an_implementation_that_skips_preflight(self) -> None:
        @dataclass
        class NonConformingGenerator:
            descriptor: CapabilityDescriptor
            invocations: int = 0

            async def generate(
                self,
                request: TextGenerationRequest,
                context: RunContext,
            ) -> TextGenerationResult:
                del request, context
                self.invocations += 1
                return text_result()

        generator = NonConformingGenerator(descriptor=text_descriptor())
        suite = TextGeneratorContractSuite(
            generator=generator,
            supported_request=supported_text_request(),
            unsupported_request=unsupported_text_request(),
            context=create_context(),
            cancelled_context=cancelled_context(),
            invocation_count=lambda: generator.invocations,
        )

        with self.assertRaisesRegex(
            AssertionError,
            "did not fail during preflight",
        ):
            asyncio.run(suite.verify())


class ScriptedGenerationValidationTest(unittest.TestCase):
    def test_rejects_invalid_descriptors_outcomes_and_runtime_values(self) -> None:
        with self.assertRaises(ValueError):
            ScriptedTextGenerator(
                descriptor=structured_descriptor(),
                outcomes=(text_result(),),
            )
        with self.assertRaises(ValueError):
            ScriptedTextGenerator(descriptor=text_descriptor(), outcomes=())
        with self.assertRaises(TypeError):
            ScriptedTextGenerator(
                descriptor=text_descriptor(),
                outcomes=("invalid",),  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            ScriptedStructuredGenerator[Outline](
                descriptor=structured_descriptor(),
                outcomes=("invalid",),  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            ScriptedStructuredGenerator[Outline](
                descriptor=text_descriptor(),
                outcomes=(structured_outcome(),),
            )
        with self.assertRaises(TypeError):
            ScriptedTextGenerator(
                descriptor=text_descriptor(),
                outcomes=(text_result(),),
                repeat_last=1,  # type: ignore[arg-type]
            )

        generator = ScriptedTextGenerator(
            descriptor=text_descriptor(),
            outcomes=(text_result(),),
        )
        with self.assertRaises(TypeError):
            asyncio.run(
                generator.generate(
                    "invalid",  # type: ignore[arg-type]
                    create_context(),
                )
            )
        with self.assertRaises(TypeError):
            asyncio.run(
                generator.generate(
                    supported_text_request(),
                    "invalid",  # type: ignore[arg-type]
                )
            )
        self.assertEqual(generator.calls, ())

    def test_rejects_invalid_structured_scenario_metadata(self) -> None:
        with self.assertRaises(TypeError):
            ScriptedStructuredGeneration(
                encoded_output={"title": "Opening"},
                usage="invalid",  # type: ignore[arg-type]
                model=ModelMetadata(
                    provider="scripted",
                    model_id="structured-1",
                ),
                finish_reason=TextGenerationFinishReason.COMPLETED,
            )
        with self.assertRaises(ValueError):
            ScriptedStructuredGeneration(
                encoded_output=object(),  # type: ignore[arg-type]
                usage=GenerationUsage(),
                model=ModelMetadata(
                    provider="scripted",
                    model_id="structured-1",
                ),
                finish_reason=TextGenerationFinishReason.COMPLETED,
            )


if __name__ == "__main__":
    unittest.main()
