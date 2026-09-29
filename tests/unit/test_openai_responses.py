from __future__ import annotations

import asyncio
import json
import traceback
import unittest
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agentrig.capabilities import (
    DataRetention,
    StructuredDecodeDiagnostic,
    StructuredGenerationRequest,
    StructuredOutputDecodeError,
    StructuredOutputSchema,
    TextGenerationFinishReason,
    TextGenerationRequest,
)
from agentrig.core import (
    AgentRigError,
    ArtifactId,
    ArtifactRef,
    CancellationSource,
    FailureKind,
    JsonValue,
    ResolvedArtifact,
    RunCancelled,
    RunContext,
    RunId,
)
from agentrig.core.errors import normalize_exception
from agentrig.integrations.openai import (
    OPENAI_RESPONSES_STRUCTURED_CAPABILITY,
    OpenAIResponsesClient,
    OpenAIResponsesRequest,
    OpenAIResponsesResult,
    OpenAIResponsesStatus,
    OpenAIResponsesStructuredGenerator,
)


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 20, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


@dataclass
class Ids:
    value: int = 0

    def generate(self) -> RunId:
        self.value += 1
        return RunId(f"run-{self.value}")


def context(*, source: CancellationSource | None = None) -> RunContext:
    owned = source if source is not None else CancellationSource()
    return RunContext.create_root(
        clock=FixedClock(),
        id_generator=Ids(),
        cancellation=owned.token,
    )


def image() -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId("image-1"),
        kind="image",
        media_type="image/png",
        producer_run_id=RunId("producer-1"),
        workspace_path="private/image.png",
    )


def schema() -> StructuredOutputSchema[str]:
    def decode(value: JsonValue) -> str:
        if not isinstance(value, Mapping) or set(value) != {"description"}:
            raise ValueError("invalid description")
        description = value["description"]
        if not isinstance(description, str):
            raise ValueError("invalid description")
        return description

    return StructuredOutputSchema(
        schema_id="photo.observation.v1",
        json_schema={
            "type": "object",
            "properties": {"description": {"type": "string"}},
            "required": ["description"],
            "additionalProperties": False,
        },
        decoder=decode,
    )


@dataclass
class Resolver:
    artifact: ArtifactRef
    calls: int = 0

    async def resolve(self, artifact: ArtifactRef) -> ResolvedArtifact:
        self.calls += 1
        return ResolvedArtifact(artifact=self.artifact, content=b"private-image")


@dataclass
class Client:
    response: OpenAIResponsesResult
    requests: list[OpenAIResponsesRequest] = field(default_factory=list)
    closed: int = 0

    async def create(self, request: OpenAIResponsesRequest) -> OpenAIResponsesResult:
        self.requests.append(request)
        return self.response

    async def close(self) -> None:
        self.closed += 1


@dataclass
class Factory:
    client: OpenAIResponsesClient
    calls: int = 0

    def create(self) -> OpenAIResponsesClient:
        self.calls += 1
        return self.client


class OpenAIResponsesGeneratorTest(unittest.TestCase):
    def _setup_generation(
        self,
        output_text: str,
        *,
        status: OpenAIResponsesStatus = OpenAIResponsesStatus.COMPLETED,
        output_tokens: int = 30,
        max_output_tokens: int | None = 100,
        output_schema: StructuredOutputSchema[str] | None = None,
    ) -> tuple[
        OpenAIResponsesStructuredGenerator[str],
        Client,
        StructuredGenerationRequest[str],
    ]:
        client = Client(
            OpenAIResponsesResult(
                output_text=output_text,
                model="vision-model",
                status=status,
                input_tokens=20,
                output_tokens=output_tokens,
            )
        )
        generator = OpenAIResponsesStructuredGenerator[str](
            client_factory=Factory(client),
            artifact_resolver=Resolver(image()),
            model="vision-model",
        )
        request = StructuredGenerationRequest(
            input=TextGenerationRequest(
                prompt="private-input-sentinel",
                max_output_tokens=max_output_tokens,
            ),
            output_schema=output_schema if output_schema is not None else schema(),
        )
        return generator, client, request

    def test_declares_conservative_multimodal_contract(self) -> None:
        descriptor = OPENAI_RESPONSES_STRUCTURED_CAPABILITY
        self.assertEqual(descriptor.data_retention, DataRetention.PROVIDER_MANAGED)
        self.assertEqual(descriptor.kind.value, "structured_generation")
        self.assertIn("multimodal_input", {item.value for item in descriptor.features})

    def test_resolves_images_and_returns_schema_decoded_output(self) -> None:
        artifact = image()
        resolver = Resolver(artifact)
        client = Client(
            OpenAIResponsesResult(
                output_text='{"description":"two people"}',
                model="vision-model",
                status=OpenAIResponsesStatus.COMPLETED,
                input_tokens=20,
                output_tokens=5,
            )
        )
        factory = Factory(client)
        generator = OpenAIResponsesStructuredGenerator[str](
            client_factory=factory,
            artifact_resolver=resolver,
            model="vision-model",
        )
        request = StructuredGenerationRequest(
            input=TextGenerationRequest(
                prompt="Describe only allowed visible attributes.",
                input_artifacts=(artifact,),
                max_output_tokens=100,
            ),
            output_schema=schema(),
        )

        result = asyncio.run(generator.generate(request, context()))

        self.assertEqual(result.output, "two people")
        self.assertEqual(result.usage.total_tokens, 25)
        self.assertEqual(resolver.calls, 1)
        self.assertEqual(factory.calls, 1)
        self.assertEqual(client.closed, 1)
        self.assertEqual(client.requests[0].schema_name, "photo_observation_v1")
        self.assertNotIn("private-image", repr(client.requests[0]))

    def test_invalid_output_retains_only_stage_status_and_usage(self) -> None:
        cases = (
            (
                "completed malformed JSON",
                "private-output-sentinel",
                OpenAIResponsesStatus.COMPLETED,
                10,
                "json_parse",
            ),
            (
                "incomplete truncated JSON",
                '{"description":"private-output-sentinel',
                OpenAIResponsesStatus.INCOMPLETE,
                100,
                "json_parse",
            ),
            (
                "completed decoder rejection",
                '{"wrong":"private-output-sentinel"}',
                OpenAIResponsesStatus.COMPLETED,
                30,
                "schema_decode",
            ),
            (
                "nonfinite JSON value",
                '{"description":NaN}',
                OpenAIResponsesStatus.COMPLETED,
                30,
                "schema_decode",
            ),
        )
        for name, output, status, tokens, stage in cases:
            with self.subTest(name=name):
                generator, client, request = self._setup_generation(
                    output, status=status, output_tokens=tokens,
                )
                with self.assertRaises(AgentRigError) as raised:
                    asyncio.run(generator.generate(request, context()))
                failure = raised.exception.failure
                self.assertEqual(failure.kind, FailureKind.PERMANENT_PROVIDER)
                self.assertEqual(failure.code, "openai.responses.invalid_output")
                self.assertFalse(failure.is_retryable)
                self.assertEqual(dict(failure.metadata), {
                    "output_stage": stage,
                    "response_status": status.value,
                    "input_tokens": "20",
                    "output_tokens": str(tokens),
                    "max_output_tokens": "100",
                })
                for sentinel in ("private-input-sentinel", "private-output-sentinel"):
                    self.assertNotIn(sentinel, repr(failure))
                    self.assertNotIn(sentinel, str(raised.exception))
                with self.assertRaises(TypeError):
                    failure.metadata["input_tokens"] = "0"  # type: ignore[index]
                self.assertEqual(len(client.requests), 1)
                self.assertEqual(client.closed, 1)

    def test_decoder_rejection_is_normalized_once_without_raw_error(self) -> None:
        for error_type in (ValueError, TypeError):
            with self.subTest(error_type=error_type):
                decoded: list[JsonValue] = []

                def reject(value: JsonValue) -> str:
                    decoded.append(value)
                    raise error_type("private-decoder-error")

                rejected_schema = StructuredOutputSchema(
                    schema_id="synthetic.rejection.v1",
                    json_schema=schema().json_schema,
                    decoder=reject,
                )
                generator, client, request = self._setup_generation(
                    '{"description":"private-output-sentinel"}',
                    max_output_tokens=None,
                    output_schema=rejected_schema,
                )
                with self.assertRaises(AgentRigError) as raised:
                    asyncio.run(generator.generate(request, context()))
                failure = raised.exception.failure
                self.assertEqual(dict(failure.metadata), {
                    "output_stage": "schema_decode",
                    "response_status": "completed",
                    "input_tokens": "20",
                    "output_tokens": "30",
                })
                self.assertNotIn("private-decoder-error", repr(failure))
                self.assertNotIn("private-output-sentinel", repr(failure))
                self.assertEqual(len(decoded), 1)
                self.assertEqual(len(client.requests), 1)
                self.assertEqual(client.closed, 1)

    def test_unexpected_decoder_error_does_not_become_retryable_output(self) -> None:
        def broken_decoder(value: JsonValue) -> str:
            raise RuntimeError("private-implementation-error")

        output_schema = StructuredOutputSchema(
            schema_id="synthetic.broken.v1",
            json_schema=schema().json_schema,
            decoder=broken_decoder,
        )
        generator, client, request = self._setup_generation(
            '{"description":"synthetic"}', output_schema=output_schema,
        )
        with self.assertRaises(RuntimeError) as raised:
            asyncio.run(generator.generate(request, context()))
        self.assertNotIsInstance(raised.exception, AgentRigError)
        failure = normalize_exception(raised.exception)
        self.assertEqual(failure.kind, FailureKind.UNEXPECTED)
        self.assertNotEqual(failure.code, "openai.responses.invalid_output")
        self.assertNotIn("private-implementation-error", repr(failure))
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.closed, 1)

    def test_registered_decoder_diagnostic_retains_only_trusted_pair(self) -> None:
        sentinels = tuple(
            "-".join(("private", part, "sentinel"))
            for part in ("decoder", "note", "cause", "output", "input", "forged")
        )
        error = StructuredOutputDecodeError("invalid_region")
        error.args = (sentinels[0],)
        error.add_note(sentinels[1])
        error.__cause__ = RuntimeError(sentinels[2])
        error.decoder_code = sentinels[5]  # type: ignore[attr-defined]
        error.decoder_location = sentinels[5]  # type: ignore[attr-defined]
        decoded: list[JsonValue] = []

        def reject(value: JsonValue) -> str:
            decoded.append(value)
            raise error

        output_schema = StructuredOutputSchema(
            schema_id="synthetic.diagnostic.v1",
            json_schema=schema().json_schema,
            decoder=reject,
            decode_diagnostics={
                "invalid_region": StructuredDecodeDiagnostic(
                    code="region.invalid_bounds", location="/findings/*/region",
                ),
            },
        )
        generator, client, _ = self._setup_generation(
            json.dumps({"description": sentinels[3]}),
            output_schema=output_schema,
        )
        request = StructuredGenerationRequest(
            input=TextGenerationRequest(prompt=sentinels[4], max_output_tokens=100),
            output_schema=output_schema,
        )
        try:
            asyncio.run(generator.generate(request, context()))
        except AgentRigError as raised:
            failure = raised.failure
            printed_traceback = "".join(traceback.format_exception(raised))
            self.assertIs(normalize_exception(raised), failure)
            self.assertIsNone(raised.__cause__)
            self.assertTrue(raised.__suppress_context__)
            for sentinel in sentinels:
                self.assertNotIn(sentinel, repr(failure))
                self.assertNotIn(sentinel, str(raised))
                self.assertNotIn(sentinel, printed_traceback)
        else:
            self.fail("registered decoder rejection must fail generation")

        self.assertEqual(failure.kind, FailureKind.PERMANENT_PROVIDER)
        self.assertEqual(failure.code, "openai.responses.invalid_output")
        self.assertFalse(failure.is_retryable)
        self.assertEqual(dict(failure.metadata), {
            "output_stage": "schema_decode",
            "response_status": "completed",
            "input_tokens": "20",
            "output_tokens": "30",
            "max_output_tokens": "100",
            "decoder_code": "region.invalid_bounds",
            "decoder_location": "/findings/*/region",
        })
        self.assertEqual(len(decoded), 1)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.closed, 1)

    def test_unknown_or_unregistered_decoder_keys_preserve_legacy_metadata(
        self,
    ) -> None:
        diagnostic = StructuredDecodeDiagnostic(
            code="region.invalid_bounds", location="/findings/*/region",
        )
        for name, key, registry in (
            ("unknown key", "unknown_region", {"invalid_region": diagnostic}),
            ("no registry", "invalid_region", {}),
        ):
            with self.subTest(name=name):
                decoded: list[JsonValue] = []

                def reject(value: JsonValue) -> str:
                    decoded.append(value)
                    raise StructuredOutputDecodeError(key)

                output_schema = StructuredOutputSchema(
                    schema_id="synthetic.unregistered.v1",
                    json_schema=schema().json_schema,
                    decoder=reject,
                    decode_diagnostics=registry,
                )
                generator, client, request = self._setup_generation(
                    '{"description":"synthetic"}', output_schema=output_schema,
                )
                with self.assertRaises(AgentRigError) as raised:
                    asyncio.run(generator.generate(request, context()))
                failure = raised.exception.failure
                self.assertEqual(failure.kind, FailureKind.PERMANENT_PROVIDER)
                self.assertEqual(failure.code, "openai.responses.invalid_output")
                self.assertFalse(failure.is_retryable)
                self.assertEqual(dict(failure.metadata), {
                    "output_stage": "schema_decode",
                    "response_status": "completed",
                    "input_tokens": "20",
                    "output_tokens": "30",
                    "max_output_tokens": "100",
                })
                self.assertEqual(len(decoded), 1)
                self.assertEqual(len(client.requests), 1)
                self.assertEqual(client.closed, 1)

    def test_untrusted_exception_attributes_are_never_promoted(self) -> None:
        class ForgedDiagnostic(ValueError):
            diagnostic_key = "invalid_region"
            code = "forged.code"
            location = "/forged/location"
            decoder_code = "forged.decoder_code"
            decoder_location = "/forged/decoder_location"

        class MaliciousDecodeError(StructuredOutputDecodeError):
            def __getattribute__(self, name: str) -> object:
                if name == "diagnostic_key":
                    raise AssertionError("subclass diagnostic must not be read")
                return super().__getattribute__(name)

        for error in (ForgedDiagnostic(), MaliciousDecodeError("invalid_region")):
            with self.subTest(error_type=type(error).__name__):
                decoded: list[JsonValue] = []

                def reject(value: JsonValue) -> str:
                    decoded.append(value)
                    raise error

                output_schema = StructuredOutputSchema(
                    schema_id="synthetic.untrusted.v1",
                    json_schema=schema().json_schema,
                    decoder=reject,
                    decode_diagnostics={
                        "invalid_region": StructuredDecodeDiagnostic(
                            code="region.invalid_bounds",
                            location="/findings/*/region",
                        ),
                    },
                )
                generator, client, request = self._setup_generation(
                    '{"description":"synthetic"}', output_schema=output_schema,
                )
                with self.assertRaises(AgentRigError) as raised:
                    asyncio.run(generator.generate(request, context()))
                failure = raised.exception.failure
                self.assertEqual(failure.kind, FailureKind.PERMANENT_PROVIDER)
                self.assertEqual(failure.code, "openai.responses.invalid_output")
                self.assertFalse(failure.is_retryable)
                self.assertEqual(dict(failure.metadata), {
                    "output_stage": "schema_decode",
                    "response_status": "completed",
                    "input_tokens": "20",
                    "output_tokens": "30",
                    "max_output_tokens": "100",
                })
                self.assertNotIn("forged", repr(failure))
                self.assertEqual(len(decoded), 1)
                self.assertEqual(len(client.requests), 1)
                self.assertEqual(client.closed, 1)

    def test_valid_incomplete_output_retains_length_finish_reason(self) -> None:
        decoded: list[JsonValue] = []

        def decode(value: JsonValue) -> str:
            decoded.append(value)
            return schema().decode(value)

        output_schema = StructuredOutputSchema(
            schema_id="synthetic.incomplete.v1",
            json_schema=schema().json_schema,
            decoder=decode,
        )
        generator, client, request = self._setup_generation(
            '{"description":"synthetic"}',
            status=OpenAIResponsesStatus.INCOMPLETE,
            output_schema=output_schema,
        )
        result = asyncio.run(generator.generate(request, context()))
        self.assertEqual(result.output, "synthetic")
        self.assertEqual(result.finish_reason, TextGenerationFinishReason.LENGTH)
        self.assertEqual(result.usage.total_tokens, 50)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.closed, 1)

    def test_cancelled_request_never_consumes_an_invalid_output(self) -> None:
        generator, client, request = self._setup_generation("invalid JSON")
        source = CancellationSource()
        source.cancel()
        with self.assertRaises(RunCancelled):
            asyncio.run(generator.generate(request, context(source=source)))
        self.assertEqual(client.requests, [])
        self.assertEqual(client.closed, 0)

    def test_mismatched_resolution_fails_before_client_creation(self) -> None:
        requested = image()
        other = ArtifactRef(
            artifact_id=ArtifactId("image-2"),
            kind="image",
            media_type="image/png",
            producer_run_id=RunId("producer-1"),
            workspace_path="private/other.png",
        )
        client = Client(
            OpenAIResponsesResult(
                output_text='{"description":"unused"}',
                model="vision-model",
                status=OpenAIResponsesStatus.COMPLETED,
                input_tokens=0,
                output_tokens=0,
            )
        )
        factory = Factory(client)
        generator = OpenAIResponsesStructuredGenerator[str](
            client_factory=factory,
            artifact_resolver=Resolver(other),
            model="vision-model",
        )
        request = StructuredGenerationRequest(
            input=TextGenerationRequest(
                prompt="Observe.",
                input_artifacts=(requested,),
            ),
            output_schema=schema(),
        )

        with self.assertRaisesRegex(Exception, "mismatched"):
            asyncio.run(generator.generate(request, context()))
        self.assertEqual(factory.calls, 0)


if __name__ == "__main__":
    unittest.main()
