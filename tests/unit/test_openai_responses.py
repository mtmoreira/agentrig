from __future__ import annotations

import asyncio
import json
import unittest
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from unittest.mock import patch

from agentrig.capabilities import (
    DataRetention,
    StructuredGenerationRequest,
    StructuredOutputSchema,
    TextGenerationRequest,
)
from agentrig.core import (
    ArtifactId,
    AgentRigError,
    ArtifactRef,
    CancellationSource,
    JsonValue,
    ResolvedArtifact,
    RunContext,
    RunId,
    PrivateTraceCapture,
    PrivateTraceRecord,
    FailureKind,
    InMemoryEventSink,
)
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
    def test_hanging_close_is_bounded_and_available_response_is_already_captured(self) -> None:
        records: list[PrivateTraceRecord] = []

        class Recorder:
            def record(self, record: PrivateTraceRecord) -> None:
                records.append(record)

        class HangingCloseClient(Client):
            close_cancelled = False

            async def close(self) -> None:
                try:
                    await asyncio.Event().wait()
                finally:
                    self.close_cancelled = True

        client = HangingCloseClient(OpenAIResponsesResult(
            output_text='{"description":"fixture"}', model='vision-model',
            status=OpenAIResponsesStatus.COMPLETED, input_tokens=20, output_tokens=5))
        generator = OpenAIResponsesStructuredGenerator[str](
            client_factory=Factory(client), artifact_resolver=Resolver(image()), model='vision-model')
        request = StructuredGenerationRequest(input=TextGenerationRequest(prompt='synthetic input'), output_schema=schema())
        sink = InMemoryEventSink()
        run = replace(context(), event_sink=sink,
            private_trace_capture=PrivateTraceCapture(recorder=Recorder()))
        with patch('agentrig.integrations.openai.responses._CLIENT_CLOSE_GRACE_SECONDS', 0.001):
            with self.assertRaises(AgentRigError) as raised:
                asyncio.run(generator.generate(request, run))
        self.assertEqual(raised.exception.failure.code, 'openai.responses.client_close_timeout')
        self.assertEqual(raised.exception.failure.kind, FailureKind.DEADLINE_EXCEEDED)
        self.assertTrue(client.close_cancelled)
        self.assertEqual(json.loads(records[-1].content)['output_text'], '{"description":"fixture"}')
        self.assertEqual(sink.events[-1].attributes['input_tokens'], 20)
        self.assertEqual(sink.events[-1].attributes['cleanup_status'], 'timeout')

    def test_private_capture_survives_structured_output_rejection_without_image_bytes(self) -> None:
        records: list[PrivateTraceRecord] = []

        class Recorder:
            def record(self, record: PrivateTraceRecord) -> None:
                records.append(record)

        client = Client(OpenAIResponsesResult(
            output_text="synthetic invalid JSON", model="vision-model",
            status=OpenAIResponsesStatus.COMPLETED, input_tokens=20, output_tokens=5,
            cached_input_tokens=2, reasoning_tokens=1, reasoning_summary="synthetic summary"))
        generator = OpenAIResponsesStructuredGenerator[str](
            client_factory=Factory(client), artifact_resolver=Resolver(image()), model="vision-model")
        request = StructuredGenerationRequest(input=TextGenerationRequest(
            prompt="synthetic input", input_artifacts=(image(),)), output_schema=schema())
        run = replace(context(), private_trace_capture=PrivateTraceCapture(
            recorder=Recorder(), include_reasoning=True))
        with self.assertRaises(AgentRigError) as raised:
            asyncio.run(generator.generate(request, run))
        self.assertEqual(raised.exception.failure.code, "openai.responses.invalid_output")
        self.assertEqual([record.kind for record in records], ["provider.request", "provider.response"])
        self.assertEqual(json.loads(records[1].content)["output_text"], "synthetic invalid JSON")
        self.assertEqual(json.loads(records[1].content)["reasoning_summary"], "synthetic summary")
        self.assertNotIn("private-image", records[0].content)
        self.assertTrue(client.requests[0].capture_reasoning)

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
