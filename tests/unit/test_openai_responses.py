from __future__ import annotations

import asyncio
import unittest
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agentrig.capabilities import (
    DataRetention,
    StructuredGenerationRequest,
    StructuredOutputSchema,
    TextGenerationRequest,
)
from agentrig.core import (
    ArtifactId,
    ArtifactRef,
    CancellationSource,
    JsonValue,
    ResolvedArtifact,
    RunContext,
    RunId,
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
