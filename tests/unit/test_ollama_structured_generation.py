from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
import unittest

from agentrig.capabilities import (
    StructuredGenerationRequest, StructuredOutputSchema, TextGenerationRequest,
)
from agentrig.core import (
    ArtifactId, ArtifactRef, CancellationSource, JsonValue, ResolvedArtifact,
    RunContext, RunId,
)
from agentrig.integrations.ollama import (
    OllamaChatRequest, OllamaChatResponse, OllamaClient, OllamaFinishReason,
    OllamaStructuredGenerator,
)


@dataclass(frozen=True)
class Clock:
    def now(self) -> datetime:
        return datetime(2026, 10, 10, tzinfo=UTC)
    def monotonic(self) -> float:
        return 1.0


@dataclass
class Ids:
    def generate(self) -> RunId:
        return RunId("structured-run")


def context() -> RunContext:
    return RunContext.create_root(clock=Clock(), id_generator=Ids(),
        cancellation=CancellationSource().token)


@dataclass
class Resolver:
    async def resolve(self, artifact: ArtifactRef) -> ResolvedArtifact:
        return ResolvedArtifact(artifact=artifact, content=b"synthetic-image")


@dataclass
class Client:
    requests: list[OllamaChatRequest] = field(default_factory=list)
    async def chat(self, request: OllamaChatRequest) -> OllamaChatResponse:
        self.requests.append(request)
        return OllamaChatResponse(content='{"status":"pass"}', model="gemma3:4b",
            finish_reason=OllamaFinishReason.STOP, input_tokens=9, output_tokens=3)
    async def close(self) -> None:
        return None


@dataclass
class Factory:
    client: OllamaClient
    def create(self) -> OllamaClient:
        return self.client


def decode(value: JsonValue) -> str:
    if not isinstance(value, Mapping) or value.get("status") not in {"pass", "warn"}:
        raise ValueError("invalid status")
    return str(value["status"])


class OllamaStructuredGenerationTests(unittest.TestCase):
    def test_multimodal_request_resolves_image_and_decodes_schema(self) -> None:
        image = ArtifactRef(artifact_id=ArtifactId("cover"), kind="image",
            media_type="image/png", producer_run_id=RunId("source"),
            workspace_path="inputs/cover.png")
        schema = StructuredOutputSchema(schema_id="inspection.v1",
            json_schema={"type": "object", "properties": {"status": {
                "type": "string", "enum": ["pass", "warn"]}},
                "required": ["status"], "additionalProperties": False}, decoder=decode)
        client = Client()
        generator = OllamaStructuredGenerator(client_factory=Factory(client),
            artifact_resolver=Resolver(), model="gemma3:4b")

        result = asyncio.run(generator.generate(StructuredGenerationRequest(
            input=TextGenerationRequest(prompt="Inspect this drawing.",
                input_artifacts=(image,)), output_schema=schema), context()))

        self.assertEqual(result.output, "pass")
        self.assertEqual(result.model.provider, "ollama")
        self.assertEqual(client.requests[0].messages[0].images, (b"synthetic-image",))
        self.assertEqual(client.requests[0].output_schema["type"], "object")


if __name__ == "__main__":
    unittest.main()
