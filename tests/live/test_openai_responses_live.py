from __future__ import annotations

import asyncio
import base64
import os
import unittest
from collections.abc import Mapping

from agentrig.capabilities import (
    StructuredGenerationRequest,
    StructuredOutputSchema,
    TextGenerationRequest,
)
from agentrig.core import (
    ArtifactId,
    ArtifactRef,
    CancellationSource,
    Deadline,
    JsonValue,
    ResolvedArtifact,
    RunContext,
    RunId,
    SystemClock,
    Uuid4IdGenerator,
)
from agentrig.integrations.openai import OpenAIResponsesStructuredGenerator
from agentrig.integrations.openai.responses_sdk import OpenAIResponsesSdkClientFactory
from tests.support.live_test_main import require_live_test_opt_in

_API_KEY_ENVIRONMENT_VARIABLE = "AGENTRIG_OPENAI_LIVE_API_KEY"
_MODEL_ENVIRONMENT_VARIABLE = "AGENTRIG_OPENAI_LIVE_MODEL"
_SYNTHETIC_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8A"
    "AQUBAScY42YAAAAASUVORK5CYII="
)


class Authentication:
    def resolve_api_key(self) -> str:
        value = os.environ.get(_API_KEY_ENVIRONMENT_VARIABLE)
        if value is None or not value or value != value.strip():
            raise RuntimeError(f"{_API_KEY_ENVIRONMENT_VARIABLE} is required")
        return value


class Resolver:
    async def resolve(self, artifact: ArtifactRef) -> ResolvedArtifact:
        return ResolvedArtifact(artifact=artifact, content=_SYNTHETIC_PNG)


def _model() -> str:
    value = os.environ.get(_MODEL_ENVIRONMENT_VARIABLE)
    if value is None or not value or value != value.strip():
        raise RuntimeError(f"{_MODEL_ENVIRONMENT_VARIABLE} is required")
    return value


def _decode(value: JsonValue) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"synthetic_image"}:
        raise ValueError("live output is invalid")
    result = value["synthetic_image"]
    if not isinstance(result, bool):
        raise ValueError("live output is invalid")
    return result


class OpenAIResponsesLiveTest(unittest.TestCase):
    def test_observes_only_a_synthetic_bounded_image(self) -> None:
        require_live_test_opt_in()
        clock = SystemClock()
        context = RunContext.create_root(
            clock=clock,
            id_generator=Uuid4IdGenerator(RunId),
            cancellation=CancellationSource().token,
            deadline=Deadline.after(60.0, clock),
            labels={"test_mode": "live", "fixture_kind": "synthetic"},
        )
        artifact = ArtifactRef(
            artifact_id=ArtifactId("synthetic-pixel"),
            kind="image",
            media_type="image/png",
            producer_run_id=RunId("synthetic-fixture"),
            uri="memory://synthetic-pixel.png",
        )
        schema = StructuredOutputSchema(
            schema_id="agentrig.live.synthetic_image.v1",
            json_schema={
                "type": "object",
                "properties": {"synthetic_image": {"type": "boolean"}},
                "required": ["synthetic_image"],
                "additionalProperties": False,
            },
            decoder=_decode,
        )
        generator = OpenAIResponsesStructuredGenerator[bool](
            client_factory=OpenAIResponsesSdkClientFactory(
                authentication_source=Authentication()
            ),
            artifact_resolver=Resolver(),
            model=_model(),
        )

        result = asyncio.run(
            generator.generate(
                StructuredGenerationRequest(
                    input=TextGenerationRequest(
                        prompt=(
                            "This is a generated one-pixel test image. Return "
                            "synthetic_image=true and no other fields."
                        ),
                        input_artifacts=(artifact,),
                        max_output_tokens=64,
                    ),
                    output_schema=schema,
                ),
                context,
            )
        )

        self.assertIs(result.output, True)
        self.assertIsNotNone(result.usage.total_tokens)
        self.assertGreater(result.usage.total_tokens or 0, 0)


if __name__ == "__main__":
    unittest.main()
