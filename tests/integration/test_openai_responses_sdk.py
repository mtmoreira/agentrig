from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from agentrig.core import AgentRigError, FailureKind
from agentrig.integrations.openai import (
    OpenAIResponsesImage,
    OpenAIResponsesMessage,
    OpenAIResponsesRequest,
    OpenAIResponsesStatus,
)
from agentrig.capabilities import TextMessageRole
from agentrig.integrations.openai.responses_sdk import OpenAIResponsesSdkClientFactory


@dataclass
class RawResponses:
    owner: RawClient

    async def create(self, **kwargs: Any) -> object:
        self.owner.calls.append(dict(kwargs))
        if self.owner.error is not None:
            raise self.owner.error
        if self.owner.response is None:
            raise AssertionError("raw response required")
        return self.owner.response


class RawClient:
    def __init__(
        self,
        *,
        response: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []
        self.closed = False
        self.responses = RawResponses(self)

    async def close(self) -> None:
        self.closed = True


def request() -> OpenAIResponsesRequest:
    return OpenAIResponsesRequest(
        model="vision-model",
        messages=(
            OpenAIResponsesMessage(
                role=TextMessageRole.USER,
                text="private prompt",
                images=(
                    OpenAIResponsesImage(
                        media_type="image/png",
                        content=b"private-image",
                    ),
                ),
            ),
        ),
        schema_name="photo_observation_v1",
        output_schema={
            "type": "object",
            "properties": {"description": {"type": "string"}},
            "required": ["description"],
            "additionalProperties": False,
        },
        max_output_tokens=100,
    )


class OpenAIResponsesSdkBridgeTest(unittest.TestCase):
    def test_resolves_authentication_only_at_creation(self) -> None:
        private_key = "private-api-key"
        captured: list[str] = []

        class Source:
            calls = 0

            def resolve_api_key(self) -> str:
                self.calls += 1
                return private_key

        source = Source()
        factory = OpenAIResponsesSdkClientFactory(
            authentication_source=source,
            raw_client_builder=lambda key: captured.append(key) or RawClient(),
        )
        self.assertEqual(source.calls, 0)
        client = factory.create()
        self.assertEqual(source.calls, 1)
        self.assertEqual(captured, [private_key])
        self.assertNotIn(private_key, repr(factory))
        self.assertNotIn(private_key, repr(client))

    def test_maps_stateless_tool_free_strict_multimodal_request(self) -> None:
        raw = RawClient(
            response=SimpleNamespace(
                status="completed",
                output_text='{"description":"two people"}',
                model="vision-model",
                usage=SimpleNamespace(input_tokens=20, output_tokens=5),
            )
        )
        client = OpenAIResponsesSdkClientFactory(
            authentication_source=_Source(),
            raw_client_builder=lambda key: raw,
        ).create()

        result = asyncio.run(client.create(request()))
        asyncio.run(client.close())

        self.assertEqual(result.status, OpenAIResponsesStatus.COMPLETED)
        self.assertEqual(result.input_tokens, 20)
        call = raw.calls[0]
        self.assertIs(call["store"], False)
        self.assertEqual(call["tools"], [])
        self.assertEqual(call["truncation"], "disabled")
        text = call["text"]
        self.assertIsInstance(text, dict)
        self.assertIs(text["format"]["strict"], True)  # type: ignore[index]
        image_url = call["input"][0]["content"][1]["image_url"]  # type: ignore[index]
        self.assertTrue(image_url.startswith("data:image/png;base64,"))
        self.assertTrue(raw.closed)

    def test_safe_authentication_and_transport_failures(self) -> None:
        private_value = "private-failure"

        class BadSource:
            def resolve_api_key(self) -> str:
                raise RuntimeError(private_value)

        with self.assertRaises(AgentRigError) as auth:
            OpenAIResponsesSdkClientFactory(
                authentication_source=BadSource(),
                raw_client_builder=lambda key: RawClient(),
            ).create()
        self.assertEqual(auth.exception.failure.kind, FailureKind.PERMANENT_PROVIDER)
        self.assertNotIn(private_value, repr(auth.exception.failure))

        error = RuntimeError(private_value)
        raw = RawClient(error=error)
        client = OpenAIResponsesSdkClientFactory(
            authentication_source=_Source(),
            raw_client_builder=lambda key: raw,
        ).create()
        with self.assertRaises(AgentRigError) as transport:
            asyncio.run(client.create(request()))
        self.assertEqual(
            transport.exception.failure.kind,
            FailureKind.TRANSIENT_PROVIDER,
        )
        self.assertNotIn(private_value, repr(transport.exception.failure))


class _Source:
    def resolve_api_key(self) -> str:
        return "test-key"


if __name__ == "__main__":
    unittest.main()
