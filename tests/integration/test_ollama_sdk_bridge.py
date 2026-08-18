from __future__ import annotations

import asyncio
import gc
import unittest
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from agentrig.core import AgentRigError, FailureKind
from agentrig.integrations.ollama import (
    OllamaChatMessage,
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaClient,
    OllamaFinishReason,
)
from agentrig.integrations.ollama.sdk import OllamaSdkClientFactory


@dataclass
class FakeRawClient:
    response: object | None = None
    error: Exception | None = None

    def __post_init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.closed = False

    async def chat(self, **kwargs: Any) -> object:
        self.calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        if self.response is None:
            raise AssertionError("fake response is required")
        return self.response

    async def close(self) -> None:
        self.closed = True


def request() -> OllamaChatRequest:
    return OllamaChatRequest(
        model="qwen3:8b",
        messages=(
            OllamaChatMessage(role="system", content="private instructions"),
            OllamaChatMessage(role="user", content="private input"),
        ),
        output_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        options={"temperature": 0.2, "num_predict": 64},
        keep_alive="5m",
    )


class OllamaSdkBridgeTest(unittest.TestCase):
    def test_host_is_exact_and_rejects_embedded_credentials(self) -> None:
        captured: list[tuple[str, Mapping[str, str]]] = []

        def build(host: str, headers: Mapping[str, str]) -> FakeRawClient:
            captured.append((host, dict(headers)))
            return FakeRawClient()

        factory = OllamaSdkClientFactory(
            host="http://127.0.0.1:11434/api",
            raw_client_builder=build,
        )
        factory.create()

        self.assertEqual(captured, [("http://127.0.0.1:11434/api", {})])
        for invalid in (
            "127.0.0.1:11434",
            " http://127.0.0.1:11434",
            "ftp://127.0.0.1:11434",
            "http://user:private@127.0.0.1:11434",
            "http://127.0.0.1:11434?token=private",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    OllamaSdkClientFactory(host=invalid)

    def test_resolves_and_copies_authentication_only_at_creation(self) -> None:
        private_value = "Bearer private-authentication-value"
        source_headers = {"Authorization": private_value}
        captured: list[tuple[str, Mapping[str, str]]] = []

        class AuthenticationSource:
            calls = 0

            def resolve_headers(self) -> Mapping[str, str]:
                self.calls += 1
                return source_headers

        def build(host: str, headers: Mapping[str, str]) -> FakeRawClient:
            captured.append((host, dict(headers)))
            return FakeRawClient()

        source = AuthenticationSource()
        factory = OllamaSdkClientFactory(
            host="https://ollama.example.invalid",
            authentication_source=source,
            raw_client_builder=build,
        )

        self.assertEqual(source.calls, 0)
        client = factory.create()
        source_headers["Authorization"] = "changed"

        self.assertEqual(source.calls, 1)
        self.assertEqual(captured[0][1], {"Authorization": private_value})
        self.assertNotIn(private_value, repr(factory))
        self.assertNotIn(private_value, repr(client))

    def test_authentication_failure_is_safe_and_skips_builder(self) -> None:
        private_value = "private-authentication-failure"
        builder_calls = 0

        class AuthenticationSource:
            def resolve_headers(self) -> Mapping[str, str]:
                raise RuntimeError(private_value)

        def build(host: str, headers: Mapping[str, str]) -> FakeRawClient:
            nonlocal builder_calls
            del host, headers
            builder_calls += 1
            raise AssertionError("builder must not run")

        factory = OllamaSdkClientFactory(
            host="http://127.0.0.1:11434",
            authentication_source=AuthenticationSource(),
            raw_client_builder=build,
        )

        with self.assertRaises(AgentRigError) as raised:
            factory.create()

        self.assertEqual(builder_calls, 0)
        self.assertEqual(
            raised.exception.failure.code,
            "ollama.authentication_resolution_failed",
        )
        self.assertNotIn(private_value, repr(raised.exception))
        self.assertNotIn(private_value, repr(raised.exception.failure))

    def test_maps_structured_request_and_normalizes_response(self) -> None:
        raw = FakeRawClient(
            response=SimpleNamespace(
                message=SimpleNamespace(content='{"value":"complete"}'),
                model="qwen3:8b",
                done_reason="stop",
                prompt_eval_count=12,
                eval_count=4,
            )
        )
        client = OllamaSdkClientFactory(
            host="http://127.0.0.1:11434",
            raw_client_builder=lambda host, headers: raw,
        ).create()

        async def exercise() -> OllamaChatResponse:
            response = await client.chat(request())
            await client.close()
            return response

        response = asyncio.run(exercise())

        self.assertIsInstance(client, OllamaClient)
        self.assertEqual(response.content, '{"value":"complete"}')
        self.assertEqual(response.model, "qwen3:8b")
        self.assertIs(response.finish_reason, OllamaFinishReason.STOP)
        self.assertEqual(response.input_tokens, 12)
        self.assertEqual(response.output_tokens, 4)
        self.assertTrue(raw.closed)
        self.assertEqual(
            raw.calls,
            [
                {
                    "model": "qwen3:8b",
                    "messages": [
                        {"role": "system", "content": "private instructions"},
                        {"role": "user", "content": "private input"},
                    ],
                    "stream": False,
                    "format": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                    "options": {"temperature": 0.2, "num_predict": 64},
                    "keep_alive": "5m",
                }
            ],
        )

    def test_malformed_response_and_raw_error_are_safely_normalized(self) -> None:
        private_value = "private-provider-error"
        malformed = FakeRawClient(response=SimpleNamespace(private=private_value))
        malformed_client = OllamaSdkClientFactory(
            host="http://127.0.0.1:11434",
            raw_client_builder=lambda host, headers: malformed,
        ).create()

        with self.assertRaises(AgentRigError) as malformed_raised:
            asyncio.run(malformed_client.chat(request()))

        self.assertEqual(
            malformed_raised.exception.failure.code,
            "ollama.invalid_response",
        )
        self.assertNotIn(private_value, repr(malformed_raised.exception))

        class RawResponseError(RuntimeError):
            status_code = 503

        failed = FakeRawClient(error=RawResponseError(private_value))
        failed_client = OllamaSdkClientFactory(
            host="http://127.0.0.1:11434",
            raw_client_builder=lambda host, headers: failed,
        ).create()

        with self.assertRaises(AgentRigError) as failed_raised:
            asyncio.run(failed_client.chat(request()))

        failure = failed_raised.exception.failure
        self.assertIs(failure.kind, FailureKind.TRANSIENT_PROVIDER)
        self.assertEqual(failure.code, "ollama.request_failed")
        self.assertEqual(failure.metadata, {"status_code": "503"})
        self.assertNotIn(private_value, repr(failed_raised.exception))
        self.assertNotIn(private_value, repr(failure))

    def test_default_factory_initializes_and_closes_offline(self) -> None:
        client = OllamaSdkClientFactory(
            host="http://127.0.0.1:11434",
        ).create()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            asyncio.run(client.close())
            gc.collect()

        resource_warnings = [
            warning
            for warning in caught
            if issubclass(warning.category, ResourceWarning)
        ]
        self.assertEqual(resource_warnings, [])


if __name__ == "__main__":
    unittest.main()
