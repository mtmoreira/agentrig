from __future__ import annotations

import asyncio
import math
import unittest
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from agentrig.capabilities import (
    StructuredDecodeDiagnostic,
    StructuredGenerationRequest,
    StructuredOutputDecodeError,
    StructuredOutputSchema,
    TextGenerationRequest,
    TextMessageRole,
)
from agentrig.core import (
    AgentRigError,
    ArtifactRef,
    CancellationSource,
    FailureKind,
    JsonValue,
    ResolvedArtifact,
    RunContext,
    RunId,
)
from agentrig.integrations.openai import (
    OpenAIResponsesImage,
    OpenAIResponsesMessage,
    OpenAIResponsesRequest,
    OpenAIResponsesStatus,
    OpenAIResponsesStructuredGenerator,
)
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
    def test_explicit_effort_reaches_sdk_and_unspecified_effort_preserves_omission(self) -> None:
        raw = RawClient(response=SimpleNamespace(
            status="completed", output_text='{"description":"fixture"}', model="vision-model",
            usage=SimpleNamespace(input_tokens=20, output_tokens=5)))
        client = OpenAIResponsesSdkClientFactory(
            authentication_source=_Source(), raw_client_builder=lambda key: raw).create()
        try:
            asyncio.run(client.create(request()))
            baseline = raw.calls[0]
            self.assertNotIn("reasoning", baseline)
            for effort in ("none", "low", "medium", "high", "xhigh", "max"):
                with self.subTest(effort=effort):
                    asyncio.run(client.create(replace(request(), reasoning_effort=effort)))
                    self.assertEqual(raw.calls[-1], {**baseline, "reasoning": {"effort": effort}})
        finally:
            asyncio.run(client.close())
        self.assertTrue(raw.closed)

    def test_usage_details_and_only_returned_reasoning_summary_are_retained(self) -> None:
        raw = RawClient(response=SimpleNamespace(
            status="completed", output_text='{"description":"fixture"}', model="vision-model",
            usage=SimpleNamespace(input_tokens=20, output_tokens=10,
                input_tokens_details=SimpleNamespace(cached_tokens=4),
                output_tokens_details=SimpleNamespace(reasoning_tokens=6)),
            output=[SimpleNamespace(type="reasoning", summary=[
                SimpleNamespace(type="summary_text", text="synthetic returned summary")])],
            id="resp_fixture", created_at=1789819200.125, service_tier="default",
        ))
        client = OpenAIResponsesSdkClientFactory(authentication_source=_Source(),
            raw_client_builder=lambda key: raw).create()
        default = asyncio.run(client.create(request()))
        captured = asyncio.run(client.create(replace(request(), capture_reasoning=True)))
        self.assertEqual(default.cached_input_tokens, 4)
        self.assertEqual(default.reasoning_tokens, 6)
        self.assertIsNone(default.reasoning_summary)
        self.assertEqual(captured.reasoning_summary, "synthetic returned summary")
        self.assertEqual(captured.response_id, "resp_fixture")
        self.assertEqual(captured.created_at, 1789819200.125)
        self.assertEqual(captured.service_tier, "default")
        self.assertNotIn("synthetic returned summary", repr(captured))
        self.assertEqual(raw.calls[0], raw.calls[1])

    def test_registered_decoder_diagnostic_crosses_the_offline_sdk_seam(self) -> None:
        raw = RawClient(
            response=SimpleNamespace(
                status="completed",
                output_text='{"description":"synthetic scene"}',
                model="vision-model",
                usage=SimpleNamespace(input_tokens=20, output_tokens=5),
            )
        )
        built: list[RawClient] = []
        decoded: list[JsonValue] = []

        def build(key: str) -> RawClient:
            built.append(raw)
            return raw

        def reject(value: JsonValue) -> str:
            decoded.append(value)
            raise StructuredOutputDecodeError("invalid_region")

        factory = OpenAIResponsesSdkClientFactory(
            authentication_source=_Source(), raw_client_builder=build,
        )
        output_schema = StructuredOutputSchema(
            schema_id="synthetic.diagnostic.v1",
            json_schema=request().output_schema,
            decoder=reject,
            decode_diagnostics={
                "invalid_region": StructuredDecodeDiagnostic(
                    code="region.invalid_bounds", location="/findings/*/region",
                ),
            },
        )
        generator = OpenAIResponsesStructuredGenerator[str](
            client_factory=factory,
            artifact_resolver=_UnusedResolver(),
            model="vision-model",
        )
        generation_request = StructuredGenerationRequest(
            input=TextGenerationRequest(prompt="synthetic prompt", max_output_tokens=100),
            output_schema=output_schema,
        )
        run_context = RunContext.create_root(
            clock=_Clock(), id_generator=_Ids(),
            cancellation=CancellationSource().token,
        )

        with self.assertRaises(AgentRigError) as raised:
            asyncio.run(generator.generate(generation_request, run_context))

        failure = raised.exception.failure
        self.assertEqual(failure.kind, FailureKind.PERMANENT_PROVIDER)
        self.assertEqual(failure.code, "openai.responses.invalid_output")
        self.assertFalse(failure.is_retryable)
        self.assertEqual(dict(failure.metadata), {
            "output_stage": "schema_decode",
            "response_status": "completed",
            "input_tokens": "20",
            "output_tokens": "5",
            "max_output_tokens": "100",
            "decoder_code": "region.invalid_bounds",
            "decoder_location": "/findings/*/region",
        })
        self.assertEqual(built, [raw])
        self.assertEqual(len(decoded), 1)
        self.assertEqual(len(raw.calls), 1)
        self.assertEqual(raw.calls[0]["model"], "vision-model")
        self.assertEqual(raw.calls[0]["max_output_tokens"], 100)
        self.assertIs(raw.calls[0]["store"], False)
        self.assertEqual(raw.calls[0]["tools"], [])
        self.assertTrue(raw.closed)
        self.assertNotIn("synthetic scene", repr(failure))
        self.assertNotIn("synthetic prompt", repr(failure))

    def test_default_sdk_client_has_one_bounded_attempt_and_late_authentication(
        self,
    ) -> None:
        raw = RawClient(
            response=SimpleNamespace(
                status="completed",
                output_text='{"description":"synthetic scene"}',
                model="vision-model",
                usage=SimpleNamespace(input_tokens=20, output_tokens=5),
            )
        )
        sdk = Mock()
        sdk.AsyncOpenAI.return_value = raw

        class Source:
            calls = 0

            def resolve_api_key(self) -> str:
                self.calls += 1
                return "test-only-key"

        source = Source()
        with patch.dict("sys.modules", {"openai": sdk}):
            factory = OpenAIResponsesSdkClientFactory(authentication_source=source)
            self.assertEqual(source.calls, 0)
            sdk.AsyncOpenAI.assert_not_called()
            client = factory.create()
            sdk.AsyncOpenAI.assert_called_once_with(
                api_key="test-only-key", max_retries=0, timeout=300.0
            )
            self.assertEqual(source.calls, 1)
            self.assertEqual(raw.calls, [])
            self.assertFalse(raw.closed)
            try:
                result = asyncio.run(client.create(request()))
            finally:
                asyncio.run(client.close())

        self.assertEqual(result.status, OpenAIResponsesStatus.COMPLETED)
        self.assertEqual(len(raw.calls), 1)
        self.assertEqual(raw.calls[0]["model"], "vision-model")
        self.assertEqual(raw.calls[0]["max_output_tokens"], 100)
        self.assertTrue(raw.closed)
        self.assertNotIn("test-only-key", repr(factory))
        self.assertNotIn("test-only-key", repr(client))

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

    def test_incomplete_status_and_aggregate_usage_reach_the_typed_seam(self) -> None:
        raw = RawClient(
            response=SimpleNamespace(
                status="incomplete",
                output_text='{"description":"unfinished',
                model="vision-model",
                usage=SimpleNamespace(input_tokens=20, output_tokens=100),
                incomplete_details=SimpleNamespace(reason="private-provider-detail"),
            )
        )
        client = OpenAIResponsesSdkClientFactory(
            authentication_source=_Source(),
            raw_client_builder=lambda key: raw,
        ).create()
        try:
            result = asyncio.run(client.create(request()))
        finally:
            asyncio.run(client.close())

        self.assertEqual(result.status, OpenAIResponsesStatus.INCOMPLETE)
        self.assertEqual(result.input_tokens, 20)
        self.assertEqual(result.output_tokens, 100)
        self.assertNotIn("unfinished", repr(result))
        self.assertNotIn("private-provider-detail", repr(result))
        self.assertEqual(len(raw.calls), 1)
        self.assertTrue(raw.closed)

    def test_invalid_response_envelopes_do_not_become_invalid_output_retries(
        self,
    ) -> None:
        for name, overrides in (
            ("unknown status", {"status": "private-provider-status"}),
            ("missing usage", {"usage": None}),
            ("invalid tokens", {"usage": SimpleNamespace(input_tokens=True, output_tokens=5)}),
            ("missing output", {"output_text": None}),
            ("empty output", {"output_text": ""}),
            ("boolean timestamp", {"created_at": True}),
            ("negative timestamp", {"created_at": -0.25}),
            ("nonfinite timestamp", {"created_at": math.inf}),
        ):
            with self.subTest(name=name):
                fields: dict[str, Any] = {
                    "status": "completed",
                    "output_text": "private-provider-output",
                    "model": "vision-model",
                    "usage": SimpleNamespace(input_tokens=20, output_tokens=5),
                }
                fields.update(overrides)
                raw = RawClient(response=SimpleNamespace(**fields))
                client = OpenAIResponsesSdkClientFactory(
                    authentication_source=_Source(),
                    raw_client_builder=lambda key: raw,
                ).create()
                try:
                    with self.assertRaises(AgentRigError) as raised:
                        asyncio.run(client.create(request()))
                finally:
                    asyncio.run(client.close())

                failure = raised.exception.failure
                self.assertEqual(failure.kind, FailureKind.PERMANENT_PROVIDER)
                self.assertEqual(failure.code, "openai.responses.invalid_response")
                self.assertEqual(dict(failure.metadata), {})
                self.assertNotIn("private-provider", repr(failure))
                self.assertEqual(len(raw.calls), 1)
                self.assertTrue(raw.closed)


class _Source:
    def resolve_api_key(self) -> str:
        return "test-key"


class _UnusedResolver:
    async def resolve(self, artifact: ArtifactRef) -> ResolvedArtifact:
        raise AssertionError("text-only generation must not resolve an artifact")


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 8, 20, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


class _Ids:
    def generate(self) -> RunId:
        return RunId("run-1")


if __name__ == "__main__":
    unittest.main()
