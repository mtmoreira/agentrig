from __future__ import annotations

import asyncio
import json
import math
import unittest
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from unittest.mock import patch

from agentrig.agents import (
    AgentContract,
    AgentExecutionRequest,
    AgentLimits,
    AgentRuntime,
)
from agentrig.capabilities import (
    CapabilityFeature,
    CapabilityKind,
    CapabilityRequirements,
    DataRetention,
    TextGenerationRequest,
    TextGenerator,
    TextMessage,
    TextMessageRole,
)
from agentrig.core import (
    CancellationSource,
    Deadline,
    EffectProfile,
    EventId,
    EventKind,
    FailureKind,
    InMemoryEventSink,
    NoOpRedactionPolicy,
    PrivateTraceCapture,
    PrivateTraceRecord,
    AgentRigError,
    Failure,
    RunContext,
    RunId,
)
from agentrig.integrations.ollama import (
    OLLAMA_AGENT_RUNTIME_CAPABILITY,
    OLLAMA_CLIENT_VERSION,
    OLLAMA_TEXT_GENERATION_CAPABILITY,
    OllamaAgentRuntime,
    OllamaAuthenticationSource,
    OllamaChatMessage,
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaClient,
    OllamaFinishReason,
    OllamaRuntimeOptions,
    OllamaTextGenerator,
)


@dataclass(frozen=True)
class FixedClock:
    monotonic_value: float = 100.0

    def now(self) -> datetime:
        return datetime(2026, 8, 17, 12, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return self.monotonic_value


@dataclass
class SequentialIdGenerator:
    prefix: str
    next_value: int = 1

    def generate(self) -> RunId | EventId:
        value = f"{self.prefix}-{self.next_value}"
        self.next_value += 1
        return RunId(value) if self.prefix == "run" else EventId(value)


def create_context(
    *,
    source: CancellationSource | None = None,
    deadline: Deadline | None = None,
) -> tuple[RunContext, InMemoryEventSink]:
    sink = InMemoryEventSink(redaction_policy=NoOpRedactionPolicy())
    owned_source = source if source is not None else CancellationSource()
    return (
        RunContext.create_root(
            clock=FixedClock(),
            id_generator=SequentialIdGenerator("run"),  # type: ignore[arg-type]
            cancellation=owned_source.token,
            event_sink=sink,
            event_id_generator=SequentialIdGenerator("event"),  # type: ignore[arg-type]
            deadline=deadline,
        ),
        sink,
    )


def create_contract(
    *,
    capabilities: tuple[str, ...] | None = None,
    tools: tuple[str, ...] = (),
    permissions: dict[str, str] | None = None,
) -> AgentContract[object, object]:
    return AgentContract(
        agent_id="ollama-test",
        version="1",
        purpose="Exercise one structured Ollama turn",
        input_schema="example.input.v1",
        output_schema="example.output.v1",
        prompt_version="prompt-1",
        effect_profile=EffectProfile.READ_ONLY,
        limits=AgentLimits(max_turns=1, max_tool_calls=0),
        stopping_policy="structured_output",
        allowed_tools=tools,
        allowed_capabilities=(
            capabilities
            if capabilities is not None
            else (OLLAMA_AGENT_RUNTIME_CAPABILITY.capability_id,)
        ),
        permissions=(
            permissions
            if permissions is not None
            else {"workspace": "denied", "network": "allowed"}
        ),
    )


def create_request(
    *,
    contract: AgentContract[object, object] | None = None,
    provider_options: dict[str, object] | None = None,
) -> AgentExecutionRequest:
    return AgentExecutionRequest(
        contract=contract if contract is not None else create_contract(),
        instructions="Return one structured result.",
        input={"private_value": "draft"},
        provider_options=provider_options or {},  # type: ignore[arg-type]
    )


@dataclass
class FakeClient:
    response: OllamaChatResponse | None = None
    error: Exception | None = None
    block: bool = False
    requests: list[OllamaChatRequest] = field(default_factory=list)
    closes: int = 0
    cancelled: bool = False

    async def chat(self, request: OllamaChatRequest) -> OllamaChatResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        if self.block:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        if self.response is None:
            raise AssertionError("fake response required")
        return self.response

    async def close(self) -> None:
        self.closes += 1


@dataclass
class FakeFactory:
    client: FakeClient
    calls: int = 0

    def create(self) -> OllamaClient:
        self.calls += 1
        return self.client


def successful_client(content: str = '{"result":"complete"}') -> FakeClient:
    return FakeClient(
        response=OllamaChatResponse(
            content=content,
            model="gemma-test",
            finish_reason=OllamaFinishReason.STOP,
            input_tokens=12,
            output_tokens=4,
        )
    )


def create_runtime(
    client: FakeClient,
    *,
    options: OllamaRuntimeOptions | None = None,
) -> tuple[OllamaAgentRuntime, FakeFactory]:
    factory = FakeFactory(client)
    runtime = OllamaAgentRuntime(
        client_factory=factory,
        model="gemma-test",
        output_schemas={
            "example.output.v1": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
                "additionalProperties": False,
            }
        },
        options=options,
    )
    return runtime, factory


class OllamaContractTest(unittest.TestCase):
    def test_assistant_messages_preserve_raw_repair_content(self) -> None:
        message = OllamaChatMessage(role="assistant", content="  {bad\n")
        self.assertEqual(message.content, "  {bad\n")
        self.assertNotIn("{bad", repr(message))
        with self.assertRaises(ValueError):
            OllamaChatMessage(role="user", content="  ")

    def test_declares_conservative_structured_runtime_capability(self) -> None:
        descriptor = OLLAMA_AGENT_RUNTIME_CAPABILITY

        self.assertEqual(OLLAMA_CLIENT_VERSION, "0.6.2")
        self.assertEqual(descriptor.capability_id, "ollama.agent_runtime")
        self.assertEqual(descriptor.kind, CapabilityKind.AGENT_RUNTIME)
        self.assertEqual(
            descriptor.features,
            frozenset(
                {
                    CapabilityFeature.CANCELLATION,
                    CapabilityFeature.STRUCTURED_OUTPUT,
                    CapabilityFeature.USAGE_REPORTING,
                }
            ),
        )
        self.assertEqual(descriptor.data_retention, DataRetention.UNKNOWN)
        CapabilityRequirements(
            kind=CapabilityKind.AGENT_RUNTIME,
            features=frozenset({CapabilityFeature.STRUCTURED_OUTPUT}),
        ).require(descriptor)

    def test_authentication_source_is_injected_and_response_repr_is_private(self) -> None:
        class ExampleAuthenticationSource:
            def resolve_headers(self) -> dict[str, str]:
                return {"Authorization": "Bearer private"}

        response = OllamaChatResponse(
            content="private output",
            model="gemma-test",
        )

        self.assertIsInstance(ExampleAuthenticationSource(), OllamaAuthenticationSource)
        self.assertNotIn("private output", repr(response))

    def test_options_are_strict_and_translate_to_detached_provider_values(self) -> None:
        options = OllamaRuntimeOptions(
            temperature=0.2,
            seed=7,
            max_output_tokens=128,
            think="low",
            keep_alive="5m",
        )

        self.assertEqual(
            options.to_provider_options(),
            {"temperature": 0.2, "seed": 7, "num_predict": 128},
        )
        with self.assertRaises(TypeError):
            options.to_provider_options()["seed"] = 8  # type: ignore[index]
        for invalid in (-1.0, math.inf, math.nan):
            with self.subTest(temperature=invalid), self.assertRaises(ValueError):
                OllamaRuntimeOptions(temperature=invalid)
        with self.assertRaises(ValueError):
            OllamaRuntimeOptions(max_output_tokens=0)
        for invalid_think in ("", "enabled", 1, []):
            with self.subTest(think=invalid_think), self.assertRaises(ValueError):
                OllamaRuntimeOptions(think=invalid_think)  # type: ignore[arg-type]

    def test_request_freezes_schema_options_and_omits_private_messages(self) -> None:
        schema: dict[str, object] = {"type": "object"}
        options: dict[str, object] = {"seed": 7}
        request = OllamaChatRequest(
            model="gemma-test",
            messages=(
                OllamaChatMessage(role="user", content="private prompt"),
            ),
            output_schema=schema,  # type: ignore[arg-type]
            options=options,  # type: ignore[arg-type]
        )
        schema["type"] = "string"
        options["seed"] = 8

        self.assertEqual(request.output_schema, {"type": "object"})
        self.assertEqual(request.options, {"seed": 7})
        self.assertIs(request.think, False)
        self.assertNotIn("private prompt", repr(request))


class OllamaTextGeneratorTest(unittest.TestCase):
    def test_private_capture_retains_raw_content_before_caller_decode(self) -> None:
        records: list[PrivateTraceRecord] = []

        class Recorder:
            def record(self, record: PrivateTraceRecord) -> None:
                records.append(record)

        client = successful_client("  {not-json\n")
        assert client.response is not None
        client.response = replace(client.response, thinking="returned reasoning")
        generator = OllamaTextGenerator(
            client_factory=FakeFactory(client),
            model="gemma-test",
        )
        context, _ = create_context()
        context = replace(context, private_trace_capture=PrivateTraceCapture(
            recorder=Recorder(), include_reasoning=True,
        ))

        result = asyncio.run(generator.generate(TextGenerationRequest(
            prompt="Return JSON.",
            output_schema={"type": "object"},
        ), context))

        self.assertEqual(result.text, "  {not-json\n")
        self.assertEqual(
            [record.kind for record in records],
            ["provider.request", "provider.response"],
        )
        self.assertEqual(json.loads(records[1].content)["content"], "  {not-json\n")
        self.assertEqual(json.loads(records[1].content)["thinking"], "returned reasoning")
        self.assertTrue(client.requests[0].capture_reasoning)

    def test_raw_schema_response_and_repair_history_are_not_decoded(self) -> None:
        raw_content = "  {not-json\n"
        client = successful_client(raw_content)
        factory = FakeFactory(client)
        generator = OllamaTextGenerator(
            client_factory=factory,
            model="gemma-test",
            options=OllamaRuntimeOptions(temperature=0, think=False),
        )
        context, _ = create_context()
        request = TextGenerationRequest(
            messages=(
                TextMessage(role=TextMessageRole.SYSTEM, text="Return JSON."),
                TextMessage(role=TextMessageRole.USER, text="First try."),
                TextMessage(role=TextMessageRole.ASSISTANT, text="  {bad\n"),
                TextMessage(role=TextMessageRole.USER, text="Repair it."),
            ),
            output_schema={"type": "object"},
            max_output_tokens=1536,
        )

        result = asyncio.run(generator.generate(request, context))

        self.assertIsInstance(generator, TextGenerator)
        self.assertEqual(generator.descriptor, OLLAMA_TEXT_GENERATION_CAPABILITY)
        self.assertEqual(factory.calls, 1)
        self.assertEqual(result.text, raw_content)
        self.assertEqual(result.usage.input_tokens, 12)
        self.assertEqual(client.closes, 1)
        chat = client.requests[0]
        self.assertEqual(
            [(item.role, item.content) for item in chat.messages],
            [("system", "Return JSON."), ("user", "First try."),
             ("assistant", "  {bad\n"), ("user", "Repair it.")],
        )
        self.assertEqual(chat.output_schema, {"type": "object"})
        self.assertEqual(chat.options, {"temperature": 0.0, "num_predict": 1536})
        self.assertIs(chat.think, False)

    def test_preflight_and_provider_failure_never_retry(self) -> None:
        client = FakeClient(error=AgentRigError(Failure(
            kind=FailureKind.TRANSIENT_PROVIDER,
            code="ollama.request_failed",
            message="Ollama request failed",
        )))
        factory = FakeFactory(client)
        generator = OllamaTextGenerator(client_factory=factory, model="gemma-test")
        context, _ = create_context()

        with self.assertRaises(AgentRigError) as missing_schema:
            asyncio.run(generator.generate(TextGenerationRequest(prompt="hello"), context))
        self.assertEqual(missing_schema.exception.failure.code, "ollama.output_schema_required")
        self.assertEqual(factory.calls, 0)

        request = TextGenerationRequest(prompt="hello", output_schema={"type": "object"})
        with self.assertRaises(AgentRigError) as failed:
            asyncio.run(generator.generate(request, context))
        self.assertEqual(failed.exception.failure.code, "ollama.request_failed")
        self.assertEqual(factory.calls, 1)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.closes, 1)

    def test_inflight_cancellation_is_typed_and_closes_client(self) -> None:
        source = CancellationSource()
        client = FakeClient(block=True)
        factory = FakeFactory(client)
        generator = OllamaTextGenerator(client_factory=factory, model="gemma-test")
        context, _ = create_context(source=source)
        request = TextGenerationRequest(prompt="hello", output_schema={"type": "object"})

        async def run_and_cancel() -> None:
            task = asyncio.create_task(generator.generate(request, context))
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            source.cancel("caller stopped")
            with self.assertRaises(AgentRigError) as stopped:
                await task
            self.assertEqual(stopped.exception.failure.kind, FailureKind.CANCELLED)

        asyncio.run(run_and_cancel())
        self.assertEqual(factory.calls, 1)
        self.assertTrue(client.cancelled)
        self.assertEqual(client.closes, 1)


class OllamaAgentRuntimeTest(unittest.TestCase):
    def test_executes_exact_structured_request_and_emits_safe_usage(self) -> None:
        client = successful_client()
        runtime, factory = create_runtime(
            client,
            options=OllamaRuntimeOptions(
                temperature=0.1,
                seed=11,
                max_output_tokens=64,
                think="low",
                keep_alive="2m",
            ),
        )
        context, sink = create_context()

        result = asyncio.run(runtime.execute(create_request(), context))

        self.assertIsInstance(runtime, AgentRuntime)
        self.assertEqual(factory.calls, 1)
        self.assertEqual(result.result.output, {"result": "complete"})
        self.assertEqual(result.usage.input_tokens, 12)
        self.assertIsNone(result.usage.cached_input_tokens)
        self.assertEqual(result.usage.output_tokens, 4)
        self.assertEqual(result.usage.total_tokens, 16)
        self.assertEqual(
            result.provider_metadata,
            {
                "provider": "ollama",
                "model": "gemma-test",
                "finish_reason": "stop",
                "elapsed_ms": "0",
            },
        )
        request = client.requests[0]
        self.assertEqual(request.model, "gemma-test")
        self.assertEqual(
            tuple(message.role for message in request.messages),
            ("system", "user"),
        )
        self.assertEqual(request.messages[1].content, '{"private_value":"draft"}')
        self.assertEqual(
            request.options,
            {"temperature": 0.1, "seed": 11, "num_predict": 64},
        )
        self.assertEqual(request.think, "low")
        self.assertEqual(request.keep_alive, "2m")
        self.assertEqual(client.closes, 1)
        self.assertEqual(
            [event.kind for event in sink.events],
            [
                EventKind.PROVIDER_CALL_STARTED,
                EventKind.USAGE_REPORTED,
                EventKind.PROVIDER_CALL_COMPLETED,
            ],
        )
        self.assertEqual(
            sink.events[1].attributes["input_tokens"],
            result.usage.input_tokens,
        )
        self.assertEqual(
            sink.events[1].attributes["output_tokens"],
            result.usage.output_tokens,
        )
        self.assertNotIn("draft", repr(sink.events))

    def test_rejects_authority_and_options_before_client_creation(self) -> None:
        client = successful_client()
        runtime, factory = create_runtime(client)
        invalid_requests = (
            create_request(contract=create_contract(capabilities=())),
            create_request(contract=create_contract(tools=("search",))),
            create_request(
                contract=create_contract(
                    permissions={"workspace": "read_only", "network": "allowed"}
                )
            ),
            create_request(
                contract=create_contract(
                    permissions={"workspace": "denied", "network": "denied"}
                )
            ),
            create_request(provider_options={"temperature": 0.5}),
        )

        for index, request in enumerate(invalid_requests):
            with self.subTest(case=index):
                context, sink = create_context()
                result = asyncio.run(runtime.execute(request, context))
                self.assertEqual(result.result.failure.kind, FailureKind.INVALID_INPUT)  # type: ignore[union-attr]
                self.assertEqual(sink.events, ())

        self.assertEqual(factory.calls, 0)

    def test_private_capture_retains_invalid_response_and_native_timings(self) -> None:
        records: list[PrivateTraceRecord] = []

        class Recorder:
            def record(self, record: PrivateTraceRecord) -> None:
                records.append(record)

        client = successful_client("synthetic invalid JSON")
        assert client.response is not None
        client.response = replace(client.response, thinking="synthetic provider-emitted thought",
                                  total_duration_ns=900, load_duration_ns=200,
                                  prompt_eval_duration_ns=300, eval_duration_ns=400)
        runtime, _ = create_runtime(client)
        context, sink = create_context()
        context = replace(context, private_trace_capture=PrivateTraceCapture(
            recorder=Recorder(), include_reasoning=True))
        result = asyncio.run(runtime.execute(create_request(), context))
        self.assertEqual(result.result.failure.code, "ollama.invalid_output")  # type: ignore[union-attr]
        self.assertEqual(result.provider_metadata["eval_duration_ns"], "400")
        self.assertEqual(result.usage.input_tokens, 12)
        self.assertEqual([record.kind for record in records], ["provider.request", "provider.response"])
        self.assertEqual(json.loads(records[1].content)["content"], "synthetic invalid JSON")
        self.assertEqual(json.loads(records[1].content)["thinking"], "synthetic provider-emitted thought")
        self.assertFalse(client.requests[0].think)
        self.assertTrue(client.requests[0].capture_reasoning)
        self.assertNotIn("synthetic", repr(sink.events))

    def test_missing_usage_is_unavailable_and_capture_is_disabled_by_default(self) -> None:
        client = successful_client()
        assert client.response is not None
        client.response = replace(client.response, input_tokens=None, output_tokens=None)
        runtime, _ = create_runtime(client)
        context, _ = create_context()
        result = asyncio.run(runtime.execute(create_request(), context))
        self.assertFalse(result.usage.is_reported)
        self.assertIsNone(result.usage.total_tokens)
        self.assertFalse(client.requests[0].capture_reasoning)

    def test_json_whitespace_is_valid_and_empty_text_is_a_structured_output_failure(self) -> None:
        for text, success in ((" \n{\"result\":\"complete\"}\n", True), ("", False)):
            runtime, _ = create_runtime(successful_client(text))
            context, _ = create_context()
            result = asyncio.run(runtime.execute(create_request(), context))
            self.assertEqual(result.result.is_success, success)
            if not success:
                self.assertEqual(result.result.failure.code, "ollama.invalid_output")  # type: ignore[union-attr]

    def test_hanging_close_is_bounded_and_keeps_usage_or_primary_cancellation(self) -> None:
        class HangingCloseClient(FakeClient):
            close_cancelled = False

            async def close(self) -> None:
                try:
                    await asyncio.Event().wait()
                finally:
                    self.close_cancelled = True

        for primary in (None, AgentRigError(Failure(kind=FailureKind.CANCELLED, message="caller stopped"))):
            client = HangingCloseClient(response=successful_client().response, error=primary)
            runtime, _ = create_runtime(client)
            context, sink = create_context()
            with patch('agentrig.integrations.ollama.runtime._CLIENT_CLOSE_GRACE_SECONDS', 0.001):
                result = asyncio.run(runtime.execute(create_request(), context))
            self.assertTrue(client.close_cancelled)
            self.assertEqual(result.provider_metadata['cleanup_status'], 'timeout')
            self.assertEqual(sink.events[-1].attributes['cleanup_status'], 'timeout')
            if primary is None:
                self.assertEqual(result.result.failure.code, 'ollama.client_close_timeout')  # type: ignore[union-attr]
                self.assertEqual(result.usage.total_tokens, 16)
            else:
                self.assertEqual(result.result.failure.kind, FailureKind.CANCELLED)  # type: ignore[union-attr]

    def test_invalid_output_and_transport_errors_are_sanitized(self) -> None:
        invalid_runtime, _ = create_runtime(successful_client("private prose"))
        context, _ = create_context()
        invalid = asyncio.run(invalid_runtime.execute(create_request(), context))

        failing_client = FakeClient(error=RuntimeError("token=private"))
        failing_runtime, _ = create_runtime(failing_client)
        context, _ = create_context()
        transport = asyncio.run(failing_runtime.execute(create_request(), context))

        self.assertEqual(invalid.result.failure.code, "ollama.invalid_output")  # type: ignore[union-attr]
        self.assertEqual(invalid.result.failure.kind, FailureKind.PERMANENT_PROVIDER)  # type: ignore[union-attr]
        self.assertEqual(invalid.usage.total_tokens, 16)
        self.assertEqual(transport.result.failure.code, "ollama.transport_failed")  # type: ignore[union-attr]
        self.assertFalse(transport.usage.is_reported)
        self.assertNotIn("private", repr(transport.result.failure))
        self.assertEqual(failing_client.closes, 1)

    def test_cancellation_stops_inflight_chat_and_closes_client(self) -> None:
        source = CancellationSource()
        client = FakeClient(block=True)
        runtime, _ = create_runtime(client)
        context, _ = create_context(source=source)

        async def run_and_cancel() -> object:
            task = asyncio.create_task(runtime.execute(create_request(), context))
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            source.cancel("caller stopped")
            return await task

        result = asyncio.run(run_and_cancel())

        self.assertEqual(result.result.failure.kind, FailureKind.CANCELLED)  # type: ignore[attr-defined,union-attr]
        self.assertEqual(result.result.failure.message, "caller stopped")  # type: ignore[attr-defined,union-attr]
        self.assertTrue(client.cancelled)
        self.assertEqual(client.closes, 1)

    def test_expired_deadline_prevents_client_creation(self) -> None:
        runtime, factory = create_runtime(successful_client())
        deadline = Deadline(
            expires_at=datetime(2026, 8, 17, 12, 0, tzinfo=UTC),
            monotonic_deadline=100.0,
        )
        context, _ = create_context(deadline=deadline)

        result = asyncio.run(runtime.execute(create_request(), context))

        self.assertEqual(result.result.failure.kind, FailureKind.DEADLINE_EXCEEDED)  # type: ignore[union-attr]
        self.assertEqual(factory.calls, 0)


if __name__ == "__main__":
    unittest.main()
