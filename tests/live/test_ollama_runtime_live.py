from __future__ import annotations

import asyncio
import os
import unittest
from collections.abc import Mapping

from agentrig.agents import AgentContract, AgentExecutionRequest, AgentLimits
from agentrig.core import (
    CancellationSource,
    Deadline,
    EffectProfile,
    EventId,
    EventKind,
    InMemoryEventSink,
    JsonValue,
    RunContext,
    RunId,
    SystemClock,
    Uuid4IdGenerator,
)
from agentrig.integrations.ollama import (
    OLLAMA_AGENT_RUNTIME_CAPABILITY,
    OllamaAgentRuntime,
    OllamaRuntimeOptions,
)
from agentrig.integrations.ollama.sdk import OllamaSdkClientFactory

_HOST_ENVIRONMENT_VARIABLE = "AGENTRIG_OLLAMA_LIVE_HOST"
_MODEL_ENVIRONMENT_VARIABLE = "AGENTRIG_OLLAMA_LIVE_MODEL"
_API_KEY_ENVIRONMENT_VARIABLE = "AGENTRIG_OLLAMA_LIVE_API_KEY"
_OUTPUT_SCHEMA_ID = "agentrig.live.ollama_result.v1"
_PRIVATE_SENTINEL = "private-ollama-live-prompt-sentinel"
_OUTPUT_SCHEMA: Mapping[str, JsonValue] = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "preserves_unknown": {"type": "boolean"},
    },
    "required": ["status", "preserves_unknown"],
    "additionalProperties": False,
}


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value or value != value.strip():
        raise RuntimeError(f"{name} must be nonempty and trimmed")
    return value


class ApplicationAuthenticationSource:
    def resolve_headers(self) -> dict[str, str]:
        api_key = _required_environment(_API_KEY_ENVIRONMENT_VARIABLE)
        return {"Authorization": f"Bearer {api_key}"}


def _authentication_source() -> ApplicationAuthenticationSource | None:
    if _API_KEY_ENVIRONMENT_VARIABLE not in os.environ:
        return None
    return ApplicationAuthenticationSource()


def _contract() -> AgentContract[object, object]:
    return AgentContract(
        agent_id="live.ollama.structured_output",
        version="1",
        purpose="Validate the live bounded Ollama runtime",
        input_schema="agentrig.live.ollama_request.v1",
        output_schema=_OUTPUT_SCHEMA_ID,
        prompt_version="1",
        effect_profile=EffectProfile.READ_ONLY,
        limits=AgentLimits(max_turns=1, max_tool_calls=0),
        stopping_policy="structured_output_produced",
        allowed_capabilities=(
            OLLAMA_AGENT_RUNTIME_CAPABILITY.capability_id,
        ),
        permissions={
            "workspace": "denied",
            "network": "allowed",
        },
    )


def _context() -> RunContext:
    clock = SystemClock()
    cancellation = CancellationSource()
    return RunContext.create_root(
        clock=clock,
        id_generator=Uuid4IdGenerator(RunId),
        cancellation=cancellation.token,
        event_sink=InMemoryEventSink(),
        event_id_generator=Uuid4IdGenerator(EventId),
        deadline=Deadline.after(120.0, clock),
        labels={"test_mode": "live"},
    )


class OllamaRuntimeLiveTest(unittest.TestCase):
    def test_returns_strict_output_through_a_bounded_live_turn(self) -> None:
        host = _required_environment(_HOST_ENVIRONMENT_VARIABLE)
        model = _required_environment(_MODEL_ENVIRONMENT_VARIABLE)
        context = _context()
        if not isinstance(context.event_sink, InMemoryEventSink):
            raise AssertionError("live test requires an in-memory event sink")
        runtime = OllamaAgentRuntime(
            client_factory=OllamaSdkClientFactory(
                host=host,
                authentication_source=_authentication_source(),
            ),
            model=model,
            output_schemas={_OUTPUT_SCHEMA_ID: _OUTPUT_SCHEMA},
            options=OllamaRuntimeOptions(
                temperature=0,
                seed=7,
                max_output_tokens=64,
                think=False,
            ),
        )
        request = AgentExecutionRequest(
            contract=_contract(),
            instructions=(
                "Return one JSON object matching the output schema. Set status "
                "to agentrig-live-ok and preserves_unknown to true. Do not use "
                "tools or invent any other fields."
            ),
            input={
                "task": "Return the requested validation object.",
                "unstated_detail": None,
                "private_marker": _PRIVATE_SENTINEL,
            },
        )

        execution = asyncio.run(runtime.execute(request, context))

        if not execution.result.is_success:
            failure = execution.result.failure
            if failure is None:
                raise AssertionError("failed live result has no failure")
            self.fail(
                "live Ollama run failed with normalized "
                f"kind={failure.kind.value} code={failure.code or 'none'}"
            )

        output = execution.result.unwrap()
        self.assertIsInstance(output, Mapping)
        if not isinstance(output, Mapping):
            raise AssertionError("live Ollama output is not an object")
        if output.get("status") != "agentrig-live-ok":
            self.fail("live Ollama output violated the status contract")
        if output.get("preserves_unknown") is not True:
            self.fail("live Ollama output violated the unknown-state contract")
        self.assertEqual(execution.provider_metadata["provider"], "ollama")
        self.assertEqual(execution.provider_metadata["model"], model)

        events = context.event_sink.events
        self.assertEqual(
            tuple(event.kind for event in events),
            (
                EventKind.PROVIDER_CALL_STARTED,
                EventKind.USAGE_REPORTED,
                EventKind.PROVIDER_CALL_COMPLETED,
            ),
        )
        usage = events[1].attributes
        self.assertGreaterEqual(usage["input_tokens"], 0)
        self.assertGreaterEqual(usage["output_tokens"], 0)
        serialized_events = "\n".join(event.to_json() for event in events)
        if _PRIVATE_SENTINEL in serialized_events:
            self.fail("live Ollama events retained private input")
        if "agentrig-live-ok" in serialized_events:
            self.fail("live Ollama events retained provider output")


if __name__ == "__main__":
    unittest.main()
