from __future__ import annotations

import asyncio
import os
import unittest
from collections.abc import Mapping
from pathlib import Path

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
from agentrig.integrations.openai import (
    CODEX_AGENT_RUNTIME_CAPABILITY,
    CodexAgentRuntime,
    CodexSandboxMode,
    CodexSandboxPolicy,
)
from agentrig.integrations.openai.sdk import CodexSdkClientFactory

_MODEL_ENVIRONMENT_VARIABLE = "AGENTRIG_CODEX_LIVE_MODEL"
_API_KEY_ENVIRONMENT_VARIABLE = "AGENTRIG_CODEX_LIVE_API_KEY"
_DEFAULT_MODEL = "gpt-5.6-terra"
_OUTPUT_SCHEMA_ID = "agentrig.live.codex_result.v1"
_PRIVATE_SENTINEL = "private-live-prompt-sentinel"
_OUTPUT_SCHEMA: Mapping[str, JsonValue] = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "read_only": {"type": "boolean"},
    },
    "required": ["status", "read_only"],
    "additionalProperties": False,
}


def _model() -> str:
    model = os.environ.get(_MODEL_ENVIRONMENT_VARIABLE, _DEFAULT_MODEL)
    if not model or model != model.strip():
        raise ValueError(
            f"{_MODEL_ENVIRONMENT_VARIABLE} must be nonempty and trimmed"
        )
    return model


class ApplicationAuthenticationSource:
    def resolve_environment(self) -> dict[str, str]:
        api_key = os.environ.get(_API_KEY_ENVIRONMENT_VARIABLE)
        if api_key is None or not api_key or api_key != api_key.strip():
            raise RuntimeError(
                f"{_API_KEY_ENVIRONMENT_VARIABLE} must be nonempty and trimmed"
            )
        return {"OPENAI_API_KEY": api_key}


def _contract() -> AgentContract[object, object]:
    return AgentContract(
        agent_id="live.codex.structured_output",
        version="1",
        purpose="Validate the live bounded Codex runtime",
        input_schema="agentrig.live.codex_request.v1",
        output_schema=_OUTPUT_SCHEMA_ID,
        prompt_version="1",
        effect_profile=EffectProfile.READ_ONLY,
        limits=AgentLimits(max_turns=1, max_tool_calls=0),
        stopping_policy="structured_output_produced",
        allowed_capabilities=(
            CODEX_AGENT_RUNTIME_CAPABILITY.capability_id,
        ),
        permissions={
            "workspace": "read_only",
            "network": "denied",
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


class CodexRuntimeLiveTest(unittest.TestCase):
    def test_returns_strict_output_through_a_bounded_live_turn(self) -> None:
        context = _context()
        if not isinstance(context.event_sink, InMemoryEventSink):
            raise AssertionError("live test requires an in-memory event sink")
        runtime = CodexAgentRuntime(
            client_factory=CodexSdkClientFactory(
                authentication_source=ApplicationAuthenticationSource()
            ),
            model=_model(),
            sandbox=CodexSandboxPolicy(
                mode=CodexSandboxMode.READ_ONLY,
                cwd=str(Path.cwd().resolve()),
                network_access=False,
            ),
            output_schemas={_OUTPUT_SCHEMA_ID: _OUTPUT_SCHEMA},
            ephemeral=True,
        )
        request = AgentExecutionRequest(
            contract=_contract(),
            instructions=(
                "Return one JSON object matching the output schema. Do not "
                "inspect files, execute commands, use tools, or access the "
                "network. Set status to agentrig-live-ok and read_only to true."
            ),
            input={
                "task": "Return the requested validation object.",
                "private_marker": _PRIVATE_SENTINEL,
            },
        )

        execution = asyncio.run(runtime.execute(request, context))

        if not execution.result.is_success:
            failure = execution.result.failure
            if failure is None:
                raise AssertionError("failed live result has no failure")
            self.fail(
                "live Codex run failed with normalized "
                f"kind={failure.kind.value} code={failure.code or 'none'}"
            )

        output = execution.result.unwrap()
        self.assertIsInstance(output, Mapping)
        if not isinstance(output, Mapping):
            raise AssertionError("live Codex output is not an object")
        self.assertEqual(output["status"], "agentrig-live-ok")
        self.assertIs(output["read_only"], True)
        self.assertEqual(execution.provider_metadata["provider"], "openai.codex")
        self.assertTrue(execution.provider_metadata["thread_id"])
        self.assertTrue(execution.provider_metadata["turn_id"])

        events = context.event_sink.events
        self.assertGreaterEqual(len(events), 2)
        kinds = tuple(event.kind for event in events)
        self.assertEqual(kinds[0], EventKind.PROVIDER_CALL_STARTED)
        self.assertEqual(kinds[-1], EventKind.PROVIDER_CALL_COMPLETED)
        self.assertIn(EventKind.USAGE_REPORTED, kinds)
        self.assertNotIn(EventKind.TOOL_CALL_STARTED, kinds)
        self.assertNotIn(EventKind.TOOL_CALL_COMPLETED, kinds)
        self.assertNotIn(EventKind.APPROVAL_REQUESTED, kinds)
        self.assertTrue(
            set(kinds).issubset(
                {
                    EventKind.PROVIDER_CALL_STARTED,
                    EventKind.PROGRESS_REPORTED,
                    EventKind.USAGE_REPORTED,
                    EventKind.PROVIDER_CALL_COMPLETED,
                }
            )
        )
        serialized_events = "\n".join(event.to_json() for event in events)
        self.assertNotIn(_PRIVATE_SENTINEL, serialized_events)
        self.assertNotIn("agentrig-live-ok", serialized_events)


if __name__ == "__main__":
    unittest.main()
