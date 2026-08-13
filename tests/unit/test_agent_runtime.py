from __future__ import annotations

import asyncio
import math
import unittest
from dataclasses import dataclass
from datetime import UTC, datetime

from agentrig.agents import (
    AgentContract,
    AgentExecutionRequest,
    AgentExecutionResult,
    AgentLimits,
    AgentRuntime,
    AgentStatus,
)
from agentrig.core import (
    CancellationSource,
    EffectProfile,
    Failure,
    FailureKind,
    RunContext,
    RunId,
)


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 13, 20, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


@dataclass
class SequentialRunIdGenerator:
    next_value: int = 1

    def generate(self) -> RunId:
        run_id = RunId(f"run-{self.next_value}")
        self.next_value += 1
        return run_id


def create_context() -> RunContext:
    source = CancellationSource()
    return RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialRunIdGenerator(),
        cancellation=source.token,
    )


def create_contract() -> AgentContract[str, str]:
    return AgentContract(
        agent_id="writer",
        version="1",
        purpose="Write a bounded response",
        input_schema="example.prompt.v1",
        output_schema="example.response.v1",
        prompt_version="prompt-2",
        effect_profile=EffectProfile.READ_ONLY,
        limits=AgentLimits(max_turns=3, max_tool_calls=1),
        stopping_policy="output_schema_satisfied",
        allowed_tools=("search",),
        permissions={"network": "search_only"},
    )


def create_request() -> AgentExecutionRequest:
    return AgentExecutionRequest(
        contract=create_contract(),
        instructions="Return one concise answer.",
        input={"prompt": "hello"},
        provider_options={"model": "example-1", "temperature": 0.2},
    )


@dataclass(frozen=True)
class EchoRuntime:
    async def execute(
        self,
        request: AgentExecutionRequest,
        context: RunContext,
    ) -> AgentExecutionResult:
        context.cancellation.raise_if_cancelled()
        return AgentExecutionResult.succeeded(
            {"agent_id": request.contract.agent_id},
            provider_metadata={"session_id": "session-1"},
        )


async def run_typed_runtime(
    runtime: AgentRuntime,
    request: AgentExecutionRequest,
    context: RunContext,
) -> AgentExecutionResult:
    return await runtime.execute(request, context)


class AgentExecutionRequestTest(unittest.TestCase):
    def test_copies_and_freezes_encoded_input_and_provider_options(self) -> None:
        input_value = {"messages": [{"text": "hello"}]}
        provider_options = {"temperature": 0.2, "flags": ["strict"]}

        request = AgentExecutionRequest(
            contract=create_contract(),
            instructions="Return one concise answer.",
            input=input_value,
            provider_options=provider_options,
        )
        input_value["messages"] = []
        provider_options["temperature"] = 1.0

        self.assertEqual(
            request.input,
            {"messages": ({"text": "hello"},)},
        )
        self.assertEqual(request.provider_options["temperature"], 0.2)
        with self.assertRaises(TypeError):
            request.provider_options["model"] = "other"  # type: ignore[index]

    def test_requires_contract_instructions_and_json_values(self) -> None:
        with self.assertRaises(TypeError):
            AgentExecutionRequest(
                contract="not-a-contract",  # type: ignore[arg-type]
                instructions="Run",
                input=None,
            )
        with self.assertRaises(ValueError):
            AgentExecutionRequest(
                contract=create_contract(),
                instructions=" padded ",
                input=None,
            )
        with self.assertRaises(ValueError):
            AgentExecutionRequest(
                contract=create_contract(),
                instructions="Run",
                input=object(),  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            AgentExecutionRequest(
                contract=create_contract(),
                instructions="Run",
                input=None,
                provider_options={"temperature": math.inf},
            )


class AgentExecutionResultTest(unittest.TestCase):
    def test_output_is_json_safe_and_provider_metadata_is_separate(self) -> None:
        output = {"answer": ["draft"]}
        provider_metadata = {
            "provider": "example",
            "session_id": "private-session",
        }

        execution = AgentExecutionResult.succeeded(
            output,
            provider_metadata=provider_metadata,
        )
        output["answer"] = []
        provider_metadata["session_id"] = "changed"

        self.assertEqual(execution.result.unwrap(), {"answer": ("draft",)})
        self.assertEqual(execution.result.status, AgentStatus.SUCCEEDED)
        self.assertNotIn("session_id", execution.result.unwrap())
        self.assertEqual(
            execution.provider_metadata["session_id"],
            "private-session",
        )
        with self.assertRaises(TypeError):
            execution.provider_metadata["session_id"] = "other"  # type: ignore[index]

    def test_failure_remains_normalized_and_metadata_stays_available(self) -> None:
        failure = Failure(
            kind=FailureKind.TRANSIENT_PROVIDER,
            message="provider temporarily unavailable",
            code="provider.busy",
        )

        execution = AgentExecutionResult.from_failure(
            failure,
            provider_metadata={"provider": "example"},
        )

        self.assertEqual(execution.result.status, AgentStatus.FAILED)
        self.assertIs(execution.result.failure, failure)
        self.assertEqual(execution.provider_metadata["provider"], "example")

    def test_rejects_invalid_result_output_and_metadata(self) -> None:
        with self.assertRaises(TypeError):
            AgentExecutionResult(result="not-a-result")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            AgentExecutionResult.succeeded(object())  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            AgentExecutionResult.succeeded(
                None,
                provider_metadata={"session_id": ""},
            )


class AgentRuntimeContractTest(unittest.TestCase):
    def test_protocol_supports_one_provider_neutral_runtime(self) -> None:
        runtime = EchoRuntime()

        execution = asyncio.run(
            run_typed_runtime(runtime, create_request(), create_context())
        )

        self.assertIsInstance(runtime, AgentRuntime)
        self.assertEqual(
            execution.result.unwrap(),
            {"agent_id": "writer"},
        )


if __name__ == "__main__":
    unittest.main()
