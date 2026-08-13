from __future__ import annotations

import asyncio
import unittest
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agentrig.agents import (
    Agent,
    AgentContract,
    AgentExecutionRequest,
    AgentExecutionResult,
    AgentInputCodec,
    AgentLimits,
    AgentOutputCodec,
    AgentResult,
    AgentStatus,
    ConfiguredAgent,
)
from agentrig.core import (
    CancellationSource,
    Deadline,
    EffectProfile,
    Failure,
    FailureKind,
    JsonValue,
    RunContext,
    RunId,
)


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 13, 21, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


@dataclass
class SequentialRunIdGenerator:
    next_value: int = 1

    def generate(self) -> RunId:
        run_id = RunId(f"run-{self.next_value}")
        self.next_value += 1
        return run_id


def create_context(
    source: CancellationSource | None = None,
    *,
    deadline: Deadline | None = None,
) -> RunContext:
    owned_source = source if source is not None else CancellationSource()
    return RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialRunIdGenerator(),
        cancellation=owned_source.token,
        deadline=deadline,
    )


def create_contract(
    agent_id: str = "writer",
    *,
    input_schema: str = "example.text.v1",
    output_schema: str = "example.text.v1",
    allowed_tools: tuple[str, ...] = ("write",),
    permissions: dict[str, str] | None = None,
) -> AgentContract[str, str]:
    return AgentContract(
        agent_id=agent_id,
        version="1",
        purpose=f"Run the {agent_id} role",
        input_schema=input_schema,
        output_schema=output_schema,
        prompt_version="prompt-1",
        effect_profile=EffectProfile.READ_ONLY,
        limits=AgentLimits(max_turns=3, max_tool_calls=2),
        stopping_policy="output_schema_satisfied",
        allowed_tools=allowed_tools,
        permissions=permissions if permissions is not None else {},
    )


@dataclass(frozen=True)
class TextInputCodec:
    schema_id: str = "example.text.v1"

    def encode(self, value: str) -> JsonValue:
        if not isinstance(value, str):
            raise ValueError("text input required")
        return {"text": value}


@dataclass(frozen=True)
class TextOutputCodec:
    schema_id: str = "example.text.v1"

    def decode(self, value: JsonValue) -> str:
        if not isinstance(value, Mapping) or set(value) != {"text"}:
            raise ValueError("text object required")
        text = value["text"]
        if not isinstance(text, str):
            raise ValueError("text field required")
        return text


@dataclass
class RoutingRuntime:
    requests: list[AgentExecutionRequest] = field(default_factory=list)

    async def execute(
        self,
        request: AgentExecutionRequest,
        context: RunContext,
    ) -> AgentExecutionResult:
        context.cancellation.raise_if_cancelled()
        self.requests.append(request)
        return AgentExecutionResult.succeeded(
            {"text": f"{request.contract.agent_id} complete"},
            provider_metadata={"session_id": f"{request.contract.agent_id}-1"},
        )


def create_agent(
    runtime: RoutingRuntime,
    contract: AgentContract[str, str] | None = None,
) -> ConfiguredAgent[str, str]:
    return ConfiguredAgent(
        runtime=runtime,
        contract=contract if contract is not None else create_contract(),
        instructions="Return one validated text object.",
        input_codec=TextInputCodec(),
        output_codec=TextOutputCodec(),
        provider_options={"model": "fake-model"},
    )


async def run_typed_agent(
    agent: Agent[str, str],
    input: str,
    context: RunContext,
) -> AgentResult[str]:
    return await agent.run(input, context)


class ConfiguredAgentTest(unittest.TestCase):
    def test_same_runtime_backs_distinct_contracts_tools_and_permissions(
        self,
    ) -> None:
        runtime = RoutingRuntime()
        writer = create_agent(
            runtime,
            create_contract(
                "writer",
                allowed_tools=("write",),
                permissions={"workspace": "read_write"},
            ),
        )
        researcher = create_agent(
            runtime,
            create_contract(
                "researcher",
                allowed_tools=("search", "read"),
                permissions={"network": "search_only"},
            ),
        )

        writer_result = asyncio.run(
            run_typed_agent(writer, "draft", create_context())
        )
        researcher_result = asyncio.run(
            run_typed_agent(researcher, "topic", create_context())
        )

        self.assertIsInstance(writer, Agent)
        self.assertEqual(writer_result.unwrap(), "writer complete")
        self.assertEqual(researcher_result.unwrap(), "researcher complete")
        self.assertEqual(
            [request.contract.allowed_tools for request in runtime.requests],
            [("write",), ("search", "read")],
        )
        self.assertEqual(
            runtime.requests[0].contract.permissions["workspace"],
            "read_write",
        )
        self.assertEqual(
            runtime.requests[1].contract.permissions["network"],
            "search_only",
        )

    def test_builds_an_encoded_request_and_drops_provider_metadata(self) -> None:
        runtime = RoutingRuntime()
        provider_options = {"model": "fake-model"}
        agent = ConfiguredAgent(
            runtime=runtime,
            contract=create_contract(),
            instructions="Return one validated text object.",
            input_codec=TextInputCodec(),
            output_codec=TextOutputCodec(),
            provider_options=provider_options,
        )
        provider_options["model"] = "changed"

        result = asyncio.run(agent.run("draft", create_context()))

        self.assertEqual(result.unwrap(), "writer complete")
        self.assertEqual(runtime.requests[0].input, {"text": "draft"})
        self.assertEqual(
            runtime.requests[0].provider_options["model"],
            "fake-model",
        )
        self.assertFalse(hasattr(result, "provider_metadata"))
        self.assertFalse(hasattr(result, "session_id"))

    def test_output_schema_mismatch_is_a_sanitized_contract_failure(self) -> None:
        @dataclass(frozen=True)
        class InvalidOutputRuntime:
            async def execute(
                self,
                request: AgentExecutionRequest,
                context: RunContext,
            ) -> AgentExecutionResult:
                del request, context
                return AgentExecutionResult.succeeded({"unexpected": "value"})

        agent = ConfiguredAgent(
            runtime=InvalidOutputRuntime(),
            contract=create_contract(),
            instructions="Return one validated text object.",
            input_codec=TextInputCodec(),
            output_codec=TextOutputCodec(),
        )

        result = asyncio.run(agent.run("draft", create_context()))

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIsNotNone(result.failure)
        if result.failure is None:
            raise AssertionError("failed result has no failure")
        self.assertEqual(result.failure.kind, FailureKind.INVALID_INPUT)
        self.assertEqual(result.failure.code, "agent.output_schema_mismatch")
        self.assertEqual(
            result.failure.metadata["schema_id"],
            "example.text.v1",
        )
        self.assertNotIn("unexpected", result.failure.message)

    def test_input_schema_mismatch_prevents_runtime_execution(self) -> None:
        runtime = RoutingRuntime()
        agent = create_agent(runtime)

        result = asyncio.run(
            agent.run(123, create_context())  # type: ignore[arg-type]
        )

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(runtime.requests, [])
        self.assertIsNotNone(result.failure)
        if result.failure is None:
            raise AssertionError("failed result has no failure")
        self.assertEqual(result.failure.kind, FailureKind.INVALID_INPUT)
        self.assertEqual(result.failure.code, "agent.input_schema_mismatch")

    def test_normalized_runtime_failure_is_preserved(self) -> None:
        failure = Failure(
            kind=FailureKind.POLICY_REFUSAL,
            message="provider policy refused the request",
            code="provider.policy",
        )

        @dataclass(frozen=True)
        class FailingRuntime:
            async def execute(
                self,
                request: AgentExecutionRequest,
                context: RunContext,
            ) -> AgentExecutionResult:
                del request, context
                return AgentExecutionResult.from_failure(failure)

        agent = ConfiguredAgent(
            runtime=FailingRuntime(),
            contract=create_contract(),
            instructions="Return one validated text object.",
            input_codec=TextInputCodec(),
            output_codec=TextOutputCodec(),
        )

        result = asyncio.run(agent.run("draft", create_context()))

        self.assertIs(result.failure, failure)

    def test_raw_runtime_exception_is_normalized_without_its_message(self) -> None:
        @dataclass(frozen=True)
        class BrokenRuntime:
            async def execute(
                self,
                request: AgentExecutionRequest,
                context: RunContext,
            ) -> AgentExecutionResult:
                del request, context
                raise RuntimeError("password=private")

        agent = ConfiguredAgent(
            runtime=BrokenRuntime(),
            contract=create_contract(),
            instructions="Return one validated text object.",
            input_codec=TextInputCodec(),
            output_codec=TextOutputCodec(),
        )

        result = asyncio.run(agent.run("draft", create_context()))

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIsNotNone(result.failure)
        if result.failure is None:
            raise AssertionError("failed result has no failure")
        self.assertEqual(result.failure.kind, FailureKind.UNEXPECTED)
        self.assertNotIn("private", result.failure.message)

    def test_constraints_prevent_or_override_runtime_success(self) -> None:
        source = CancellationSource()
        source.cancel("caller stopped")
        cancelled_runtime = RoutingRuntime()

        cancelled = asyncio.run(
            create_agent(cancelled_runtime).run(
                "draft",
                create_context(source),
            )
        )

        self.assertEqual(cancelled.status, AgentStatus.CANCELLED)
        self.assertEqual(cancelled_runtime.requests, [])

        expired = Deadline(
            expires_at=datetime(2026, 8, 13, 20, 0, tzinfo=UTC),
            monotonic_deadline=99.0,
        )
        deadline_runtime = RoutingRuntime()
        deadline_result = asyncio.run(
            create_agent(deadline_runtime).run(
                "draft",
                create_context(deadline=expired),
            )
        )

        self.assertEqual(deadline_result.status, AgentStatus.FAILED)
        self.assertEqual(deadline_runtime.requests, [])
        self.assertIsNotNone(deadline_result.failure)
        if deadline_result.failure is None:
            raise AssertionError("failed result has no failure")
        self.assertEqual(
            deadline_result.failure.kind,
            FailureKind.DEADLINE_EXCEEDED,
        )

    def test_configuration_requires_matching_contract_and_codecs(self) -> None:
        runtime = RoutingRuntime()
        with self.assertRaises(ValueError):
            ConfiguredAgent(
                runtime=runtime,
                contract=create_contract(),
                instructions="Run",
                input_codec=TextInputCodec(schema_id="other.input.v1"),
                output_codec=TextOutputCodec(),
            )
        with self.assertRaises(ValueError):
            ConfiguredAgent(
                runtime=runtime,
                contract=create_contract(),
                instructions="Run",
                input_codec=TextInputCodec(),
                output_codec=TextOutputCodec(schema_id="other.output.v1"),
            )
        with self.assertRaises(TypeError):
            ConfiguredAgent(
                runtime="not-a-runtime",  # type: ignore[arg-type]
                contract=create_contract(),
                instructions="Run",
                input_codec=TextInputCodec(),
                output_codec=TextOutputCodec(),
            )
        with self.assertRaises(TypeError):
            asyncio.run(
                create_agent(runtime).run(
                    "draft",
                    "not-a-context",  # type: ignore[arg-type]
                )
            )

    def test_codec_protocols_are_runtime_checkable(self) -> None:
        self.assertIsInstance(TextInputCodec(), AgentInputCodec)
        self.assertIsInstance(TextOutputCodec(), AgentOutputCodec)


if __name__ == "__main__":
    unittest.main()
