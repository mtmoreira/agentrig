from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass
from datetime import UTC, datetime

from agentrig.agents import (
    AgentContract,
    AgentExecutionRequest,
    AgentExecutionResult,
    AgentLimits,
    AgentRuntime,
    AgentRuntimeUsage,
    AgentStatus,
)
from agentrig.core import (
    CancellationSource,
    Deadline,
    DeadlineExceeded,
    EffectProfile,
    EventId,
    EventKind,
    Failure,
    FailureKind,
    InMemoryEventSink,
    NoOpRedactionPolicy,
    RunCancelled,
    RunContext,
    RunId,
)
from agentrig.testing import (
    ScriptedAgentProgress,
    ScriptedAgentRuntime,
    ScriptedAgentScenario,
    ScriptedApprovalRequest,
    ScriptedToolRequest,
)


@dataclass(frozen=True)
class FixedClock:
    monotonic_time: float = 100.0

    def now(self) -> datetime:
        return datetime(2026, 8, 13, 22, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return self.monotonic_time


@dataclass
class SequentialRunIdGenerator:
    next_value: int = 1

    def generate(self) -> RunId:
        run_id = RunId(f"run-{self.next_value}")
        self.next_value += 1
        return run_id


@dataclass
class SequentialEventIdGenerator:
    next_value: int = 1

    def generate(self) -> EventId:
        event_id = EventId(f"event-{self.next_value}")
        self.next_value += 1
        return event_id


def create_contract(
    *,
    allowed_tools: tuple[str, ...] = ("search",),
) -> AgentContract[str, str]:
    return AgentContract(
        agent_id="researcher",
        version="1",
        purpose="Research one bounded topic",
        input_schema="example.topic.v1",
        output_schema="example.answer.v1",
        prompt_version="prompt-1",
        effect_profile=EffectProfile.READ_ONLY,
        limits=AgentLimits(max_turns=3, max_tool_calls=2),
        stopping_policy="output_schema_satisfied",
        allowed_tools=allowed_tools,
        permissions={"network": "search_only"},
    )


def create_request(
    *,
    allowed_tools: tuple[str, ...] = ("search",),
) -> AgentExecutionRequest:
    return AgentExecutionRequest(
        contract=create_contract(allowed_tools=allowed_tools),
        instructions="Return one sourced answer.",
        input={"topic": "runtime contracts"},
    )


def create_context(
    *,
    source: CancellationSource | None = None,
    clock: FixedClock | None = None,
    deadline: Deadline | None = None,
) -> tuple[RunContext, InMemoryEventSink]:
    effective_source = source if source is not None else CancellationSource()
    effective_clock = clock if clock is not None else FixedClock()
    sink = InMemoryEventSink(redaction_policy=NoOpRedactionPolicy())
    context = RunContext.create_root(
        clock=effective_clock,
        id_generator=SequentialRunIdGenerator(),
        cancellation=effective_source.token,
        event_sink=sink,
        event_id_generator=SequentialEventIdGenerator(),
        deadline=deadline,
        correlation={"request_id": "request-1"},
    )
    return context, sink


async def execute_runtime(
    runtime: AgentRuntime,
    request: AgentExecutionRequest,
    context: RunContext,
) -> AgentExecutionResult:
    return await runtime.execute(request, context)


def approval_failure(approval_id: str = "approval-1") -> Failure:
    return Failure(
        kind=FailureKind.APPROVAL_REQUIRED,
        message="agent action requires approval",
        code="agent.approval_required",
        metadata={"approval_id": approval_id},
    )


class ScriptedAgentRuntimeTest(unittest.TestCase):
    def test_emits_progress_and_allowed_tool_lifecycle_in_order(self) -> None:
        expected = AgentExecutionResult.succeeded(
            {"answer": "complete"},
            provider_metadata={"session_id": "session-1"},
        )
        runtime = ScriptedAgentRuntime(
            scenarios=(
                ScriptedAgentScenario(
                    actions=(
                        ScriptedAgentProgress(message="Searching sources"),
                        ScriptedToolRequest(tool_name="search"),
                    ),
                    result=expected,
                ),
            )
        )
        request = create_request()
        context, sink = create_context()

        result = asyncio.run(execute_runtime(runtime, request, context))

        self.assertIsInstance(runtime, AgentRuntime)
        self.assertIs(result, expected)
        self.assertEqual(
            [event.kind for event in sink.events],
            [
                EventKind.PROVIDER_CALL_STARTED,
                EventKind.PROGRESS_REPORTED,
                EventKind.TOOL_CALL_STARTED,
                EventKind.TOOL_CALL_COMPLETED,
                EventKind.PROVIDER_CALL_COMPLETED,
            ],
        )
        self.assertEqual(
            [event.event_id for event in sink.events],
            [EventId(f"event-{index}") for index in range(1, 6)],
        )
        self.assertEqual(
            sink.events[1].attributes["message"],
            "Searching sources",
        )
        self.assertEqual(sink.events[2].attributes["tool_name"], "search")
        self.assertEqual(sink.events[3].attributes["status"], "succeeded")
        self.assertEqual(sink.events[-1].attributes["status"], "succeeded")
        self.assertEqual(sink.events[0].correlation["request_id"], "request-1")
        self.assertEqual(len(runtime.calls), 1)
        self.assertIs(runtime.calls[0].request, request)
        self.assertIs(runtime.calls[0].context, context)
        self.assertTrue(runtime.is_exhausted)

    def test_emits_normalized_usage_from_the_scripted_result(self) -> None:
        usage = AgentRuntimeUsage(
            input_tokens=9,
            cached_input_tokens=2,
            output_tokens=4,
        )
        expected = AgentExecutionResult.succeeded(
            {"answer": "private-output-sentinel"},
            usage=usage,
        )
        runtime = ScriptedAgentRuntime(
            scenarios=(ScriptedAgentScenario(result=expected),)
        )
        context, sink = create_context()

        result = asyncio.run(runtime.execute(create_request(), context))

        self.assertIs(result, expected)
        self.assertIs(result.usage, usage)
        self.assertEqual(
            [event.kind for event in sink.events],
            [
                EventKind.PROVIDER_CALL_STARTED,
                EventKind.USAGE_REPORTED,
                EventKind.PROVIDER_CALL_COMPLETED,
            ],
        )
        self.assertEqual(
            sink.events[1].attributes,
            {
                "agent_id": "researcher",
                "agent_version": "1",
                "call_index": 0,
                "input_tokens": 9,
                "cached_input_tokens": 2,
                "output_tokens": 4,
            },
        )
        self.assertNotIn("private-output-sentinel", repr(sink.events))

    def test_disallowed_tool_is_rejected_before_tool_execution(self) -> None:
        runtime = ScriptedAgentRuntime(
            scenarios=(
                ScriptedAgentScenario(
                    actions=(
                        ScriptedAgentProgress(message="Planning edit"),
                        ScriptedToolRequest(tool_name="write"),
                    ),
                    result=AgentExecutionResult.succeeded(
                        {"answer": "unreachable"}
                    ),
                ),
            )
        )
        context, sink = create_context()

        result = asyncio.run(
            runtime.execute(
                create_request(allowed_tools=("search",)),
                context,
            )
        )

        self.assertEqual(result.result.status, AgentStatus.FAILED)
        self.assertIsNotNone(result.result.failure)
        if result.result.failure is None:
            raise AssertionError("failed result has no failure")
        self.assertEqual(result.result.failure.kind, FailureKind.INVALID_INPUT)
        self.assertEqual(result.result.failure.code, "agent.tool_not_allowed")
        self.assertEqual(result.result.failure.metadata["tool_name"], "write")
        self.assertEqual(
            [event.kind for event in sink.events],
            [
                EventKind.PROVIDER_CALL_STARTED,
                EventKind.PROVIDER_CALL_COMPLETED,
            ],
        )
        self.assertEqual(
            sink.events[-1].attributes["failure_code"],
            "agent.tool_not_allowed",
        )

    def test_approval_request_blocks_and_emits_a_terminal_request(self) -> None:
        failure = approval_failure()
        runtime = ScriptedAgentRuntime(
            scenarios=(
                ScriptedAgentScenario(
                    actions=(
                        ScriptedAgentProgress(message="Prepared change"),
                        ScriptedApprovalRequest(
                            approval_id="approval-1",
                            summary="Apply the prepared change",
                        ),
                    ),
                    result=AgentExecutionResult.from_failure(failure),
                ),
            )
        )
        context, sink = create_context()

        result = asyncio.run(runtime.execute(create_request(), context))

        self.assertEqual(result.result.status, AgentStatus.BLOCKED)
        self.assertIs(result.result.failure, failure)
        self.assertEqual(
            [event.kind for event in sink.events],
            [
                EventKind.PROVIDER_CALL_STARTED,
                EventKind.PROGRESS_REPORTED,
                EventKind.APPROVAL_REQUESTED,
                EventKind.PROVIDER_CALL_COMPLETED,
            ],
        )
        self.assertEqual(
            sink.events[2].attributes["approval_id"],
            "approval-1",
        )
        self.assertEqual(sink.events[-1].attributes["status"], "blocked")

    def test_returns_failure_and_cancellation_scenarios_in_order(self) -> None:
        provider_failure = Failure(
            kind=FailureKind.TRANSIENT_PROVIDER,
            message="provider temporarily unavailable",
            code="provider.busy",
        )
        cancellation = Failure(
            kind=FailureKind.CANCELLED,
            message="scripted caller cancellation",
        )
        runtime = ScriptedAgentRuntime(
            scenarios=(
                ScriptedAgentScenario(
                    result=AgentExecutionResult.from_failure(provider_failure)
                ),
                ScriptedAgentScenario(
                    result=AgentExecutionResult.from_failure(cancellation)
                ),
            )
        )
        context, _ = create_context()

        first = asyncio.run(runtime.execute(create_request(), context))
        snapshot = runtime.calls
        second = asyncio.run(runtime.execute(create_request(), context))

        self.assertIs(first.result.failure, provider_failure)
        self.assertEqual(second.result.status, AgentStatus.CANCELLED)
        self.assertIs(second.result.failure, cancellation)
        self.assertEqual(tuple(call.index for call in snapshot), (0,))
        self.assertEqual(tuple(call.index for call in runtime.calls), (0, 1))
        self.assertTrue(runtime.is_exhausted)

    def test_cancellation_and_deadline_do_not_consume_the_script(self) -> None:
        scenario = ScriptedAgentScenario(
            result=AgentExecutionResult.succeeded({"answer": "complete"})
        )
        source = CancellationSource()
        source.cancel("caller stopped")
        cancelled_runtime = ScriptedAgentRuntime(scenarios=(scenario,))
        cancelled_context, cancelled_sink = create_context(source=source)

        with self.assertRaises(RunCancelled):
            asyncio.run(
                cancelled_runtime.execute(
                    create_request(),
                    cancelled_context,
                )
            )
        self.assertEqual(cancelled_runtime.calls, ())
        self.assertEqual(cancelled_sink.events, ())
        self.assertFalse(cancelled_runtime.is_exhausted)

        clock = FixedClock()
        deadline_runtime = ScriptedAgentRuntime(scenarios=(scenario,))
        deadline_context, deadline_sink = create_context(
            clock=clock,
            deadline=Deadline.after(0, clock),
        )
        with self.assertRaises(DeadlineExceeded):
            asyncio.run(
                deadline_runtime.execute(
                    create_request(),
                    deadline_context,
                )
            )
        self.assertEqual(deadline_runtime.calls, ())
        self.assertEqual(deadline_sink.events, ())
        self.assertFalse(deadline_runtime.is_exhausted)

    def test_exhaustion_returns_a_sanitized_failure_and_records_call(self) -> None:
        runtime = ScriptedAgentRuntime(
            scenarios=(
                ScriptedAgentScenario(
                    result=AgentExecutionResult.succeeded(
                        {"answer": "complete"}
                    )
                ),
            )
        )
        context, sink = create_context()
        asyncio.run(runtime.execute(create_request(), context))

        exhausted = asyncio.run(runtime.execute(create_request(), context))

        self.assertEqual(exhausted.result.status, AgentStatus.FAILED)
        self.assertIsNotNone(exhausted.result.failure)
        if exhausted.result.failure is None:
            raise AssertionError("failed result has no failure")
        self.assertEqual(
            exhausted.result.failure.code,
            "scripted_agent_runtime.exhausted",
        )
        self.assertEqual(
            exhausted.result.failure.metadata,
            {"agent_id": "researcher", "agent_version": "1"},
        )
        self.assertEqual(tuple(call.index for call in runtime.calls), (0, 1))
        self.assertEqual(
            [event.kind for event in sink.events[-2:]],
            [
                EventKind.PROVIDER_CALL_STARTED,
                EventKind.PROVIDER_CALL_COMPLETED,
            ],
        )

    def test_repeat_last_supports_unbounded_deterministic_runs(self) -> None:
        result = AgentExecutionResult.succeeded({"answer": "repeat"})
        runtime = ScriptedAgentRuntime(
            scenarios=(ScriptedAgentScenario(result=result),),
            repeat_last=True,
        )
        context, _ = create_context()

        results = tuple(
            asyncio.run(runtime.execute(create_request(), context))
            for _ in range(3)
        )

        self.assertEqual(results, (result, result, result))
        self.assertFalse(runtime.is_exhausted)

    def test_validates_scenarios_and_approval_invariants(self) -> None:
        success = AgentExecutionResult.succeeded(None)
        approval_result = AgentExecutionResult.from_failure(approval_failure())
        approval = ScriptedApprovalRequest(
            approval_id="approval-1",
            summary="Apply change",
        )

        with self.assertRaises(TypeError):
            ScriptedAgentScenario(
                result="invalid",  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            ScriptedAgentScenario(
                result=success,
                actions=("invalid",),  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            ScriptedAgentScenario(result=success, actions=(approval,))
        with self.assertRaises(ValueError):
            ScriptedAgentScenario(result=approval_result)
        with self.assertRaises(ValueError):
            ScriptedAgentScenario(
                result=approval_result,
                actions=(approval, ScriptedAgentProgress(message="late")),
            )
        with self.assertRaises(ValueError):
            ScriptedAgentScenario(
                result=approval_result,
                actions=(approval, approval),
            )
        with self.assertRaises(ValueError):
            ScriptedAgentRuntime(scenarios=())
        with self.assertRaises(TypeError):
            ScriptedAgentRuntime(
                scenarios=("invalid",),  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            ScriptedAgentRuntime(
                scenarios=(ScriptedAgentScenario(result=success),),
                repeat_last=1,  # type: ignore[arg-type]
            )

    def test_requires_runtime_request_and_context_values(self) -> None:
        runtime = ScriptedAgentRuntime(
            scenarios=(
                ScriptedAgentScenario(
                    result=AgentExecutionResult.succeeded(None)
                ),
            )
        )
        context, _ = create_context()

        with self.assertRaises(TypeError):
            asyncio.run(
                runtime.execute(
                    "invalid",  # type: ignore[arg-type]
                    context,
                )
            )
        with self.assertRaises(TypeError):
            asyncio.run(
                runtime.execute(
                    create_request(),
                    "invalid",  # type: ignore[arg-type]
                )
            )
        self.assertEqual(runtime.calls, ())


if __name__ == "__main__":
    unittest.main()
