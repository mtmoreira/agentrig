from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass
from datetime import UTC, datetime

from agentrig.core import (
    AgentRigError,
    CancellationSource,
    EventId,
    EventKind,
    ExecutionStatus,
    Failure,
    FailureKind,
    InMemoryEventSink,
    NoOpRedactionPolicy,
    RunContext,
    RunId,
)
from agentrig.workflow import (
    EffectProfile,
    FunctionStep,
    Sequence,
    StepDescriptor,
    execute_step,
)


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 13, 18, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


@dataclass
class SequentialIdGenerator:
    prefix: str
    next_value: int = 1

    def generate(self) -> RunId | EventId:
        value = f"{self.prefix}-{self.next_value}"
        self.next_value += 1
        if self.prefix == "run":
            return RunId(value)
        return EventId(value)


def create_descriptor(step_id: str) -> StepDescriptor:
    return StepDescriptor(
        step_id=step_id,
        version="1",
        effect_profile=EffectProfile.READ_ONLY,
    )


def create_context(
    sink: InMemoryEventSink,
    source: CancellationSource | None = None,
) -> RunContext:
    owned_source = source if source is not None else CancellationSource()
    return RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialIdGenerator("run"),  # type: ignore[arg-type]
        cancellation=owned_source.token,
        event_sink=sink,
        event_id_generator=SequentialIdGenerator("event"),  # type: ignore[arg-type]
        correlation={"request_id": "request-1"},
    )


class StepExecutionTest(unittest.TestCase):
    def test_success_returns_outcome_and_emits_safe_lifecycle(self) -> None:
        sink = InMemoryEventSink(redaction_policy=NoOpRedactionPolicy())

        async def transform(input: str, context: RunContext) -> str:
            del context
            return f"secret-output-for-{input}"

        context = create_context(sink).derive_child()
        step = FunctionStep(
            descriptor=create_descriptor("transform"),
            function=transform,
        )

        outcome = asyncio.run(
            execute_step(step, "secret-input", context, attempt=2)
        )

        self.assertEqual(outcome.unwrap(), "secret-output-for-secret-input")
        self.assertEqual(
            [event.kind for event in sink.events],
            [EventKind.STEP_STARTED, EventKind.STEP_COMPLETED],
        )
        self.assertEqual(
            [event.event_id for event in sink.events],
            [EventId("event-1"), EventId("event-2")],
        )
        self.assertEqual(sink.events[0].run_id, context.run_id)
        self.assertEqual(sink.events[0].parent_run_id, context.parent_run_id)
        self.assertEqual(sink.events[0].correlation["request_id"], "request-1")
        self.assertEqual(sink.events[0].attributes["attempt"], 2)
        self.assertEqual(sink.events[1].attributes["status"], "succeeded")
        serialized = " ".join(event.to_json() for event in sink.events)
        self.assertNotIn("secret-input", serialized)
        self.assertNotIn("secret-output", serialized)

    def test_raw_exception_is_normalized_without_retaining_its_message(
        self,
    ) -> None:
        sink = InMemoryEventSink(redaction_policy=NoOpRedactionPolicy())

        async def fail(input: str, context: RunContext) -> str:
            del input, context
            raise RuntimeError("password=private")

        outcome = asyncio.run(
            execute_step(
                FunctionStep(
                    descriptor=create_descriptor("fail"),
                    function=fail,
                ),
                "input",
                create_context(sink).derive_child(),
            )
        )

        self.assertEqual(outcome.status, ExecutionStatus.FAILED)
        failure = outcome.failure
        self.assertIsNotNone(failure)
        if failure is None:
            raise AssertionError("failed outcome has no failure")
        self.assertEqual(failure.kind, FailureKind.UNEXPECTED)
        self.assertEqual(sink.events[-1].attributes["status"], "failed")
        self.assertEqual(
            sink.events[-1].attributes["failure_kind"],
            "unexpected",
        )
        self.assertNotIn("private", sink.events[-1].to_json())

    def test_normalized_failure_preserves_category_and_safe_code(self) -> None:
        sink = InMemoryEventSink(redaction_policy=NoOpRedactionPolicy())
        failure = Failure(
            kind=FailureKind.WORKFLOW_BLOCKED,
            message="workflow requires external state",
            code="workflow.waiting",
        )

        async def block(input: str, context: RunContext) -> str:
            del input, context
            raise AgentRigError(failure)

        outcome = asyncio.run(
            execute_step(
                FunctionStep(
                    descriptor=create_descriptor("block"),
                    function=block,
                ),
                "input",
                create_context(sink).derive_child(),
            )
        )

        self.assertEqual(outcome.status, ExecutionStatus.BLOCKED)
        self.assertIs(outcome.failure, failure)
        self.assertEqual(
            sink.events[-1].attributes["failure_code"],
            "workflow.waiting",
        )

    def test_cancellation_is_captured_as_a_terminal_outcome(self) -> None:
        sink = InMemoryEventSink(redaction_policy=NoOpRedactionPolicy())
        source = CancellationSource()
        source.cancel("caller stopped")
        calls: list[str] = []

        async def unreachable(input: str, context: RunContext) -> str:
            del input, context
            calls.append("unreachable")
            return "output"

        outcome = asyncio.run(
            execute_step(
                FunctionStep(
                    descriptor=create_descriptor("cancelled"),
                    function=unreachable,
                ),
                "input",
                create_context(sink, source).derive_child(),
            )
        )

        self.assertEqual(outcome.status, ExecutionStatus.CANCELLED)
        self.assertEqual(calls, [])
        self.assertEqual(
            [event.kind for event in sink.events],
            [EventKind.STEP_STARTED, EventKind.STEP_COMPLETED],
        )
        self.assertEqual(sink.events[-1].attributes["status"], "cancelled")
        self.assertEqual(
            sink.events[-1].attributes["failure_kind"],
            "cancelled",
        )

    def test_sequence_execute_returns_failure_and_stops_downstream(self) -> None:
        sink = InMemoryEventSink(redaction_policy=NoOpRedactionPolicy())
        calls: list[str] = []

        async def fail(input: str, context: RunContext) -> int:
            del input, context
            calls.append("fail")
            raise AgentRigError(
                Failure(
                    kind=FailureKind.PERMANENT_PROVIDER,
                    message="provider rejected request",
                )
            )

        async def downstream(input: int, context: RunContext) -> str:
            del input, context
            calls.append("downstream")
            return "unreachable"

        sequence = Sequence(
            FunctionStep(
                descriptor=create_descriptor("fail"),
                function=fail,
            ),
            FunctionStep(
                descriptor=create_descriptor("downstream"),
                function=downstream,
            ),
        )

        outcome = asyncio.run(sequence.execute("input", create_context(sink)))

        self.assertEqual(outcome.status, ExecutionStatus.FAILED)
        self.assertEqual(calls, ["fail"])
        self.assertEqual(
            [event.kind for event in sink.events],
            [EventKind.STEP_STARTED, EventKind.STEP_COMPLETED],
        )

    def test_successful_sequence_emits_a_parented_event_pair_per_step(self) -> None:
        sink = InMemoryEventSink(redaction_policy=NoOpRedactionPolicy())

        async def length(input: str, context: RunContext) -> int:
            del context
            return len(input)

        async def render(input: int, context: RunContext) -> str:
            del context
            return str(input)

        sequence = Sequence(
            FunctionStep(
                descriptor=create_descriptor("length"),
                function=length,
            ),
            FunctionStep(
                descriptor=create_descriptor("render"),
                function=render,
            ),
        )
        context = create_context(sink)

        outcome = asyncio.run(sequence.execute("draft", context))

        self.assertEqual(outcome.unwrap(), "5")
        self.assertEqual(
            [event.kind for event in sink.events],
            [
                EventKind.STEP_STARTED,
                EventKind.STEP_COMPLETED,
                EventKind.STEP_STARTED,
                EventKind.STEP_COMPLETED,
            ],
        )
        self.assertEqual(
            [event.run_id for event in sink.events],
            [RunId("run-2"), RunId("run-2"), RunId("run-3"), RunId("run-3")],
        )
        self.assertEqual(
            [event.parent_run_id for event in sink.events],
            [context.run_id, context.run_id, context.run_id, context.run_id],
        )

    def test_configuration_requires_valid_attempt_step_and_context(self) -> None:
        sink = InMemoryEventSink()

        async def identity(input: str, context: RunContext) -> str:
            del context
            return input

        step = FunctionStep(
            descriptor=create_descriptor("identity"),
            function=identity,
        )
        context = create_context(sink)

        for attempt in (True, 0, -1, 1.5):
            with self.subTest(attempt=attempt):
                with self.assertRaises(ValueError):
                    asyncio.run(
                        execute_step(  # type: ignore[arg-type]
                            step,
                            "input",
                            context,
                            attempt=attempt,
                        )
                    )
        with self.assertRaises(TypeError):
            asyncio.run(
                execute_step(  # type: ignore[arg-type]
                    "not-a-step",
                    "input",
                    context,
                )
            )
        with self.assertRaises(TypeError):
            asyncio.run(
                execute_step(  # type: ignore[arg-type]
                    step,
                    "input",
                    "not-a-context",
                )
            )


if __name__ == "__main__":
    unittest.main()
