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
    RetryPolicy,
    Sequence,
    StepDescriptor,
    execute_step_with_retry,
)


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 13, 19, 0, tzinfo=UTC)

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


def create_context(sink: InMemoryEventSink) -> RunContext:
    return RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialIdGenerator("run"),  # type: ignore[arg-type]
        cancellation=CancellationSource().token,
        event_sink=sink,
        event_id_generator=SequentialIdGenerator("event"),  # type: ignore[arg-type]
    )


def descriptor(
    step_id: str,
    *,
    effect_profile: EffectProfile = EffectProfile.READ_ONLY,
) -> StepDescriptor:
    return StepDescriptor(
        step_id=step_id,
        version="1",
        effect_profile=effect_profile,
    )


def transient_failure() -> AgentRigError:
    return AgentRigError(
        Failure(
            kind=FailureKind.TRANSIENT_PROVIDER,
            message="provider temporarily unavailable",
            code="provider.overloaded",
        )
    )


class RetryPolicyTest(unittest.TestCase):
    def test_repeatable_transient_failure_retries_until_success(self) -> None:
        sink = InMemoryEventSink(redaction_policy=NoOpRedactionPolicy())
        calls: list[str] = []

        async def eventually_succeeds(
            input: str,
            context: RunContext,
        ) -> str:
            del context
            calls.append(input)
            if len(calls) < 3:
                raise transient_failure()
            return input.upper()

        step = FunctionStep(
            descriptor=descriptor(
                "provider.call",
                effect_profile=EffectProfile.IDEMPOTENT,
            ),
            function=eventually_succeeds,
        )

        outcome = asyncio.run(
            execute_step_with_retry(
                step,
                "draft",
                create_context(sink).derive_child(),
                retry_policy=RetryPolicy(max_attempts=3),
            )
        )

        self.assertEqual(outcome.unwrap(), "DRAFT")
        self.assertEqual(calls, ["draft", "draft", "draft"])
        self.assertEqual(
            [event.kind for event in sink.events],
            [
                EventKind.STEP_STARTED,
                EventKind.STEP_COMPLETED,
                EventKind.RETRY_SCHEDULED,
                EventKind.STEP_STARTED,
                EventKind.STEP_COMPLETED,
                EventKind.RETRY_SCHEDULED,
                EventKind.STEP_STARTED,
                EventKind.STEP_COMPLETED,
            ],
        )
        self.assertEqual(
            [
                event.attributes["attempt"]
                for event in sink.events
                if event.kind is EventKind.STEP_STARTED
            ],
            [1, 2, 3],
        )
        retry_events = tuple(
            event
            for event in sink.events
            if event.kind is EventKind.RETRY_SCHEDULED
        )
        self.assertEqual(
            [event.attributes["next_attempt"] for event in retry_events],
            [2, 3],
        )
        self.assertEqual(retry_events[0].attributes["max_attempts"], 3)
        self.assertEqual(
            retry_events[0].attributes["failure_kind"],
            "transient_provider",
        )
        self.assertEqual(
            retry_events[0].attributes["failure_code"],
            "provider.overloaded",
        )
        self.assertEqual(sink.events[-1].attributes["status"], "succeeded")
        self.assertNotIn(
            "temporarily unavailable",
            " ".join(event.to_json() for event in sink.events),
        )

    def test_non_repeatable_step_is_never_automatically_retried(self) -> None:
        sink = InMemoryEventSink(redaction_policy=NoOpRedactionPolicy())
        calls: list[str] = []

        async def fail(input: str, context: RunContext) -> str:
            del context
            calls.append(input)
            raise transient_failure()

        outcome = asyncio.run(
            execute_step_with_retry(
                FunctionStep(
                    descriptor=descriptor(
                        "workspace.mutate",
                        effect_profile=EffectProfile.NON_REPEATABLE,
                    ),
                    function=fail,
                ),
                "draft",
                create_context(sink).derive_child(),
                retry_policy=RetryPolicy(max_attempts=3),
            )
        )

        self.assertEqual(outcome.status, ExecutionStatus.FAILED)
        self.assertEqual(calls, ["draft"])
        self.assertEqual(
            [event.kind for event in sink.events],
            [EventKind.STEP_STARTED, EventKind.STEP_COMPLETED],
        )

    def test_only_classified_transient_failures_are_retried(self) -> None:
        sink = InMemoryEventSink(redaction_policy=NoOpRedactionPolicy())
        calls: list[str] = []

        async def fail(input: str, context: RunContext) -> str:
            del context
            calls.append(input)
            raise AgentRigError(
                Failure(
                    kind=FailureKind.PERMANENT_PROVIDER,
                    message="provider rejected request",
                )
            )

        outcome = asyncio.run(
            execute_step_with_retry(
                FunctionStep(
                    descriptor=descriptor("provider.call"),
                    function=fail,
                ),
                "draft",
                create_context(sink).derive_child(),
                retry_policy=RetryPolicy(max_attempts=3),
            )
        )

        self.assertEqual(outcome.status, ExecutionStatus.FAILED)
        self.assertEqual(calls, ["draft"])
        self.assertNotIn(
            EventKind.RETRY_SCHEDULED,
            [event.kind for event in sink.events],
        )

    def test_retry_limit_returns_the_final_failed_attempt(self) -> None:
        sink = InMemoryEventSink(redaction_policy=NoOpRedactionPolicy())
        calls: list[str] = []

        async def fail(input: str, context: RunContext) -> str:
            del context
            calls.append(input)
            raise transient_failure()

        outcome = asyncio.run(
            execute_step_with_retry(
                FunctionStep(
                    descriptor=descriptor("provider.call"),
                    function=fail,
                ),
                "draft",
                create_context(sink).derive_child(),
                retry_policy=RetryPolicy(max_attempts=2),
            )
        )

        self.assertEqual(outcome.status, ExecutionStatus.FAILED)
        self.assertEqual(calls, ["draft", "draft"])
        completed = tuple(
            event
            for event in sink.events
            if event.kind is EventKind.STEP_COMPLETED
        )
        self.assertEqual(
            [event.attributes["attempt"] for event in completed],
            [1, 2],
        )
        self.assertEqual(completed[-1].attributes["status"], "failed")
        self.assertEqual(
            sum(
                event.kind is EventKind.RETRY_SCHEDULED
                for event in sink.events
            ),
            1,
        )

    def test_sequence_applies_and_preserves_its_retry_policy(self) -> None:
        sink = InMemoryEventSink(redaction_policy=NoOpRedactionPolicy())
        calls: list[str] = []

        async def length(input: str, context: RunContext) -> int:
            del context
            calls.append("length")
            if calls.count("length") == 1:
                raise transient_failure()
            return len(input)

        async def render(input: int, context: RunContext) -> str:
            del context
            calls.append("render")
            return str(input)

        policy = RetryPolicy(max_attempts=2)
        base: Sequence[str, int] = Sequence(
            FunctionStep(
                descriptor=descriptor("length"),
                function=length,
            ),
            retry_policy=policy,
        )
        extended = base.then(
            FunctionStep(
                descriptor=descriptor("render"),
                function=render,
            )
        )

        outcome = asyncio.run(extended.execute("draft", create_context(sink)))

        self.assertIs(base.retry_policy, policy)
        self.assertIs(extended.retry_policy, policy)
        self.assertEqual(outcome.unwrap(), "5")
        self.assertEqual(calls, ["length", "length", "render"])
        self.assertEqual(
            [
                event.attributes["attempt"]
                for event in sink.events
                if event.kind is EventKind.STEP_STARTED
            ],
            [1, 2, 1],
        )

    def test_configuration_requires_positive_limits_and_policy_values(
        self,
    ) -> None:
        for max_attempts in (True, 0, -1, 1.5):
            with self.subTest(max_attempts=max_attempts):
                with self.assertRaises(ValueError):
                    RetryPolicy(max_attempts=max_attempts)  # type: ignore[arg-type]

        async def identity(input: str, context: RunContext) -> str:
            del context
            return input

        step = FunctionStep(
            descriptor=descriptor("identity"),
            function=identity,
        )
        context = create_context(InMemoryEventSink())

        with self.assertRaises(TypeError):
            asyncio.run(
                execute_step_with_retry(  # type: ignore[arg-type]
                    step,
                    "draft",
                    context,
                    retry_policy="not-a-policy",
                )
            )
        with self.assertRaises(TypeError):
            Sequence(  # type: ignore[call-overload]
                step,
                retry_policy="not-a-policy",
            )


if __name__ == "__main__":
    unittest.main()
