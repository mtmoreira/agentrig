from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass
from datetime import UTC, datetime

from agentrig.core import (
    AgentRigError,
    CancellationSource,
    Deadline,
    DeadlineExceeded,
    Failure,
    FailureKind,
    RunCancelled,
    RunContext,
    RunId,
)
from agentrig.workflow import (
    EffectProfile,
    FunctionStep,
    Sequence,
    Step,
    StepDescriptor,
)


@dataclass(frozen=True)
class FixedClock:
    monotonic_time: float = 100.0

    def now(self) -> datetime:
        return datetime(2026, 8, 13, 17, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return self.monotonic_time


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


def descriptor(step_id: str) -> StepDescriptor:
    return StepDescriptor(
        step_id=step_id,
        version="1",
        effect_profile=EffectProfile.READ_ONLY,
    )


def typed_three_step_sequence(
    first: Step[str, int],
    second: Step[int, float],
    third: Step[float, str],
) -> Sequence[str, str]:
    return Sequence(first, second, third)


class SequenceTest(unittest.TestCase):
    def test_three_steps_execute_deterministically_with_child_contexts(
        self,
    ) -> None:
        calls: list[tuple[str, object, RunId, RunId | None]] = []

        async def length(input: str, context: RunContext) -> int:
            calls.append(("length", input, context.run_id, context.parent_run_id))
            return len(input)

        async def halve(input: int, context: RunContext) -> float:
            calls.append(("halve", input, context.run_id, context.parent_run_id))
            return input / 2

        async def render(input: float, context: RunContext) -> str:
            calls.append(("render", input, context.run_id, context.parent_run_id))
            return f"{input:.1f}"

        context = create_context()
        sequence = typed_three_step_sequence(
            FunctionStep(descriptor=descriptor("length"), function=length),
            FunctionStep(descriptor=descriptor("halve"), function=halve),
            FunctionStep(descriptor=descriptor("render"), function=render),
        )

        result = asyncio.run(sequence.run("draft", context))

        self.assertEqual(result, "2.5")
        self.assertEqual(
            [(name, value) for name, value, _, _ in calls],
            [("length", "draft"), ("halve", 5), ("render", 2.5)],
        )
        self.assertEqual(
            [parent_run_id for _, _, _, parent_run_id in calls],
            [context.run_id, context.run_id, context.run_id],
        )
        self.assertEqual(
            [run_id for _, _, run_id, _ in calls],
            [RunId("run-2"), RunId("run-3"), RunId("run-4")],
        )

    def test_failure_prevents_downstream_execution(self) -> None:
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
            FunctionStep(descriptor=descriptor("fail"), function=fail),
            FunctionStep(
                descriptor=descriptor("downstream"),
                function=downstream,
            ),
        )

        with self.assertRaises(AgentRigError):
            asyncio.run(sequence.run("draft", create_context()))

        self.assertEqual(calls, ["fail"])

    def test_then_returns_a_new_typed_sequence(self) -> None:
        async def length(input: str, context: RunContext) -> int:
            del context
            return len(input)

        async def render(input: int, context: RunContext) -> str:
            del context
            return str(input)

        first = FunctionStep(descriptor=descriptor("length"), function=length)
        second = FunctionStep(descriptor=descriptor("render"), function=render)
        base: Sequence[str, int] = Sequence(first)
        extended: Sequence[str, str] = base.then(second)

        self.assertEqual(base.steps, (first,))
        self.assertEqual(extended.steps, (first, second))
        self.assertEqual(
            asyncio.run(extended.run("draft", create_context())),
            "5",
        )

    def test_constraints_are_checked_before_each_step(self) -> None:
        calls: list[str] = []
        source = CancellationSource()

        async def cancel(input: str, context: RunContext) -> str:
            del context
            calls.append("cancel")
            source.cancel("stop between steps")
            return input

        async def downstream(input: str, context: RunContext) -> str:
            del context
            calls.append("downstream")
            return input

        sequence = Sequence(
            FunctionStep(descriptor=descriptor("cancel"), function=cancel),
            FunctionStep(
                descriptor=descriptor("downstream"),
                function=downstream,
            ),
        )

        with self.assertRaises(RunCancelled):
            asyncio.run(sequence.run("draft", create_context(source)))

        self.assertEqual(calls, ["cancel"])

        expired = Deadline(
            expires_at=datetime(2026, 8, 13, 17, 0, tzinfo=UTC),
            monotonic_deadline=100.0,
        )
        with self.assertRaises(DeadlineExceeded):
            asyncio.run(
                Sequence(sequence.steps[1]).run(
                    "draft",
                    create_context(deadline=expired),
                )
            )
        self.assertEqual(calls, ["cancel"])

    def test_rejects_empty_invalid_steps_and_invalid_context(self) -> None:
        with self.assertRaises(ValueError):
            Sequence()  # type: ignore[call-overload]
        with self.assertRaises(TypeError):
            Sequence("not-a-step")  # type: ignore[call-overload]

        class InvalidDescriptorStep:
            descriptor = "not-a-descriptor"

            async def run(self, input: str, context: RunContext) -> str:
                del context
                return input

        with self.assertRaises(TypeError):
            Sequence(InvalidDescriptorStep())  # type: ignore[call-overload]

        async def identity(input: str, context: RunContext) -> str:
            del context
            return input

        sequence = Sequence(
            FunctionStep(descriptor=descriptor("identity"), function=identity)
        )
        with self.assertRaises(TypeError):
            asyncio.run(
                sequence.run("draft", "not-a-context")  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
