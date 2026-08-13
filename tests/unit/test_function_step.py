from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass
from datetime import UTC, datetime

from agentrig.core import (
    CancellationSource,
    Deadline,
    DeadlineExceeded,
    RunCancelled,
    RunContext,
    RunId,
)
from agentrig.workflow import (
    EffectProfile,
    FunctionStep,
    Step,
    StepDescriptor,
)


@dataclass(frozen=True)
class FixedClock:
    monotonic_time: float = 100.0

    def now(self) -> datetime:
        return datetime(2026, 8, 13, 8, 45, tzinfo=UTC)

    def monotonic(self) -> float:
        return self.monotonic_time


@dataclass
class SequentialRunIdGenerator:
    next_value: int = 1

    def generate(self) -> RunId:
        run_id = RunId(f"run-{self.next_value}")
        self.next_value += 1
        return run_id


def create_descriptor() -> StepDescriptor:
    return StepDescriptor(
        step_id="text.length",
        version="1",
        effect_profile=EffectProfile.READ_ONLY,
    )


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


async def invoke_typed_step(
    step: Step[str, int],
    input: str,
    context: RunContext,
) -> int:
    return await step.run(input, context)


class FunctionStepTest(unittest.TestCase):
    def test_adapts_an_async_typed_callable(self) -> None:
        calls: list[tuple[str, RunId]] = []

        async def text_length(input: str, context: RunContext) -> int:
            calls.append((input, context.run_id))
            return len(input)

        step = FunctionStep(descriptor=create_descriptor(), function=text_length)
        context = create_context()

        result = asyncio.run(invoke_typed_step(step, "draft", context))

        self.assertEqual(result, 5)
        self.assertEqual(calls, [("draft", context.run_id)])
        self.assertEqual(step.descriptor, create_descriptor())

    def test_adapts_an_explicitly_approved_sync_callable_inline(self) -> None:
        calls: list[tuple[str, RunId]] = []

        def text_length(input: str, context: RunContext) -> int:
            calls.append((input, context.run_id))
            return len(input)

        step = FunctionStep.from_sync(
            descriptor=create_descriptor(),
            function=text_length,
            approved=True,
        )
        context = create_context()

        result = asyncio.run(invoke_typed_step(step, "draft", context))

        self.assertEqual(result, 5)
        self.assertEqual(calls, [("draft", context.run_id)])

    def test_sync_adapter_requires_explicit_approval(self) -> None:
        def text_length(input: str, context: RunContext) -> int:
            del context
            return len(input)

        with self.assertRaises(ValueError):
            FunctionStep.from_sync(
                descriptor=create_descriptor(),
                function=text_length,
                approved=False,  # type: ignore[arg-type]
            )

    def test_rejects_mismatched_callable_kinds_and_invalid_values(self) -> None:
        async def async_length(input: str, context: RunContext) -> int:
            del context
            return len(input)

        def sync_length(input: str, context: RunContext) -> int:
            del context
            return len(input)

        with self.assertRaises(TypeError):
            FunctionStep(  # type: ignore[arg-type]
                descriptor=create_descriptor(),
                function=sync_length,
            )
        with self.assertRaises(TypeError):
            FunctionStep.from_sync(  # type: ignore[arg-type]
                descriptor=create_descriptor(),
                function=async_length,
                approved=True,
            )
        with self.assertRaises(TypeError):
            FunctionStep(  # type: ignore[arg-type]
                descriptor="not-a-descriptor",
                function=async_length,
            )
        with self.assertRaises(TypeError):
            FunctionStep(  # type: ignore[arg-type]
                descriptor=create_descriptor(),
                function="not-callable",
            )

    def test_constraints_are_checked_before_invoking_the_callable(self) -> None:
        calls: list[str] = []

        def record(input: str, context: RunContext) -> int:
            del context
            calls.append(input)
            return len(input)

        step = FunctionStep.from_sync(
            descriptor=create_descriptor(),
            function=record,
            approved=True,
        )
        source = CancellationSource()
        source.cancel("caller stopped")

        with self.assertRaises(RunCancelled):
            asyncio.run(step.run("cancelled", create_context(source)))

        expired = Deadline(
            expires_at=datetime(2026, 8, 13, 8, 45, tzinfo=UTC),
            monotonic_deadline=100.0,
        )
        with self.assertRaises(DeadlineExceeded):
            asyncio.run(step.run("expired", create_context(deadline=expired)))

        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
