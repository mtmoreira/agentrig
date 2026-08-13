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
    Grade,
    GradeClassification,
    GradeStatus,
    Grader,
    GraderDescriptor,
    GradingContext,
    RunCancelled,
    RunContext,
    RunId,
)
from agentrig.testing import ScriptedGrader

DESCRIPTOR = GraderDescriptor(grader_id="scripted.structure", version="1")


@dataclass(frozen=True)
class FixedClock:
    monotonic_time: float = 100.0

    def now(self) -> datetime:
        return datetime(2026, 8, 13, 8, 0, tzinfo=UTC)

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
    *,
    source: CancellationSource | None = None,
    clock: FixedClock | None = None,
    deadline: Deadline | None = None,
) -> GradingContext:
    effective_source = source if source is not None else CancellationSource()
    effective_clock = clock if clock is not None else FixedClock()
    return GradingContext(
        run_context=RunContext.create_root(
            clock=effective_clock,
            id_generator=SequentialRunIdGenerator(),
            cancellation=effective_source.token,
            deadline=deadline,
        )
    )


def create_grade(
    status: GradeStatus = GradeStatus.PASS,
    *,
    descriptor: GraderDescriptor = DESCRIPTOR,
) -> Grade:
    return Grade(
        grader=descriptor,
        metric="required_sections",
        status=status,
        classification=GradeClassification.HARD,
        explanation=f"Scripted {status.value}.",
    )


async def grade_text(
    grader: Grader[str],
    subject: str,
    context: GradingContext,
) -> Grade:
    return await grader.grade(subject, context)


class ScriptedGraderTest(unittest.TestCase):
    def test_returns_outcomes_in_order_and_records_stable_calls(self) -> None:
        passing = create_grade()
        failing = create_grade(GradeStatus.FAILURE)
        grader = ScriptedGrader[str](
            descriptor=DESCRIPTOR,
            outcomes=(passing, failing),
        )
        context = create_context()

        first = asyncio.run(grade_text(grader, "first", context))
        snapshot = grader.calls
        second = asyncio.run(grade_text(grader, "second", context))

        self.assertIsInstance(grader, Grader)
        self.assertIs(first, passing)
        self.assertIs(second, failing)
        self.assertEqual(tuple(call.subject for call in snapshot), ("first",))
        self.assertEqual(
            tuple(call.subject for call in grader.calls),
            ("first", "second"),
        )
        self.assertEqual(tuple(call.index for call in grader.calls), (0, 1))
        self.assertIs(grader.calls[0].context, context)
        self.assertTrue(grader.is_exhausted)

    def test_scripted_failure_is_normalized_and_distinct_from_subject_failure(
        self,
    ) -> None:
        grader_failure = Failure(
            kind=FailureKind.GRADER_FAILED,
            message="scripted evaluator unavailable",
            code="scripted.unavailable",
        )
        grader = ScriptedGrader[str](
            descriptor=DESCRIPTOR,
            outcomes=(grader_failure,),
        )

        with self.assertRaises(AgentRigError) as raised:
            asyncio.run(grade_text(grader, "subject", create_context()))

        self.assertIs(raised.exception.failure, grader_failure)
        self.assertEqual(raised.exception.failure.kind, FailureKind.GRADER_FAILED)
        self.assertTrue(grader.is_exhausted)

    def test_exhaustion_raises_a_sanitized_grader_failure(self) -> None:
        grader = ScriptedGrader[str](
            descriptor=DESCRIPTOR,
            outcomes=(create_grade(),),
        )
        context = create_context()
        asyncio.run(grade_text(grader, "first", context))

        with self.assertRaises(AgentRigError) as raised:
            asyncio.run(grade_text(grader, "second", context))

        self.assertEqual(raised.exception.failure.kind, FailureKind.GRADER_FAILED)
        self.assertEqual(
            raised.exception.failure.code,
            "scripted_grader.exhausted",
        )
        self.assertEqual(
            raised.exception.failure.metadata,
            {
                "grader_id": DESCRIPTOR.grader_id,
                "grader_version": DESCRIPTOR.version,
            },
        )
        self.assertEqual(len(grader.calls), 2)

    def test_repeat_last_supports_unbounded_reproducible_scenarios(self) -> None:
        passing = create_grade()
        grader = ScriptedGrader[str](
            descriptor=DESCRIPTOR,
            outcomes=(passing,),
            repeat_last=True,
        )
        context = create_context()

        results = tuple(
            asyncio.run(grade_text(grader, subject, context))
            for subject in ("one", "two", "three")
        )

        self.assertEqual(results, (passing, passing, passing))
        self.assertFalse(grader.is_exhausted)

    def test_validates_descriptor_outcomes_and_repeat_configuration(self) -> None:
        other_descriptor = GraderDescriptor(grader_id="other", version="1")
        invalid_failure = Failure(
            kind=FailureKind.INVALID_INPUT,
            message="subject was invalid",
        )

        with self.assertRaises(ValueError):
            ScriptedGrader[str](
                descriptor=DESCRIPTOR,
                outcomes=(create_grade(descriptor=other_descriptor),),
            )
        with self.assertRaises(ValueError):
            ScriptedGrader[str](
                descriptor=DESCRIPTOR,
                outcomes=(invalid_failure,),
            )
        with self.assertRaises(ValueError):
            ScriptedGrader[str](descriptor=DESCRIPTOR, outcomes=())
        with self.assertRaises(TypeError):
            ScriptedGrader[str](
                descriptor=DESCRIPTOR,
                outcomes=("not-an-outcome",),  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            ScriptedGrader[str](
                descriptor=DESCRIPTOR,
                outcomes=(create_grade(),),
                repeat_last=1,  # type: ignore[arg-type]
            )

    def test_cancellation_and_deadline_do_not_consume_the_script(self) -> None:
        source = CancellationSource()
        source.cancel("caller stopped grading")
        cancelled_grader = ScriptedGrader[str](
            descriptor=DESCRIPTOR,
            outcomes=(create_grade(),),
        )

        with self.assertRaises(RunCancelled):
            asyncio.run(
                grade_text(
                    cancelled_grader,
                    "subject",
                    create_context(source=source),
                )
            )
        self.assertEqual(cancelled_grader.calls, ())
        self.assertFalse(cancelled_grader.is_exhausted)

        clock = FixedClock()
        expired_grader = ScriptedGrader[str](
            descriptor=DESCRIPTOR,
            outcomes=(create_grade(),),
        )
        with self.assertRaises(DeadlineExceeded):
            asyncio.run(
                grade_text(
                    expired_grader,
                    "subject",
                    create_context(
                        clock=clock,
                        deadline=Deadline.after(0, clock),
                    ),
                )
            )
        self.assertEqual(expired_grader.calls, ())
        self.assertFalse(expired_grader.is_exhausted)


if __name__ == "__main__":
    unittest.main()
