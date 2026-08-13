from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agentrig.core import (
    AgentRigError,
    CancellationSource,
    Deadline,
    DeadlineExceeded,
    EventId,
    EventKind,
    ExecutionOutcome,
    ExecutionStatus,
    Failure,
    FailureKind,
    Grade,
    GradeClassification,
    GradeStatus,
    GraderDescriptor,
    GraderUsage,
    GradingContext,
    InMemoryEventSink,
    NoOpRedactionPolicy,
    RunCancelled,
    RunContext,
    RunId,
)
from agentrig.evals import (
    EvalCase,
    EvalCost,
    EvalDataset,
    EvalRunner,
    EvalSubject,
    EvalTargetDescriptor,
    EvalTargetKind,
)


@dataclass
class IncrementingClock:
    monotonic_time: float = 100.0
    increment: float = 0.5

    def now(self) -> datetime:
        return datetime(2026, 8, 14, 2, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        current = self.monotonic_time
        self.monotonic_time += self.increment
        return current


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


def create_context(
    *,
    source: CancellationSource | None = None,
    clock: IncrementingClock | None = None,
    deadline: Deadline | None = None,
    sink: InMemoryEventSink | None = None,
) -> RunContext:
    owned_source = source if source is not None else CancellationSource()
    return RunContext.create_root(
        clock=clock if clock is not None else IncrementingClock(),
        id_generator=SequentialRunIdGenerator(),
        cancellation=owned_source.token,
        event_sink=sink,
        event_id_generator=SequentialEventIdGenerator(),
        deadline=deadline,
        correlation={"suite_id": "suite-1"},
    )


TARGET_DESCRIPTOR = EvalTargetDescriptor(
    target_id="uppercase",
    version="2",
    kind=EvalTargetKind.AGENT,
)
GRADER_DESCRIPTOR = GraderDescriptor(
    grader_id="text.nonempty",
    version="3",
    agentic=True,
)


def create_dataset(*inputs: str) -> EvalDataset[str]:
    return EvalDataset(
        dataset_id="text-quality",
        version="2026-08-14",
        cases=tuple(
            EvalCase(
                case_id=f"case-{index}",
                version="1",
                input=input,
                expected_constraints=("Output must not be empty.",),
            )
            for index, input in enumerate(inputs, start=1)
        ),
    )


@dataclass
class RecordingTarget:
    descriptor: EvalTargetDescriptor = TARGET_DESCRIPTOR
    explicit_failures: dict[str, Failure] = field(default_factory=dict)
    raw_failures: frozenset[str] = frozenset()
    calls: list[tuple[str, RunContext]] = field(default_factory=list)

    async def run(
        self,
        input: str,
        context: RunContext,
    ) -> ExecutionOutcome[str]:
        self.calls.append((input, context))
        if input in self.raw_failures:
            raise RuntimeError("password=private")
        failure = self.explicit_failures.get(input)
        if failure is not None:
            return ExecutionOutcome.from_failure(failure)
        return ExecutionOutcome.succeeded(input.upper())


@dataclass
class RecordingGrader:
    descriptor: GraderDescriptor = GRADER_DESCRIPTOR
    status: GradeStatus = GradeStatus.PASS
    calls: list[
        tuple[EvalSubject[str, str], GradingContext]
    ] = field(default_factory=list)

    async def grade(
        self,
        subject: EvalSubject[str, str],
        context: GradingContext,
    ) -> Grade:
        self.calls.append((subject, context))
        return Grade(
            grader=self.descriptor,
            metric="nonempty",
            status=self.status,
            classification=GradeClassification.HARD,
            explanation="password=private",
            usage=GraderUsage(
                latency_seconds=0.25,
                cost=0.10,
                currency="USD",
            ),
        )


class EvalRunnerTest(unittest.TestCase):
    def test_runs_cases_and_graders_in_order_with_isolated_contexts(self) -> None:
        target = RecordingTarget()
        grader = RecordingGrader()
        runner = EvalRunner[str, str](target=target, graders=(grader,))
        sink = InMemoryEventSink(redaction_policy=NoOpRedactionPolicy())
        context = create_context(sink=sink)

        result = asyncio.run(runner.run(create_dataset("one", "two"), context))

        self.assertEqual(result.dataset_id, "text-quality")
        self.assertEqual(result.dataset_version, "2026-08-14")
        self.assertIs(result.target, TARGET_DESCRIPTOR)
        self.assertEqual(
            tuple(case.outcome.unwrap() for case in result.cases),
            ("ONE", "TWO"),
        )
        self.assertEqual(
            tuple(case.run_id for case in result.cases),
            (RunId("run-2"), RunId("run-4")),
        )
        self.assertEqual(
            tuple(call[1].parent_run_id for call in target.calls),
            (RunId("run-1"), RunId("run-1")),
        )
        self.assertEqual(
            tuple(call[1].correlation["eval_case_id"] for call in target.calls),
            ("case-1", "case-2"),
        )
        self.assertEqual(
            tuple(call[0].output for call in grader.calls),
            ("ONE", "TWO"),
        )
        self.assertEqual(
            tuple(call[0].case.expected_constraints for call in grader.calls),
            (
                ("Output must not be empty.",),
                ("Output must not be empty.",),
            ),
        )
        self.assertEqual(
            tuple(call[1].run_context.parent_run_id for call in grader.calls),
            (RunId("run-2"), RunId("run-4")),
        )
        self.assertEqual(result.summary.case_count, 2)
        self.assertEqual(result.summary.succeeded_cases, 2)
        self.assertEqual(result.summary.passing_grades, 2)
        self.assertEqual(result.summary.grader_failure_count, 0)
        self.assertEqual(result.summary.duration_seconds, 1.0)
        self.assertEqual(result.summary.grader_latency_seconds, 0.5)
        self.assertEqual(
            result.summary.grader_costs,
            (EvalCost(currency="USD", amount=0.20),),
        )
        self.assertEqual(
            tuple(event.kind for event in sink.events),
            (
                EventKind.RUN_STARTED,
                EventKind.GRADE_PRODUCED,
                EventKind.RUN_COMPLETED,
                EventKind.RUN_STARTED,
                EventKind.GRADE_PRODUCED,
                EventKind.RUN_COMPLETED,
            ),
        )
        self.assertEqual(
            tuple(event.run_id for event in sink.events),
            (
                RunId("run-2"),
                RunId("run-3"),
                RunId("run-2"),
                RunId("run-4"),
                RunId("run-5"),
                RunId("run-4"),
            ),
        )
        self.assertEqual(sink.events[-1].attributes["grade_count"], 1)
        self.assertNotIn("output", repr(sink.events))
        self.assertNotIn("explanation", repr(sink.events))
        self.assertNotIn("private", repr(result))
        self.assertNotIn("ONE", repr(result.cases[0]))
        self.assertNotIn("one", repr(grader.calls[0][0]))

    def test_target_failures_skip_grading_and_do_not_stop_later_cases(self) -> None:
        provider_failure = Failure(
            kind=FailureKind.TRANSIENT_PROVIDER,
            message="provider temporarily unavailable",
            code="provider.busy",
        )
        target = RecordingTarget(
            explicit_failures={"provider": provider_failure},
            raw_failures=frozenset({"broken"}),
        )
        grader = RecordingGrader()
        runner = EvalRunner[str, str](target=target, graders=(grader,))
        sink = InMemoryEventSink(redaction_policy=NoOpRedactionPolicy())

        result = asyncio.run(
            runner.run(
                create_dataset("provider", "broken", "complete"),
                create_context(sink=sink),
            )
        )

        self.assertEqual(
            tuple(case.outcome.status for case in result.cases),
            (
                ExecutionStatus.FAILED,
                ExecutionStatus.FAILED,
                ExecutionStatus.SUCCEEDED,
            ),
        )
        self.assertIs(result.cases[0].outcome.failure, provider_failure)
        raw_failure = result.cases[1].outcome.failure
        self.assertIsNotNone(raw_failure)
        if raw_failure is None:
            raise AssertionError("failed target outcome has no failure")
        self.assertEqual(raw_failure.kind, FailureKind.UNEXPECTED)
        self.assertNotIn("private", raw_failure.message)
        self.assertEqual(len(grader.calls), 1)
        self.assertEqual(grader.calls[0][0].output, "COMPLETE")
        self.assertEqual(result.summary.failed_cases, 2)
        self.assertEqual(result.summary.succeeded_cases, 1)
        self.assertEqual(result.summary.passing_grades, 1)
        self.assertEqual(
            tuple(
                event.kind
                for event in sink.events
                if event.kind is not EventKind.RUN_STARTED
                and event.kind is not EventKind.GRADE_PRODUCED
            ),
            (
                EventKind.RUN_FAILED,
                EventKind.RUN_FAILED,
                EventKind.RUN_COMPLETED,
            ),
        )
        self.assertEqual(sink.events[1].attributes["failure_code"], "provider.busy")
        self.assertNotIn("private", repr(sink.events))

    def test_grader_failures_are_recorded_and_other_graders_continue(self) -> None:
        broken_descriptor = GraderDescriptor(
            grader_id="text.broken",
            version="1",
        )

        @dataclass(frozen=True)
        class BrokenGrader:
            descriptor: GraderDescriptor

            async def grade(
                self,
                subject: EvalSubject[str, str],
                context: GradingContext,
            ) -> Grade:
                del subject, context
                raise RuntimeError("api_key=private")

        passing = RecordingGrader()
        result = asyncio.run(
            EvalRunner[str, str](
                target=RecordingTarget(),
                graders=(BrokenGrader(broken_descriptor), passing),
            ).run(create_dataset("draft"), create_context())
        )

        case = result.cases[0]
        self.assertEqual(case.outcome.status, ExecutionStatus.SUCCEEDED)
        self.assertEqual(len(case.grades), 1)
        self.assertEqual(len(case.grader_failures), 1)
        self.assertEqual(case.grader_failures[0].grader, broken_descriptor)
        failure = case.grader_failures[0].failure
        self.assertEqual(failure.kind, FailureKind.GRADER_FAILED)
        self.assertEqual(failure.code, "grader.execution_failed")
        self.assertNotIn("private", failure.message)
        self.assertEqual(len(passing.calls), 1)
        self.assertEqual(result.summary.grader_failure_count, 1)

    def test_invalid_target_and_grader_results_are_safely_classified(self) -> None:
        @dataclass(frozen=True)
        class InvalidTarget:
            descriptor: EvalTargetDescriptor = TARGET_DESCRIPTOR

            async def run(self, input: str, context: RunContext) -> object:
                del input, context
                return "invalid"

        skipped_grader = RecordingGrader()
        invalid_target_result = asyncio.run(
            EvalRunner[str, str](
                target=InvalidTarget(),  # type: ignore[arg-type]
                graders=(skipped_grader,),
            ).run(create_dataset("draft"), create_context())
        )
        target_failure = invalid_target_result.cases[0].outcome.failure
        self.assertIsNotNone(target_failure)
        if target_failure is None:
            raise AssertionError("invalid target result has no failure")
        self.assertEqual(target_failure.code, "eval_target.invalid_result")
        self.assertEqual(skipped_grader.calls, [])

        other_descriptor = GraderDescriptor(
            grader_id="other",
            version="1",
        )

        @dataclass(frozen=True)
        class MismatchedGrader:
            descriptor: GraderDescriptor = GRADER_DESCRIPTOR

            async def grade(
                self,
                subject: EvalSubject[str, str],
                context: GradingContext,
            ) -> Grade:
                del subject, context
                return Grade(
                    grader=other_descriptor,
                    metric="invalid",
                    status=GradeStatus.PASS,
                    classification=GradeClassification.HARD,
                    explanation="Descriptor does not match.",
                )

        invalid_grade_result = asyncio.run(
            EvalRunner[str, str](
                target=RecordingTarget(),
                graders=(MismatchedGrader(),),
            ).run(create_dataset("draft"), create_context())
        )
        grader_failure = invalid_grade_result.cases[0].grader_failures[0].failure
        self.assertEqual(grader_failure.code, "grader.invalid_result")

    def test_cancellation_and_deadline_abort_the_dataset(self) -> None:
        source = CancellationSource()

        @dataclass
        class CancellingTarget(RecordingTarget):
            async def run(
                self,
                input: str,
                context: RunContext,
            ) -> ExecutionOutcome[str]:
                result = await super().run(input, context)
                source.cancel("stopped during eval")
                return result

        target = CancellingTarget()
        grader = RecordingGrader()
        sink = InMemoryEventSink(redaction_policy=NoOpRedactionPolicy())
        with self.assertRaises(RunCancelled):
            asyncio.run(
                EvalRunner[str, str](target=target, graders=(grader,)).run(
                    create_dataset("first", "second"),
                    create_context(source=source, sink=sink),
                )
            )
        self.assertEqual(tuple(call[0] for call in target.calls), ("first",))
        self.assertEqual(grader.calls, [])
        self.assertEqual(
            tuple(event.kind for event in sink.events),
            (EventKind.RUN_STARTED, EventKind.RUN_CANCELLED),
        )

        clock = IncrementingClock(increment=0.0)
        expired = Deadline(
            expires_at=clock.now(),
            monotonic_deadline=clock.monotonic_time,
        )
        untouched_target = RecordingTarget()
        with self.assertRaises(DeadlineExceeded):
            asyncio.run(
                EvalRunner[str, str](
                    target=untouched_target,
                    graders=(RecordingGrader(),),
                ).run(
                    create_dataset("unreachable"),
                    create_context(clock=clock, deadline=expired),
                )
            )
        self.assertEqual(untouched_target.calls, [])

    def test_validates_runner_configuration_and_runtime_arguments(self) -> None:
        target = RecordingTarget()
        grader = RecordingGrader()
        with self.assertRaises(TypeError):
            EvalRunner(
                target="invalid",  # type: ignore[arg-type]
                graders=(grader,),
            )
        with self.assertRaises(ValueError):
            EvalRunner[str, str](target=target, graders=())
        with self.assertRaises(TypeError):
            EvalRunner(
                target=target,
                graders=("invalid",),  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            EvalRunner[str, str](target=target, graders=(grader, grader))

        runner = EvalRunner[str, str](target=target, graders=(grader,))
        with self.assertRaises(TypeError):
            asyncio.run(
                runner.run(
                    "invalid",  # type: ignore[arg-type]
                    create_context(),
                )
            )
        with self.assertRaises(TypeError):
            asyncio.run(
                runner.run(
                    create_dataset("draft"),
                    "invalid",  # type: ignore[arg-type]
                )
            )

    def test_non_grader_agent_rig_errors_are_normalized_as_grader_failures(
        self,
    ) -> None:
        @dataclass(frozen=True)
        class MisclassifiedGrader:
            descriptor: GraderDescriptor

            async def grade(
                self,
                subject: EvalSubject[str, str],
                context: GradingContext,
            ) -> Grade:
                del subject, context
                raise AgentRigError(
                    Failure(
                        kind=FailureKind.INVALID_INPUT,
                        message="grader leaked a subject failure",
                    )
                )

        descriptor = GraderDescriptor(grader_id="misclassified", version="1")
        result = asyncio.run(
            EvalRunner[str, str](
                target=RecordingTarget(),
                graders=(MisclassifiedGrader(descriptor),),
            ).run(create_dataset("draft"), create_context())
        )

        failure = result.cases[0].grader_failures[0].failure
        self.assertEqual(failure.kind, FailureKind.GRADER_FAILED)
        self.assertEqual(failure.code, "grader.execution_failed")


if __name__ == "__main__":
    unittest.main()
