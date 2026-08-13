from __future__ import annotations

import asyncio
import math
import unittest
from collections.abc import Sequence as GradeSequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agentrig.core import (
    AgentRigError,
    ArtifactId,
    ArtifactRef,
    CancellationSource,
    EffectProfile,
    EventId,
    EventKind,
    ExecutionStatus,
    Failure,
    FailureKind,
    Grade,
    GradeClassification,
    GradeDecision,
    GradeEvidence,
    GradePolicyDescriptor,
    GradeReference,
    GradeStatus,
    GraderDescriptor,
    GraderUsage,
    InMemoryEventSink,
    NoOpRedactionPolicy,
    RunContext,
    RunId,
)
from agentrig.testing import ScriptedGrader
from agentrig.workflow import (
    GradeStep,
    RepairBudget,
    RepairLoop,
    RepairLoopResult,
    RepairRequest,
    Step,
    StepDescriptor,
    execute_step,
)


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 14, 3, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


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
) -> tuple[RunContext, InMemoryEventSink]:
    owned_source = source if source is not None else CancellationSource()
    sink = InMemoryEventSink(redaction_policy=NoOpRedactionPolicy())
    context = RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialRunIdGenerator(),
        cancellation=owned_source.token,
        event_sink=sink,
        event_id_generator=SequentialEventIdGenerator(),
        correlation={"request_id": "request-1"},
    )
    return context, sink


INTEGRITY = GraderDescriptor(grader_id="story.integrity", version="1")
TARGET = GraderDescriptor(grader_id="story.target", version="1")
AGENTIC = GraderDescriptor(
    grader_id="story.agentic",
    version="1",
    agentic=True,
)


def create_grade(
    descriptor: GraderDescriptor,
    *,
    metric: str,
    status: GradeStatus,
    classification: GradeClassification = GradeClassification.HARD,
    evidence: tuple[GradeEvidence, ...] = (),
    usage: GraderUsage | None = None,
) -> Grade:
    return Grade(
        grader=descriptor,
        metric=metric,
        status=status,
        classification=classification,
        explanation="private grader explanation",
        evidence=evidence,
        usage=usage,
    )


@dataclass(frozen=True)
class AnyFailurePolicy:
    descriptor: GradePolicyDescriptor = field(
        default_factory=lambda: GradePolicyDescriptor(
            policy_id="story.repair",
            version="1",
        )
    )

    def decide(self, grades: GradeSequence[Grade]) -> GradeDecision:
        if any(grade.status is GradeStatus.FAILURE for grade in grades):
            return GradeDecision.REPAIR
        return GradeDecision.CONTINUE


@dataclass(frozen=True)
class TargetOnlyPolicy:
    descriptor: GradePolicyDescriptor = field(
        default_factory=lambda: GradePolicyDescriptor(
            policy_id="story.target-only",
            version="1",
        )
    )

    def decide(self, grades: GradeSequence[Grade]) -> GradeDecision:
        for grade in grades:
            if grade.grader == TARGET and grade.status is GradeStatus.FAILURE:
                return GradeDecision.REPAIR
        return GradeDecision.CONTINUE


@dataclass(frozen=True)
class ConstantPolicy:
    decision: GradeDecision
    descriptor: GradePolicyDescriptor = field(
        default_factory=lambda: GradePolicyDescriptor(
            policy_id="story.constant",
            version="1",
        )
    )

    def decide(self, grades: GradeSequence[Grade]) -> GradeDecision:
        del grades
        return self.decision


@dataclass
class RecordingRepair:
    outputs: tuple[str, ...] = ("repaired",)
    failure: Failure | None = None
    descriptor: StepDescriptor = field(
        default_factory=lambda: StepDescriptor(
            step_id="story.repair",
            version="2",
            effect_profile=EffectProfile.IDEMPOTENT,
        )
    )
    calls: list[tuple[RepairRequest[str], RunContext]] = field(
        default_factory=list
    )

    async def run(
        self,
        request: RepairRequest[str],
        context: RunContext,
    ) -> str:
        index = len(self.calls)
        self.calls.append((request, context))
        if self.failure is not None:
            raise AgentRigError(self.failure)
        return self.outputs[min(index, len(self.outputs) - 1)]


def create_loop(
    *,
    graders: tuple[ScriptedGrader[str], ...],
    policy: object,
    repair: RecordingRepair | None = None,
    max_attempts: int = 3,
    budget: RepairBudget | None = None,
) -> tuple[RepairLoop[str], RecordingRepair]:
    owned_repair = repair if repair is not None else RecordingRepair()
    loop = RepairLoop[str](
        repair_step=owned_repair,
        grade_step=GradeStep(
            graders=graders,
            policy=policy,  # type: ignore[arg-type]
        ),
        max_attempts=max_attempts,
        budget=(
            budget
            if budget is not None
            else RepairBudget(max_grading_cost=0)
        ),
    )
    return loop, owned_repair


class RepairLoopTest(unittest.TestCase):
    def test_accepts_an_initially_passing_subject_without_repair(self) -> None:
        grade = create_grade(
            INTEGRITY,
            metric="integrity",
            status=GradeStatus.PASS,
        )
        grader = ScriptedGrader[str](
            descriptor=INTEGRITY,
            outcomes=(grade,),
        )
        loop, repair = create_loop(
            graders=(grader,),
            policy=AnyFailurePolicy(),
        )
        context, sink = create_context()

        outcome = asyncio.run(execute_step(loop, "draft", context))

        self.assertIsInstance(loop, Step)
        self.assertEqual(
            loop.descriptor,
            StepDescriptor(
                step_id="story.repair.repair",
                version="2",
                effect_profile=EffectProfile.IDEMPOTENT,
            ),
        )
        self.assertEqual(outcome.status, ExecutionStatus.SUCCEEDED)
        result = outcome.unwrap()
        self.assertIsInstance(result, RepairLoopResult)
        self.assertEqual(result.subject, "draft")
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.repairs, 0)
        self.assertEqual(result.grading_cost, 0)
        self.assertEqual(
            result.required_hard_constraints,
            (GradeReference.from_grade(grade),),
        )
        self.assertEqual(repair.calls, [])
        self.assertNotIn(
            EventKind.PROGRESS_REPORTED,
            [event.kind for event in sink.events],
        )

    def test_repairs_only_failures_and_preserves_passing_hard_constraints(
        self,
    ) -> None:
        passing_evidence = GradeEvidence(field_path=("private", "passing"))
        target_evidence = GradeEvidence(field_path=("target",))
        regression_evidence = GradeEvidence(field_path=("integrity",))
        integrity_grades = (
            create_grade(
                INTEGRITY,
                metric="integrity",
                status=GradeStatus.PASS,
                evidence=(passing_evidence,),
            ),
            create_grade(
                INTEGRITY,
                metric="integrity",
                status=GradeStatus.FAILURE,
                evidence=(regression_evidence,),
            ),
            create_grade(
                INTEGRITY,
                metric="integrity",
                status=GradeStatus.PASS,
            ),
        )
        target_grades = (
            create_grade(
                TARGET,
                metric="target",
                status=GradeStatus.FAILURE,
                evidence=(target_evidence,),
            ),
            create_grade(
                TARGET,
                metric="target",
                status=GradeStatus.PASS,
            ),
            create_grade(
                TARGET,
                metric="target",
                status=GradeStatus.PASS,
            ),
        )
        integrity = ScriptedGrader[str](
            descriptor=INTEGRITY,
            outcomes=integrity_grades,
        )
        target = ScriptedGrader[str](
            descriptor=TARGET,
            outcomes=target_grades,
        )
        repair = RecordingRepair(outputs=("repaired-1", "repaired-2"))
        loop, _ = create_loop(
            graders=(integrity, target),
            policy=TargetOnlyPolicy(),
            repair=repair,
        )
        context, sink = create_context()

        result = asyncio.run(loop.run("draft", context))

        self.assertEqual(result.subject, "repaired-2")
        self.assertEqual(result.attempts, 3)
        self.assertEqual(result.repairs, 2)
        self.assertEqual(len(repair.calls), 2)
        first_request = repair.calls[0][0]
        self.assertEqual(first_request.current_subject, "draft")
        self.assertEqual(first_request.repair_attempt, 1)
        self.assertEqual(first_request.failed_constraints, (target_grades[0],))
        self.assertEqual(first_request.evidence, (target_evidence,))
        self.assertNotIn(passing_evidence, first_request.evidence)
        self.assertEqual(
            first_request.required_hard_constraints,
            (GradeReference.from_grade(integrity_grades[0]),),
        )

        second_request = repair.calls[1][0]
        self.assertEqual(second_request.current_subject, "repaired-1")
        self.assertEqual(
            second_request.failed_constraints,
            (integrity_grades[1],),
        )
        self.assertEqual(second_request.evidence, (regression_evidence,))
        self.assertEqual(
            second_request.required_hard_constraints,
            tuple(
                sorted(
                    {
                        GradeReference.from_grade(integrity_grades[0]),
                        GradeReference.from_grade(target_grades[1]),
                    }
                )
            ),
        )
        self.assertEqual(
            [call.subject for call in integrity.calls],
            ["draft", "repaired-1", "repaired-2"],
        )
        progress = tuple(
            event
            for event in sink.events
            if event.kind is EventKind.PROGRESS_REPORTED
        )
        self.assertEqual(len(progress), 2)
        self.assertEqual(progress[0].attributes["failed_constraint_count"], 1)
        self.assertEqual(
            progress[1].attributes["required_hard_constraint_count"],
            2,
        )
        self.assertNotIn("private", repr(progress))

    def test_exhaustion_returns_blocked_without_an_unbounded_loop(self) -> None:
        failure_grade = create_grade(
            TARGET,
            metric="target",
            status=GradeStatus.FAILURE,
            evidence=(GradeEvidence(field_path=("target",)),),
        )
        grader = ScriptedGrader[str](
            descriptor=TARGET,
            outcomes=(failure_grade,),
            repeat_last=True,
        )
        loop, repair = create_loop(
            graders=(grader,),
            policy=AnyFailurePolicy(),
            max_attempts=2,
        )
        context, sink = create_context()

        outcome = asyncio.run(execute_step(loop, "draft", context))

        self.assertEqual(outcome.status, ExecutionStatus.BLOCKED)
        self.assertIsNotNone(outcome.failure)
        if outcome.failure is None:
            raise AssertionError("exhausted outcome has no failure")
        self.assertEqual(outcome.failure.kind, FailureKind.WORKFLOW_BLOCKED)
        self.assertEqual(outcome.failure.code, "repair.attempts_exhausted")
        self.assertEqual(outcome.failure.metadata["attempt"], "2")
        self.assertEqual(len(grader.calls), 2)
        self.assertEqual(len(repair.calls), 1)
        self.assertEqual(
            sum(
                event.kind is EventKind.PROGRESS_REPORTED
                for event in sink.events
            ),
            1,
        )

    def test_repair_receives_the_current_artifact_and_relevant_evidence(self) -> None:
        original = ArtifactRef(
            artifact_id=ArtifactId("artifact-original"),
            kind="story",
            media_type="text/plain",
            producer_run_id=RunId("run-producer-1"),
            workspace_path="outputs/original.txt",
        )
        repaired = ArtifactRef(
            artifact_id=ArtifactId("artifact-repaired"),
            kind="story",
            media_type="text/plain",
            producer_run_id=RunId("run-producer-2"),
            workspace_path="outputs/repaired.txt",
            input_artifact_ids=(original.artifact_id,),
        )
        failed = create_grade(
            TARGET,
            metric="target",
            status=GradeStatus.FAILURE,
            evidence=(GradeEvidence(artifact_id=original.artifact_id),),
        )
        passed = create_grade(
            TARGET,
            metric="target",
            status=GradeStatus.PASS,
            evidence=(GradeEvidence(artifact_id=repaired.artifact_id),),
        )
        grader = ScriptedGrader[ArtifactRef](
            descriptor=TARGET,
            outcomes=(failed, passed),
        )

        @dataclass
        class ArtifactRepair:
            descriptor: StepDescriptor = field(
                default_factory=lambda: StepDescriptor(
                    step_id="story.artifact-repair",
                    version="1",
                    effect_profile=EffectProfile.IDEMPOTENT,
                )
            )
            calls: list[RepairRequest[ArtifactRef]] = field(
                default_factory=list
            )

            async def run(
                self,
                request: RepairRequest[ArtifactRef],
                context: RunContext,
            ) -> ArtifactRef:
                del context
                self.calls.append(request)
                return repaired

        repair = ArtifactRepair()
        loop = RepairLoop[ArtifactRef](
            repair_step=repair,
            grade_step=GradeStep(
                graders=(grader,),
                policy=AnyFailurePolicy(),
            ),
            max_attempts=2,
            budget=RepairBudget(max_grading_cost=0),
        )
        context, _ = create_context()

        result = asyncio.run(loop.run(original, context))

        self.assertIs(result.subject, repaired)
        self.assertEqual(len(repair.calls), 1)
        self.assertIs(repair.calls[0].current_subject, original)
        self.assertEqual(
            repair.calls[0].evidence,
            (GradeEvidence(artifact_id=original.artifact_id),),
        )
        self.assertEqual(
            [call.subject for call in grader.calls],
            [original, repaired],
        )

    def test_cumulative_grading_cost_enforces_the_explicit_budget(self) -> None:
        usage = GraderUsage(latency_seconds=0.1, cost=0.6, currency="USD")
        grade = create_grade(
            AGENTIC,
            metric="quality",
            status=GradeStatus.FAILURE,
            classification=GradeClassification.SOFT,
            evidence=(GradeEvidence(field_path=("quality",)),),
            usage=usage,
        )
        grader = ScriptedGrader[str](
            descriptor=AGENTIC,
            outcomes=(grade,),
            repeat_last=True,
        )
        loop, repair = create_loop(
            graders=(grader,),
            policy=AnyFailurePolicy(),
            budget=RepairBudget(max_grading_cost=1.0),
        )
        context, _ = create_context()

        outcome = asyncio.run(execute_step(loop, "draft", context))

        self.assertEqual(outcome.status, ExecutionStatus.FAILED)
        self.assertIsNotNone(outcome.failure)
        if outcome.failure is None:
            raise AssertionError("budget outcome has no failure")
        self.assertEqual(outcome.failure.kind, FailureKind.BUDGET_EXHAUSTED)
        self.assertEqual(outcome.failure.code, "repair.budget_exhausted")
        self.assertEqual(outcome.failure.metadata["grading_cost"], "1.2")
        self.assertEqual(len(grader.calls), 2)
        self.assertEqual(len(repair.calls), 1)

    def test_policy_block_and_approval_are_distinct_terminal_states(self) -> None:
        passing_grade = create_grade(
            TARGET,
            metric="target",
            status=GradeStatus.PASS,
        )
        for decision, expected_kind, expected_code in (
            (
                GradeDecision.BLOCK,
                FailureKind.WORKFLOW_BLOCKED,
                "repair.policy_blocked",
            ),
            (
                GradeDecision.REQUEST_APPROVAL,
                FailureKind.APPROVAL_REQUIRED,
                "repair.approval_required",
            ),
        ):
            with self.subTest(decision=decision):
                grader = ScriptedGrader[str](
                    descriptor=TARGET,
                    outcomes=(passing_grade,),
                )
                loop, repair = create_loop(
                    graders=(grader,),
                    policy=ConstantPolicy(decision),
                )
                context, _ = create_context()

                outcome = asyncio.run(execute_step(loop, "draft", context))

                self.assertEqual(outcome.status, ExecutionStatus.BLOCKED)
                self.assertIsNotNone(outcome.failure)
                if outcome.failure is None:
                    raise AssertionError("terminal outcome has no failure")
                self.assertEqual(outcome.failure.kind, expected_kind)
                self.assertEqual(outcome.failure.code, expected_code)
                self.assertEqual(repair.calls, [])

    def test_preserves_grader_and_repair_step_failures(self) -> None:
        grader_failure = Failure(
            kind=FailureKind.GRADER_FAILED,
            message="grader unavailable",
            code="grader.unavailable",
        )
        failing_grader = ScriptedGrader[str](
            descriptor=TARGET,
            outcomes=(grader_failure,),
        )
        loop, repair = create_loop(
            graders=(failing_grader,),
            policy=AnyFailurePolicy(),
        )
        context, _ = create_context()

        grader_outcome = asyncio.run(execute_step(loop, "draft", context))

        self.assertIs(grader_outcome.failure, grader_failure)
        self.assertEqual(repair.calls, [])

        repair_failure = Failure(
            kind=FailureKind.PERMANENT_PROVIDER,
            message="repair implementation failed",
            code="repair.failed",
        )
        failure_grade = create_grade(
            TARGET,
            metric="target",
            status=GradeStatus.FAILURE,
            evidence=(GradeEvidence(field_path=("target",)),),
        )
        grader = ScriptedGrader[str](
            descriptor=TARGET,
            outcomes=(failure_grade,),
        )
        repair = RecordingRepair(failure=repair_failure)
        loop, _ = create_loop(
            graders=(grader,),
            policy=AnyFailurePolicy(),
            repair=repair,
        )
        context, _ = create_context()

        repair_outcome = asyncio.run(execute_step(loop, "draft", context))

        self.assertIs(repair_outcome.failure, repair_failure)
        self.assertEqual(len(repair.calls), 1)

    def test_rejects_unscoped_repairs_invalid_limits_and_cancellation(self) -> None:
        failure_grade = create_grade(
            TARGET,
            metric="target",
            status=GradeStatus.FAILURE,
            evidence=(GradeEvidence(field_path=("target",)),),
        )
        grader = ScriptedGrader[str](
            descriptor=TARGET,
            outcomes=(failure_grade,),
        )
        loop, repair = create_loop(
            graders=(grader,),
            policy=ConstantPolicy(GradeDecision.REPAIR),
        )

        with self.assertRaises(ValueError):
            RepairBudget(max_grading_cost=-1)
        with self.assertRaises(ValueError):
            RepairBudget(max_grading_cost=math.inf)
        with self.assertRaises(ValueError):
            RepairLoop(
                repair_step=repair,
                grade_step=loop.grade_step,
                max_attempts=0,
                budget=RepairBudget(max_grading_cost=0),
            )
        with self.assertRaises(ValueError):
            RepairRequest(
                current_subject="draft",
                repair_attempt=1,
                failed_constraints=(failure_grade,),
                evidence=(),
                required_hard_constraints=(),
            )

        passing_grade = create_grade(
            TARGET,
            metric="target",
            status=GradeStatus.PASS,
        )
        passing_grader = ScriptedGrader[str](
            descriptor=TARGET,
            outcomes=(passing_grade,),
        )
        no_failure_loop, no_failure_repair = create_loop(
            graders=(passing_grader,),
            policy=ConstantPolicy(GradeDecision.REPAIR),
        )
        context, _ = create_context()
        missing = asyncio.run(
            execute_step(no_failure_loop, "draft", context)
        )
        self.assertIsNotNone(missing.failure)
        if missing.failure is None:
            raise AssertionError("missing failure outcome has no failure")
        self.assertEqual(
            missing.failure.code,
            "repair.failed_constraints_missing",
        )
        self.assertEqual(no_failure_repair.calls, [])

        source = CancellationSource()
        source.cancel("caller stopped")
        cancelled_grader = ScriptedGrader[str](
            descriptor=TARGET,
            outcomes=(passing_grade,),
        )
        cancelled_loop, cancelled_repair = create_loop(
            graders=(cancelled_grader,),
            policy=AnyFailurePolicy(),
        )
        context, _ = create_context(source=source)

        cancelled = asyncio.run(
            execute_step(cancelled_loop, "draft", context)
        )

        self.assertEqual(cancelled.status, ExecutionStatus.CANCELLED)
        self.assertEqual(cancelled_grader.calls, ())
        self.assertEqual(cancelled_repair.calls, [])


if __name__ == "__main__":
    unittest.main()
