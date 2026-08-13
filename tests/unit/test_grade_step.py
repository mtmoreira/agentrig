from __future__ import annotations

import asyncio
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
    GradeStatus,
    GraderDescriptor,
    InMemoryEventSink,
    NoOpRedactionPolicy,
    RunContext,
    RunId,
)
from agentrig.testing import ScriptedGrader
from agentrig.workflow import (
    GradeStep,
    GradeStepResult,
    Step,
    StepDescriptor,
    execute_step,
)


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 14, 1, 0, tzinfo=UTC)

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


STRUCTURE = GraderDescriptor(grader_id="story.structure", version="1")
QUALITY = GraderDescriptor(grader_id="story.quality", version="2")


def create_grade(
    descriptor: GraderDescriptor,
    *,
    metric: str,
    status: GradeStatus = GradeStatus.PASS,
    classification: GradeClassification = GradeClassification.HARD,
    explanation: str = "Stable grading explanation.",
    evidence: tuple[GradeEvidence, ...] = (),
) -> Grade:
    return Grade(
        grader=descriptor,
        metric=metric,
        status=status,
        classification=classification,
        explanation=explanation,
        evidence=evidence,
    )


@dataclass
class RecordingPolicy:
    descriptor: GradePolicyDescriptor
    decision: GradeDecision
    calls: list[tuple[Grade, ...]] = field(default_factory=list)

    def decide(self, grades: GradeSequence[Grade]) -> GradeDecision:
        copied_grades = tuple(grades)
        self.calls.append(copied_grades)
        return self.decision


def create_policy(
    decision: GradeDecision = GradeDecision.CONTINUE,
) -> RecordingPolicy:
    return RecordingPolicy(
        descriptor=GradePolicyDescriptor(
            policy_id="story.release",
            version="3",
        ),
        decision=decision,
    )


class GradeStepTest(unittest.TestCase):
    def test_records_grades_and_policy_decision_separately(self) -> None:
        structure_grade = create_grade(
            STRUCTURE,
            metric="required_sections",
            explanation="password=private",
        )
        quality_grade = create_grade(
            QUALITY,
            metric="coherence",
            status=GradeStatus.FAILURE,
            classification=GradeClassification.SOFT,
            evidence=(GradeEvidence(field_path=("chapters", 2)),),
        )
        structure = ScriptedGrader[str](
            descriptor=STRUCTURE,
            outcomes=(structure_grade,),
        )
        quality = ScriptedGrader[str](
            descriptor=QUALITY,
            outcomes=(quality_grade,),
        )
        policy = create_policy(GradeDecision.REPAIR)
        step = GradeStep[str](
            graders=(structure, quality),
            policy=policy,
        )
        context, sink = create_context()

        result = asyncio.run(step.run("draft", context))

        self.assertIsInstance(step, Step)
        self.assertEqual(
            step.descriptor,
            StepDescriptor(
                step_id="story.release",
                version="3",
                effect_profile=EffectProfile.READ_ONLY,
            ),
        )
        self.assertEqual(result.subject, "draft")
        self.assertEqual(result.grades, (structure_grade, quality_grade))
        self.assertEqual(result.policy, policy.descriptor)
        self.assertEqual(result.decision, GradeDecision.REPAIR)
        self.assertEqual(policy.calls, [(structure_grade, quality_grade)])
        self.assertEqual(
            [event.kind for event in sink.events],
            [
                EventKind.GRADE_PRODUCED,
                EventKind.GRADE_PRODUCED,
                EventKind.GRADE_POLICY_DECIDED,
            ],
        )
        self.assertEqual(
            [event.run_id for event in sink.events],
            [RunId("run-2"), RunId("run-3"), RunId("run-1")],
        )
        self.assertEqual(
            [event.parent_run_id for event in sink.events[:2]],
            [RunId("run-1"), RunId("run-1")],
        )
        grade_attributes = sink.events[1].attributes
        self.assertEqual(grade_attributes["status"], "failure")
        self.assertEqual(grade_attributes["classification"], "soft")
        self.assertNotIn("explanation", grade_attributes)
        self.assertNotIn("evidence", grade_attributes)
        self.assertNotIn("subject", grade_attributes)
        self.assertNotIn("private", repr(sink.events))
        self.assertEqual(
            sink.events[-1].attributes,
            {
                "policy_id": "story.release",
                "policy_version": "3",
                "decision": "repair",
                "grade_count": 2,
            },
        )
        self.assertEqual(structure.calls[0].subject, "draft")
        self.assertIs(structure.calls[0].context.event_sink, context.event_sink)

    def test_failing_grade_is_control_flow_data_not_execution_failure(self) -> None:
        grade = create_grade(
            STRUCTURE,
            metric="required_sections",
            status=GradeStatus.FAILURE,
        )
        step = GradeStep[str](
            graders=(
                ScriptedGrader[str](
                    descriptor=STRUCTURE,
                    outcomes=(grade,),
                ),
            ),
            policy=create_policy(GradeDecision.BLOCK),
        )
        context, _ = create_context()

        outcome = asyncio.run(execute_step(step, "draft", context))

        self.assertEqual(outcome.status, ExecutionStatus.SUCCEEDED)
        result = outcome.unwrap()
        self.assertEqual(result.decision, GradeDecision.BLOCK)
        self.assertEqual(result.grades, (grade,))

    def test_preserves_normalized_grader_failure_and_skips_policy(self) -> None:
        failure = Failure(
            kind=FailureKind.GRADER_FAILED,
            message="grader dependency unavailable",
            code="grader.unavailable",
        )
        policy = create_policy()
        step = GradeStep[str](
            graders=(
                ScriptedGrader[str](
                    descriptor=STRUCTURE,
                    outcomes=(failure,),
                ),
            ),
            policy=policy,
        )
        context, sink = create_context()

        outcome = asyncio.run(execute_step(step, "draft", context))

        self.assertEqual(outcome.status, ExecutionStatus.FAILED)
        self.assertIs(outcome.failure, failure)
        self.assertEqual(policy.calls, [])
        self.assertEqual(
            [event.kind for event in sink.events],
            [EventKind.STEP_STARTED, EventKind.STEP_COMPLETED],
        )

    def test_sanitizes_broken_grader_and_policy_implementations(self) -> None:
        @dataclass(frozen=True)
        class BrokenGrader:
            descriptor: GraderDescriptor

            async def grade(self, subject: str, context: object) -> Grade:
                del subject, context
                raise RuntimeError("password=private")

        context, _ = create_context()
        broken_grader = GradeStep[str](
            graders=(BrokenGrader(STRUCTURE),),  # type: ignore[arg-type]
            policy=create_policy(),
        )

        grader_outcome = asyncio.run(
            execute_step(broken_grader, "draft", context)
        )

        self.assertIsNotNone(grader_outcome.failure)
        if grader_outcome.failure is None:
            raise AssertionError("failed outcome has no failure")
        self.assertEqual(grader_outcome.failure.kind, FailureKind.GRADER_FAILED)
        self.assertEqual(grader_outcome.failure.code, "grader.execution_failed")
        self.assertNotIn("private", grader_outcome.failure.message)

        @dataclass(frozen=True)
        class InvalidPolicy:
            descriptor: GradePolicyDescriptor

            def decide(self, grades: GradeSequence[Grade]) -> str:
                del grades
                return "continue"

        context, _ = create_context()
        invalid_policy = GradeStep[str](
            graders=(
                ScriptedGrader[str](
                    descriptor=STRUCTURE,
                    outcomes=(
                        create_grade(STRUCTURE, metric="required_sections"),
                    ),
                ),
            ),
            policy=InvalidPolicy(
                GradePolicyDescriptor(policy_id="invalid", version="1")
            ),  # type: ignore[arg-type]
        )

        policy_outcome = asyncio.run(
            execute_step(invalid_policy, "draft", context)
        )

        self.assertIsNotNone(policy_outcome.failure)
        if policy_outcome.failure is None:
            raise AssertionError("failed outcome has no failure")
        self.assertEqual(policy_outcome.failure.kind, FailureKind.UNEXPECTED)
        self.assertEqual(
            policy_outcome.failure.code,
            "grade_policy.invalid_result",
        )

    def test_cancellation_stops_remaining_graders_and_policy(self) -> None:
        source = CancellationSource()

        @dataclass(frozen=True)
        class CancellingGrader:
            descriptor: GraderDescriptor

            async def grade(self, subject: str, context: object) -> Grade:
                del subject, context
                source.cancel("stopped during grading")
                return create_grade(self.descriptor, metric="required_sections")

        downstream = ScriptedGrader[str](
            descriptor=QUALITY,
            outcomes=(create_grade(QUALITY, metric="coherence"),),
        )
        policy = create_policy()
        step = GradeStep[str](
            graders=(
                CancellingGrader(STRUCTURE),  # type: ignore[arg-type]
                downstream,
            ),
            policy=policy,
        )
        context, sink = create_context(source=source)

        outcome = asyncio.run(execute_step(step, "draft", context))

        self.assertEqual(outcome.status, ExecutionStatus.CANCELLED)
        self.assertEqual(downstream.calls, ())
        self.assertEqual(policy.calls, [])
        self.assertEqual(
            [event.kind for event in sink.events],
            [
                EventKind.STEP_STARTED,
                EventKind.GRADE_PRODUCED,
                EventKind.STEP_COMPLETED,
            ],
        )

    def test_preserves_artifact_subject_and_evidence_for_repair(self) -> None:
        artifact = ArtifactRef(
            artifact_id=ArtifactId("artifact-draft"),
            kind="story",
            media_type="text/plain",
            producer_run_id=RunId("run-producer"),
            workspace_path="outputs/story.txt",
        )
        grade = create_grade(
            STRUCTURE,
            metric="required_sections",
            status=GradeStatus.FAILURE,
            evidence=(GradeEvidence(artifact_id=artifact.artifact_id),),
        )
        step = GradeStep[ArtifactRef](
            graders=(
                ScriptedGrader[ArtifactRef](
                    descriptor=STRUCTURE,
                    outcomes=(grade,),
                ),
            ),
            policy=create_policy(GradeDecision.REPAIR),
        )
        context, _ = create_context()

        result = asyncio.run(step.run(artifact, context))

        self.assertIs(result.subject, artifact)
        self.assertEqual(result.grades[0].evidence[0].artifact_id, artifact.artifact_id)

    def test_rejects_invalid_configuration_and_results(self) -> None:
        grade = create_grade(STRUCTURE, metric="required_sections")
        grader = ScriptedGrader[str](
            descriptor=STRUCTURE,
            outcomes=(grade,),
        )
        with self.assertRaises(ValueError):
            GradeStep[str](graders=(), policy=create_policy())
        with self.assertRaises(TypeError):
            GradeStep[str](
                graders=("invalid",),  # type: ignore[arg-type]
                policy=create_policy(),
            )
        with self.assertRaises(ValueError):
            GradeStep[str](
                graders=(grader, grader),
                policy=create_policy(),
            )
        with self.assertRaises(TypeError):
            GradeStep[str](
                graders=(grader,),
                policy="invalid",  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            asyncio.run(
                GradeStep[str](
                    graders=(grader,),
                    policy=create_policy(),
                ).run("draft", "invalid")  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            GradeStepResult(
                subject="draft",
                grades=(grade, grade),
                policy=create_policy().descriptor,
                decision=GradeDecision.CONTINUE,
            )


if __name__ == "__main__":
    unittest.main()
