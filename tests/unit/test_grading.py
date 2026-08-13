from __future__ import annotations

import asyncio
import math
import unittest
from dataclasses import dataclass
from datetime import UTC, datetime

from agentrig.core import (
    GRADE_SCHEMA_VERSION,
    AgentRigError,
    ArtifactId,
    CancellationSource,
    Failure,
    FailureKind,
    Grade,
    GradeClassification,
    GradeEvidence,
    GradeStatus,
    Grader,
    GraderDescriptor,
    GraderUsage,
    GradingContext,
    RunContext,
    RunId,
    ScoreRange,
)


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 12, 21, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


@dataclass
class SequentialRunIdGenerator:
    next_value: int = 1

    def generate(self) -> RunId:
        run_id = RunId(f"run-{self.next_value}")
        self.next_value += 1
        return run_id


def create_context() -> GradingContext:
    source = CancellationSource()
    return GradingContext(
        run_context=RunContext.create_root(
            clock=FixedClock(),
            id_generator=SequentialRunIdGenerator(),
            cancellation=source.token,
        )
    )


def create_grade(**overrides: object) -> Grade:
    values: dict[str, object] = {
        "grader": GraderDescriptor(
            grader_id="story.structure",
            version="1.0.0",
        ),
        "metric": "required_sections",
        "status": GradeStatus.FAILURE,
        "classification": GradeClassification.HARD,
        "explanation": "The ending section is missing.",
        "evidence": (GradeEvidence(field_path=("ending",)),),
    }
    values.update(overrides)
    return Grade(**values)  # type: ignore[arg-type]


class PassingGrader:
    @property
    def descriptor(self) -> GraderDescriptor:
        return GraderDescriptor(grader_id="text.present", version="1")

    async def grade(self, subject: str, context: GradingContext) -> Grade:
        context.run_context.cancellation.raise_if_cancelled()
        return create_grade(
            grader=self.descriptor,
            metric="nonempty",
            status=GradeStatus.PASS,
            explanation=f"Received {len(subject)} characters.",
            evidence=(),
        )


class FailingGrader:
    @property
    def descriptor(self) -> GraderDescriptor:
        return GraderDescriptor(grader_id="broken.grader", version="1")

    async def grade(self, subject: str, context: GradingContext) -> Grade:
        del subject, context
        raise AgentRigError(
            Failure(
                kind=FailureKind.GRADER_FAILED,
                message="grader could not evaluate the subject",
            )
        )


async def grade_text(
    grader: Grader[str],
    subject: str,
    context: GradingContext,
) -> Grade:
    return await grader.grade(subject, context)


class GradeVocabularyTest(unittest.TestCase):
    def test_wire_values_are_stable(self) -> None:
        self.assertEqual(
            {status.value for status in GradeStatus},
            {"failure", "pass", "warning"},
        )
        self.assertEqual(
            {classification.value for classification in GradeClassification},
            {"hard", "soft"},
        )


class GraderContractTest(unittest.TestCase):
    def test_protocol_supports_an_async_typed_grader(self) -> None:
        grader = PassingGrader()

        grade = asyncio.run(grade_text(grader, "draft", create_context()))

        self.assertIsInstance(grader, Grader)
        self.assertEqual(grade.status, GradeStatus.PASS)
        self.assertEqual(grade.grader, grader.descriptor)

    def test_context_requires_explicit_run_dependencies(self) -> None:
        context = create_context()

        self.assertEqual(context.run_context.run_id, RunId("run-1"))
        context.event_sink.emit  # Protocol surface is available.
        with self.assertRaises(TypeError):
            GradingContext(run_context="run-1")  # type: ignore[arg-type]

    def test_grader_failure_is_distinct_from_a_failing_grade(self) -> None:
        subject_failure = create_grade(status=GradeStatus.FAILURE)

        with self.assertRaises(AgentRigError) as raised:
            asyncio.run(grade_text(FailingGrader(), "draft", create_context()))

        self.assertEqual(subject_failure.status, GradeStatus.FAILURE)
        self.assertEqual(raised.exception.failure.kind, FailureKind.GRADER_FAILED)


class GradeEvidenceTest(unittest.TestCase):
    def test_references_an_artifact_or_structured_field(self) -> None:
        path = ["chapters", 0, "title"]
        field_evidence = GradeEvidence(field_path=path)  # type: ignore[arg-type]
        path.append("mutated")
        artifact_evidence = GradeEvidence(
            artifact_id=ArtifactId("artifact-cover")
        )

        self.assertEqual(field_evidence.field_path, ("chapters", 0, "title"))
        self.assertEqual(
            artifact_evidence.artifact_id,
            ArtifactId("artifact-cover"),
        )

    def test_rejects_ambiguous_or_invalid_references(self) -> None:
        invalid_values = (
            {},
            {
                "artifact_id": ArtifactId("artifact-1"),
                "field_path": ("title",),
            },
            {"field_path": ()},
            {"field_path": ("chapters", -1)},
            {"field_path": (True,)},
        )
        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    GradeEvidence(**values)  # type: ignore[arg-type]


class GradeTest(unittest.TestCase):
    def test_preserves_hard_failure_evidence(self) -> None:
        evidence = [GradeEvidence(field_path=("ending",))]

        grade = create_grade(evidence=evidence)
        evidence.append(GradeEvidence(field_path=("title",)))

        self.assertEqual(grade.classification, GradeClassification.HARD)
        self.assertEqual(grade.status, GradeStatus.FAILURE)
        self.assertEqual(
            grade.evidence,
            (GradeEvidence(field_path=("ending",)),),
        )

    def test_score_requires_a_finite_calibrated_range(self) -> None:
        grade = create_grade(
            classification=GradeClassification.SOFT,
            score=0.8,
            score_range=ScoreRange(minimum=0, maximum=1),
        )

        self.assertEqual(grade.score, 0.8)
        self.assertEqual(grade.score_range, ScoreRange(minimum=0, maximum=1))

        invalid_values = (
            {"score": 0.8},
            {"score_range": ScoreRange(minimum=0, maximum=1)},
            {
                "score": 2,
                "score_range": ScoreRange(minimum=0, maximum=1),
            },
            {
                "score": math.nan,
                "score_range": ScoreRange(minimum=0, maximum=1),
            },
        )
        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    create_grade(**values)

        for minimum, maximum in ((1, 1), (2, 1), (0, math.inf)):
            with self.subTest(minimum=minimum, maximum=maximum):
                with self.assertRaises(ValueError):
                    ScoreRange(minimum=minimum, maximum=maximum)

    def test_agentic_grades_include_cost_and_latency(self) -> None:
        descriptor = GraderDescriptor(
            grader_id="semantic.review",
            version="model-2026-08-12",
            agentic=True,
        )
        usage = GraderUsage(latency_seconds=1.25, cost=0.004, currency="USD")

        grade = create_grade(grader=descriptor, usage=usage)

        self.assertEqual(grade.usage, usage)
        with self.assertRaises(ValueError):
            create_grade(grader=descriptor)
        with self.assertRaises(ValueError):
            GraderUsage(latency_seconds=-1, cost=0)

    def test_json_serialization_round_trips(self) -> None:
        grade = create_grade(
            classification=GradeClassification.SOFT,
            score=0.75,
            score_range=ScoreRange(minimum=0, maximum=1),
            evidence=(
                GradeEvidence(artifact_id=ArtifactId("artifact-1")),
                GradeEvidence(field_path=("chapters", 2, "title")),
            ),
        )

        serialized = grade.to_json()
        restored = Grade.from_json(serialized)

        self.assertEqual(restored, grade)
        self.assertIn('"schema_version":1', serialized)
        data = grade.to_data()
        data["evidence"].append({})  # type: ignore[union-attr]
        self.assertEqual(len(grade.evidence), 2)

    def test_deserialization_rejects_unknown_fields_and_values(self) -> None:
        data = create_grade().to_data()
        data["unexpected"] = True
        with self.assertRaises(ValueError):
            Grade.from_data(data)

        data = create_grade().to_data()
        data["status"] = "unknown"
        with self.assertRaises(ValueError):
            Grade.from_data(data)

    def test_schema_version_and_descriptor_are_validated(self) -> None:
        self.assertEqual(create_grade().schema_version, GRADE_SCHEMA_VERSION)
        for version in (0, 2, True):
            with self.subTest(version=version):
                with self.assertRaises(ValueError):
                    create_grade(schema_version=version)
        with self.assertRaises(ValueError):
            GraderDescriptor(grader_id=" padded", version="1")
        with self.assertRaises(TypeError):
            GraderDescriptor(
                grader_id="grader",
                version="1",
                agentic=1,  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
