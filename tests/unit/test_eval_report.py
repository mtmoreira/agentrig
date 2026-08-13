from __future__ import annotations

import unittest
from collections.abc import Mapping

from agentrig.core import (
    REDACTED_VALUE,
    ArtifactId,
    ArtifactRef,
    ContentDigest,
    ExecutionOutcome,
    Failure,
    FailureKind,
    Grade,
    GradeClassification,
    GradeEvidence,
    GradeStatus,
    GraderDescriptor,
    GraderUsage,
    NoOpRedactionPolicy,
    RunId,
    SafeRedactionPolicy,
    ScoreRange,
)
from agentrig.evals import (
    EVAL_REPORT_SCHEMA_VERSION,
    EvalCaseResult,
    EvalGraderFailure,
    EvalReport,
    EvalReportRetention,
    EvalRunResult,
    EvalSummary,
    EvalTargetDescriptor,
    EvalTargetKind,
)


TARGET = EvalTargetDescriptor(
    target_id="story.generate",
    version="2",
    kind=EvalTargetKind.AGENT,
)
QUALITY = GraderDescriptor(
    grader_id="story.quality",
    version="3",
    agentic=True,
)
SAFETY = GraderDescriptor(
    grader_id="story.safety",
    version="1",
)


def create_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId("artifact-1"),
        kind="story",
        media_type="text/plain",
        producer_run_id=RunId("run-producer"),
        workspace_path="outputs/story.txt",
        content_digest=ContentDigest(algorithm="sha256", value="abc123"),
        labels={"password": "artifact-private", "stage": "draft"},
        provider_lineage={
            "session_id": "provider-private",
            "model": "example-model",
        },
    )


def create_run(*, output: object | None = None) -> EvalRunResult[object]:
    artifact = create_artifact()
    successful = EvalCaseResult[object](
        case_id="case-success",
        case_version="1",
        run_id=RunId("run-success"),
        outcome=ExecutionOutcome.succeeded(
            (
                {
                    "answer": "SENSITIVE OUTPUT",
                    "private_field": "custom-private",
                    "notes": "Bearer abcdefghijklmnop",
                }
                if output is None
                else output
            ),
            artifacts=(artifact,),
        ),
        grades=(
            Grade(
                grader=QUALITY,
                metric="quality",
                status=GradeStatus.WARNING,
                classification=GradeClassification.SOFT,
                explanation="Bearer abcdefghijklmnop",
                evidence=(GradeEvidence(artifact_id=artifact.artifact_id),),
                score=0.75,
                score_range=ScoreRange(minimum=0.0, maximum=1.0),
                usage=GraderUsage(
                    latency_seconds=0.25,
                    cost=0.10,
                    currency="USD",
                ),
            ),
        ),
        grader_failures=(
            EvalGraderFailure(
                grader=SAFETY,
                failure=Failure(
                    kind=FailureKind.GRADER_FAILED,
                    message="grader dependency unavailable",
                    code="grader.unavailable",
                    metadata={"api_key": "failure-private"},
                ),
            ),
        ),
        duration_seconds=0.5,
    )
    failed = EvalCaseResult[object](
        case_id="case-failure",
        case_version="1",
        run_id=RunId("run-failure"),
        outcome=ExecutionOutcome.from_failure(
            Failure(
                kind=FailureKind.INVALID_INPUT,
                message="target rejected the case",
                code="target.invalid_input",
                metadata={"password": "target-private"},
            )
        ),
        duration_seconds=0.25,
    )
    cases = (successful, failed)
    return EvalRunResult(
        dataset_id="story-quality",
        dataset_version="2026-08-14",
        target=TARGET,
        cases=cases,
        summary=EvalSummary.from_cases(cases),
    )


class EvalReportTest(unittest.TestCase):
    def test_safe_defaults_omit_private_payloads_and_preserve_aggregates(
        self,
    ) -> None:
        report = EvalReport.from_run(
            create_run(),
            environment={"api_key": "environment-private", "python": "3.13"},
        )

        self.assertEqual(report.schema_version, EVAL_REPORT_SCHEMA_VERSION)
        self.assertEqual(report.retention, EvalReportRetention())
        self.assertFalse(report.cases[0].output_retained)
        self.assertIsNone(report.cases[0].output)
        self.assertEqual(report.cases[0].artifacts, ())
        self.assertIsNone(report.cases[0].grades[0].explanation)
        self.assertEqual(report.cases[0].grades[0].evidence, ())
        self.assertEqual(report.summary.case_count, 2)
        self.assertEqual(report.summary.succeeded_cases, 1)
        self.assertEqual(report.summary.failed_cases, 1)
        self.assertEqual(report.summary.warning_grades, 1)
        self.assertEqual(report.summary.grader_failure_count, 1)
        self.assertEqual(report.summary.grader_latency_seconds, 0.25)
        self.assertEqual(report.environment["api_key"], REDACTED_VALUE)
        self.assertEqual(report.environment["python"], "3.13")

        serialized = report.to_json()
        for private_value in (
            "SENSITIVE OUTPUT",
            "custom-private",
            "artifact-private",
            "provider-private",
            "environment-private",
            "failure-private",
            "target-private",
            "abcdefghijklmnop",
        ):
            self.assertNotIn(private_value, serialized)

    def test_opted_in_payloads_are_still_redacted(self) -> None:
        report = EvalReport.from_run(
            create_run(),
            retention=EvalReportRetention(
                outputs=True,
                artifacts=True,
                grade_details=True,
            ),
            environment={"api_key": "environment-private"},
            redaction_policy=SafeRedactionPolicy(
                additional_sensitive_keys=frozenset({"private_field"})
            ),
        )

        successful = report.cases[0]
        self.assertTrue(successful.output_retained)
        self.assertEqual(
            successful.output,
            {
                "answer": "SENSITIVE OUTPUT",
                "private_field": REDACTED_VALUE,
                "notes": REDACTED_VALUE,
            },
        )
        self.assertEqual(successful.artifacts[0].labels["password"], REDACTED_VALUE)
        self.assertEqual(
            successful.artifacts[0].content_digest,
            "sha256:abc123",
        )
        self.assertEqual(successful.artifacts[0].labels["stage"], "draft")
        self.assertEqual(
            successful.artifacts[0].provider_lineage["session_id"],
            REDACTED_VALUE,
        )
        self.assertEqual(successful.grades[0].explanation, REDACTED_VALUE)
        self.assertEqual(
            successful.grades[0].evidence[0].artifact_id,
            ArtifactId("artifact-1"),
        )
        self.assertEqual(
            successful.grader_failures[0].failure.metadata["api_key"],
            REDACTED_VALUE,
        )
        self.assertEqual(
            report.cases[1].failure.metadata["password"],
            REDACTED_VALUE,
        )

    def test_explicit_noop_policy_retains_opted_in_payloads(self) -> None:
        report = EvalReport.from_run(
            create_run(),
            retention=EvalReportRetention(
                outputs=True,
                artifacts=True,
                grade_details=True,
            ),
            environment={"api_key": "environment-private"},
            redaction_policy=NoOpRedactionPolicy(),
        )

        output = report.cases[0].output
        if not isinstance(output, Mapping):
            raise AssertionError("retained output was not a JSON object")
        self.assertEqual(output["private_field"], "custom-private")
        self.assertEqual(
            report.cases[0].artifacts[0].labels["password"],
            "artifact-private",
        )
        self.assertEqual(
            report.cases[0].grades[0].explanation,
            "Bearer abcdefghijklmnop",
        )
        self.assertEqual(report.environment["api_key"], "environment-private")

    def test_json_round_trip_is_deterministic_and_returns_detached_data(
        self,
    ) -> None:
        report = EvalReport.from_run(create_run())

        serialized = report.to_json()
        restored = EvalReport.from_json(serialized)
        detached = restored.to_data()

        self.assertEqual(restored, report)
        self.assertEqual(restored.to_json(), serialized)
        environment = detached["environment"]
        if not isinstance(environment, dict):
            raise AssertionError("serialized environment was not an object")
        environment["new"] = "value"
        self.assertNotIn("new", restored.environment)

    def test_strict_schema_rejects_unknown_versions_and_bad_aggregates(
        self,
    ) -> None:
        report = EvalReport.from_run(create_run())

        unknown = report.to_data()
        unknown["unknown"] = True
        with self.assertRaises(ValueError):
            EvalReport.from_data(unknown)

        wrong_version = report.to_data()
        wrong_version["schema_version"] = 2
        with self.assertRaises(ValueError):
            EvalReport.from_data(wrong_version)

        wrong_summary = report.to_data()
        summary = wrong_summary["summary"]
        if not isinstance(summary, dict):
            raise AssertionError("serialized summary was not an object")
        summary["warning_grades"] = 0
        with self.assertRaises(ValueError):
            EvalReport.from_data(wrong_summary)

        with self.assertRaises(ValueError):
            EvalReport.from_json("[]")

    def test_non_json_output_is_rejected_only_when_retained(self) -> None:
        run = create_run(output=object())

        omitted = EvalReport.from_run(run)

        self.assertIsNone(omitted.cases[0].output)
        with self.assertRaises(ValueError):
            EvalReport.from_run(
                run,
                retention=EvalReportRetention(outputs=True),
            )

    def test_rejects_invalid_retention_and_report_invariants(self) -> None:
        with self.assertRaises(TypeError):
            EvalReportRetention(outputs=1)  # type: ignore[arg-type]

        report = EvalReport.from_run(
            create_run(),
            retention=EvalReportRetention(outputs=True),
            redaction_policy=NoOpRedactionPolicy(),
        )
        data = report.to_data()
        retention = data["retention"]
        if not isinstance(retention, dict):
            raise AssertionError("serialized retention was not an object")
        retention["outputs"] = False
        with self.assertRaises(ValueError):
            EvalReport.from_data(data)


if __name__ == "__main__":
    unittest.main()
