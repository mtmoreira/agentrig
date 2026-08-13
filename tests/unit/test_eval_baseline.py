from __future__ import annotations

import unittest

from agentrig.core import (
    ArtifactId,
    ArtifactRef,
    ExecutionOutcome,
    Failure,
    FailureKind,
    Grade,
    GradeClassification,
    GradeStatus,
    GraderDescriptor,
    GraderUsage,
    NoOpRedactionPolicy,
    RunId,
    ScoreRange,
)
from agentrig.evals import (
    EVAL_BASELINE_SCHEMA_VERSION,
    DeterministicPromotionPolicy,
    EvalBaseline,
    EvalCaseResult,
    EvalChangeKind,
    EvalComparisonTolerance,
    EvalGraderFailure,
    EvalInconclusiveReason,
    EvalMetric,
    EvalReport,
    EvalReportRetention,
    EvalRunResult,
    EvalSummary,
    EvalTargetDescriptor,
    EvalTargetKind,
    PromotionDecision,
    PromotionPolicy,
    PromotionPolicyDescriptor,
    compare_to_baseline,
)


QUALITY = GraderDescriptor(
    grader_id="story.quality",
    version="1",
    agentic=True,
)
ADDITIONAL = GraderDescriptor(
    grader_id="story.additional",
    version="1",
)


def create_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId("artifact-1"),
        kind="story",
        media_type="text/plain",
        producer_run_id=RunId("run-producer"),
        workspace_path="outputs/story.txt",
        labels={"password": "artifact-private"},
    )


def create_report(
    *,
    target_id: str = "story.generate",
    target_version: str = "1",
    dataset_id: str = "story-quality",
    dataset_version: str = "2026-08-14",
    case_version: str = "1",
    failure: Failure | None = None,
    grade_status: GradeStatus = GradeStatus.PASS,
    grade_classification: GradeClassification = GradeClassification.HARD,
    score: float = 0.8,
    score_range: ScoreRange | None = None,
    include_grade: bool = True,
    additional_grade_status: GradeStatus | None = None,
    grader_failure: bool = False,
    duration_seconds: float = 1.0,
    grader_latency_seconds: float = 0.25,
    grader_cost: float = 0.10,
    retain_payloads: bool = False,
) -> EvalReport:
    effective_range = (
        score_range
        if score_range is not None
        else ScoreRange(minimum=0.0, maximum=1.0)
    )
    grades: list[Grade] = []
    if include_grade and failure is None:
        grades.append(
            Grade(
                grader=QUALITY,
                metric="quality",
                status=grade_status,
                classification=grade_classification,
                explanation="Bearer abcdefghijklmnop",
                score=score,
                score_range=effective_range,
                usage=GraderUsage(
                    latency_seconds=grader_latency_seconds,
                    cost=grader_cost,
                    currency="USD",
                ),
            )
        )
    if additional_grade_status is not None and failure is None:
        grades.append(
            Grade(
                grader=ADDITIONAL,
                metric="additional",
                status=additional_grade_status,
                classification=GradeClassification.SOFT,
                explanation="Additional evidence.",
            )
        )

    grader_failures = (
        (
            EvalGraderFailure(
                grader=QUALITY,
                failure=Failure(
                    kind=FailureKind.GRADER_FAILED,
                    message="grader dependency unavailable",
                    code="grader.unavailable",
                ),
            ),
        )
        if grader_failure
        else ()
    )
    outcome: ExecutionOutcome[object]
    if failure is None:
        outcome = ExecutionOutcome.succeeded(
            {"answer": "output-private"},
            artifacts=(create_artifact(),),
        )
    else:
        outcome = ExecutionOutcome.from_failure(failure)
    case = EvalCaseResult[object](
        case_id="case-1",
        case_version=case_version,
        run_id=RunId("run-case-1"),
        outcome=outcome,
        grades=tuple(grades),
        grader_failures=grader_failures,
        duration_seconds=duration_seconds,
    )
    cases = (case,)
    run = EvalRunResult(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        target=EvalTargetDescriptor(
            target_id=target_id,
            version=target_version,
            kind=EvalTargetKind.AGENT,
        ),
        cases=cases,
        summary=EvalSummary.from_cases(cases),
    )
    return EvalReport.from_run(
        run,
        retention=(
            EvalReportRetention(
                outputs=True,
                artifacts=True,
                grade_details=True,
            )
            if retain_payloads
            else EvalReportRetention()
        ),
        environment={"api_key": "environment-private"},
        redaction_policy=NoOpRedactionPolicy(),
    )


def create_baseline() -> EvalBaseline:
    return EvalBaseline.from_report(
        baseline_id="story.release",
        version="2026-08-14",
        report=create_report(retain_payloads=True),
    )


def create_policy() -> DeterministicPromotionPolicy:
    return DeterministicPromotionPolicy(
        descriptor=PromotionPolicyDescriptor(
            policy_id="story.promotion",
            version="1",
        )
    )


class EvalBaselineTest(unittest.TestCase):
    def test_strips_payloads_redacts_and_round_trips_deterministically(
        self,
    ) -> None:
        baseline = create_baseline()

        self.assertEqual(
            baseline.schema_version,
            EVAL_BASELINE_SCHEMA_VERSION,
        )
        self.assertEqual(baseline.report.retention, EvalReportRetention())
        self.assertEqual(baseline.report.environment, {})
        self.assertFalse(baseline.report.cases[0].output_retained)
        self.assertIsNone(baseline.report.cases[0].output)
        self.assertEqual(baseline.report.cases[0].artifacts, ())
        self.assertIsNone(baseline.report.cases[0].grades[0].explanation)

        serialized = baseline.to_json()
        restored = EvalBaseline.from_json(serialized)

        self.assertEqual(restored, baseline)
        self.assertEqual(restored.to_json(), serialized)
        for private_value in (
            "output-private",
            "artifact-private",
            "environment-private",
            "abcdefghijklmnop",
        ):
            self.assertNotIn(private_value, serialized)

    def test_known_quality_and_resource_regressions_are_rejected(self) -> None:
        baseline = create_baseline()
        candidate = create_report(
            target_version="2",
            grade_status=GradeStatus.FAILURE,
            score=0.6,
            duration_seconds=1.5,
            grader_latency_seconds=0.4,
            grader_cost=0.2,
        )

        comparison = compare_to_baseline(baseline, candidate)
        policy: PromotionPolicy = create_policy()

        self.assertEqual(
            tuple(change.metric for change in comparison.regressions),
            (
                EvalMetric.GRADE_STATUS,
                EvalMetric.GRADE_SCORE,
                EvalMetric.DURATION_SECONDS,
                EvalMetric.GRADER_LATENCY_SECONDS,
                EvalMetric.GRADER_COST,
            ),
        )
        self.assertEqual(comparison.improvements, ())
        self.assertEqual(policy.decide(comparison), PromotionDecision.REJECT)
        self.assertEqual(comparison.baseline_target_version, "1")
        self.assertEqual(comparison.candidate_target_version, "2")

    def test_improvements_are_recorded_and_candidate_is_promoted(self) -> None:
        comparison = compare_to_baseline(
            create_baseline(),
            create_report(
                target_version="2",
                score=0.9,
                duration_seconds=0.5,
                grader_latency_seconds=0.1,
                grader_cost=0.05,
            ),
        )

        self.assertEqual(comparison.regressions, ())
        self.assertEqual(
            tuple(change.metric for change in comparison.improvements),
            (
                EvalMetric.GRADE_SCORE,
                EvalMetric.DURATION_SECONDS,
                EvalMetric.GRADER_LATENCY_SECONDS,
                EvalMetric.GRADER_COST,
            ),
        )
        self.assertEqual(
            create_policy().decide(comparison),
            PromotionDecision.PROMOTE,
        )

    def test_numeric_tolerances_are_explicit_and_bounded(self) -> None:
        tolerance = EvalComparisonTolerance(
            max_grade_score_drop=0.05,
            max_duration_increase_ratio=0.10,
            max_grader_latency_increase_ratio=0.10,
            max_grader_cost_increase_ratio=0.10,
        )
        within_tolerance = compare_to_baseline(
            create_baseline(),
            create_report(
                score=0.76,
                duration_seconds=1.05,
                grader_latency_seconds=0.26,
                grader_cost=0.105,
            ),
            tolerance=tolerance,
        )

        self.assertEqual(within_tolerance.changes, ())
        self.assertEqual(
            create_policy().decide(within_tolerance),
            PromotionDecision.PROMOTE,
        )

        cost_regression = compare_to_baseline(
            create_baseline(),
            create_report(grader_cost=0.12),
            tolerance=tolerance,
        )

        self.assertEqual(len(cost_regression.regressions), 1)
        change = cost_regression.regressions[0]
        self.assertEqual(change.metric, EvalMetric.GRADER_COST)
        self.assertEqual(change.currency, "USD")
        self.assertAlmostEqual(change.allowed_regression, 0.01)

    def test_missing_and_new_nonpassing_grades_are_regressions(self) -> None:
        missing = compare_to_baseline(
            create_baseline(),
            create_report(include_grade=False),
        )
        added_failure = compare_to_baseline(
            create_baseline(),
            create_report(additional_grade_status=GradeStatus.FAILURE),
        )

        self.assertEqual(
            missing.regressions[0].candidate_value,
            None,
        )
        self.assertEqual(
            added_failure.regressions[0].candidate_value,
            GradeStatus.FAILURE.value,
        )

    def test_blocked_cancelled_and_grader_failures_are_inconclusive(self) -> None:
        cases = (
            (
                Failure(
                    kind=FailureKind.WORKFLOW_BLOCKED,
                    message="provider credentials are unavailable",
                ),
                EvalInconclusiveReason.BLOCKED_CASE,
            ),
            (
                Failure(
                    kind=FailureKind.CANCELLED,
                    message="caller cancelled evaluation",
                ),
                EvalInconclusiveReason.CANCELLED_CASE,
            ),
        )
        for failure, reason in cases:
            with self.subTest(reason=reason):
                comparison = compare_to_baseline(
                    create_baseline(),
                    create_report(failure=failure),
                )
                self.assertEqual(comparison.inconclusive[0].reason, reason)
                self.assertEqual(
                    create_policy().decide(comparison),
                    PromotionDecision.INCONCLUSIVE,
                )

        grader_failure = compare_to_baseline(
            create_baseline(),
            create_report(include_grade=False, grader_failure=True),
        )
        self.assertEqual(
            grader_failure.inconclusive[0].reason,
            EvalInconclusiveReason.GRADER_FAILURE,
        )
        self.assertEqual(
            create_policy().decide(grader_failure),
            PromotionDecision.INCONCLUSIVE,
        )

    def test_requires_compatible_dataset_cases_target_and_score_range(
        self,
    ) -> None:
        baseline = create_baseline()
        incompatible = (
            create_report(dataset_version="different"),
            create_report(target_id="different"),
            create_report(case_version="different"),
            create_report(score_range=ScoreRange(minimum=0.0, maximum=2.0)),
            create_report(
                grade_classification=GradeClassification.SOFT,
            ),
        )
        for candidate in incompatible:
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    compare_to_baseline(baseline, candidate)

    def test_changed_failure_kind_is_a_regression(self) -> None:
        approved_failure = Failure(
            kind=FailureKind.INVALID_INPUT,
            message="known invalid fixture",
        )
        baseline = EvalBaseline.from_report(
            baseline_id="known-failure",
            version="1",
            report=create_report(failure=approved_failure),
        )
        candidate = create_report(
            target_version="2",
            failure=Failure(
                kind=FailureKind.UNEXPECTED,
                message="candidate crashed",
            ),
        )

        comparison = compare_to_baseline(baseline, candidate)

        self.assertEqual(len(comparison.regressions), 1)
        self.assertEqual(
            comparison.regressions[0].baseline_value,
            "failed:invalid_input",
        )
        self.assertEqual(
            comparison.regressions[0].candidate_value,
            "failed:unexpected",
        )
        self.assertEqual(
            create_policy().decide(comparison),
            PromotionDecision.REJECT,
        )

    def test_rejects_inconclusive_baselines_and_malformed_configuration(
        self,
    ) -> None:
        blocked = create_report(
            failure=Failure(
                kind=FailureKind.WORKFLOW_BLOCKED,
                message="provider credentials are unavailable",
            )
        )
        with self.assertRaises(ValueError):
            EvalBaseline.from_report(
                baseline_id="invalid",
                version="1",
                report=blocked,
            )
        with self.assertRaises(ValueError):
            EvalComparisonTolerance(max_grade_score_drop=-0.1)
        with self.assertRaises(TypeError):
            create_policy().decide("invalid")  # type: ignore[arg-type]

        baseline = create_baseline()
        unknown = baseline.to_data()
        unknown["unknown"] = True
        with self.assertRaises(ValueError):
            EvalBaseline.from_data(unknown)
        wrong_version = baseline.to_data()
        wrong_version["schema_version"] = 2
        with self.assertRaises(ValueError):
            EvalBaseline.from_data(wrong_version)
        with self.assertRaises(ValueError):
            EvalBaseline.from_json("[]")

    def test_change_vocabulary_has_stable_wire_values(self) -> None:
        self.assertEqual(
            {item.value for item in EvalChangeKind},
            {"regression", "improvement"},
        )
        self.assertEqual(
            {item.value for item in PromotionDecision},
            {"promote", "reject", "inconclusive"},
        )


if __name__ == "__main__":
    unittest.main()
