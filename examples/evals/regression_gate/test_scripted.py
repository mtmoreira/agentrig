from __future__ import annotations

import asyncio
import unittest

from agentrig.core import EventKind, RunId
from agentrig.evals import (
    EvalBaseline,
    EvalInconclusiveReason,
    EvalMetric,
    EvalReport,
    EvalReportRetention,
    EvalTarget,
    PromotionDecision,
)

from examples.evals.regression_gate.scripted import (
    APPROVED_OUTPUTS,
    REGRESSED_OUTPUTS,
    run_scripted_report,
)
from examples.evals.regression_gate.suite import (
    RELEASE_DATASET,
    approve_baseline,
    assess_candidate,
)


class EvalRegressionGateExampleTest(unittest.TestCase):
    def test_equivalent_candidate_promotes_with_private_durable_reports(
        self,
    ) -> None:
        approved = asyncio.run(
            run_scripted_report(version="1", outputs=APPROVED_OUTPUTS)
        )
        candidate = asyncio.run(
            run_scripted_report(version="2", outputs=APPROVED_OUTPUTS)
        )
        baseline = approve_baseline(approved.report)
        assessment = assess_candidate(
            baseline=baseline,
            candidate=candidate.report,
        )

        self.assertIsInstance(candidate.target, EvalTarget)
        self.assertEqual(len(candidate.target.calls), 2)
        self.assertEqual(assessment.comparison.changes, ())
        self.assertEqual(assessment.decision, PromotionDecision.PROMOTE)
        self.assertEqual(candidate.report.retention, EvalReportRetention())
        self.assertEqual(candidate.report.summary.case_count, 2)
        self.assertEqual(candidate.report.summary.passing_grades, 2)
        self.assertTrue(
            all(not case.output_retained for case in candidate.report.cases)
        )
        self.assertTrue(
            all(
                grade.explanation is None and not grade.evidence
                for case in candidate.report.cases
                for grade in case.grades
            )
        )
        self.assertEqual(
            EvalReport.from_json(candidate.report.to_json()),
            candidate.report,
        )
        self.assertEqual(EvalBaseline.from_json(baseline.to_json()), baseline)
        self.assertEqual(baseline.report.environment, {})

        private_values = (
            "private roadmap notes",
            "private launch notes",
            APPROVED_OUTPUTS["alpha"],
            APPROVED_OUTPUTS["beta"],
            "example-private-environment-value",
        )
        serialized_report = candidate.report.to_json()
        serialized_baseline = baseline.to_json()
        serialized_events = " ".join(
            event.to_json() for event in candidate.events
        )
        for private_value in private_values:
            self.assertNotIn(private_value, serialized_report)
            self.assertNotIn(private_value, serialized_baseline)
            self.assertNotIn(private_value, serialized_events)

    def test_known_grade_regression_rejects_candidate(self) -> None:
        approved = asyncio.run(
            run_scripted_report(version="1", outputs=APPROVED_OUTPUTS)
        )
        candidate = asyncio.run(
            run_scripted_report(version="2", outputs=REGRESSED_OUTPUTS)
        )

        assessment = assess_candidate(
            baseline=approve_baseline(approved.report),
            candidate=candidate.report,
        )

        self.assertEqual(assessment.decision, PromotionDecision.REJECT)
        self.assertEqual(
            {change.metric for change in assessment.comparison.regressions},
            {EvalMetric.GRADE_STATUS, EvalMetric.GRADE_SCORE},
        )
        self.assertEqual(
            {change.case_id for change in assessment.comparison.regressions},
            {"release.beta"},
        )
        self.assertEqual(candidate.report.summary.failing_grades, 1)

    def test_blocked_candidate_is_inconclusive(self) -> None:
        approved = asyncio.run(
            run_scripted_report(version="1", outputs=APPROVED_OUTPUTS)
        )
        candidate = asyncio.run(
            run_scripted_report(
                version="2",
                outputs=APPROVED_OUTPUTS,
                blocked_release_ids=frozenset({"beta"}),
            )
        )

        assessment = assess_candidate(
            baseline=approve_baseline(approved.report),
            candidate=candidate.report,
        )

        self.assertEqual(assessment.decision, PromotionDecision.INCONCLUSIVE)
        self.assertEqual(candidate.report.summary.succeeded_cases, 1)
        self.assertEqual(candidate.report.summary.blocked_cases, 1)
        self.assertEqual(candidate.report.summary.passing_grades, 1)
        self.assertEqual(
            {
                item.reason
                for item in assessment.comparison.inconclusive
            },
            {EvalInconclusiveReason.BLOCKED_CASE},
        )

    def test_cases_use_isolated_sibling_contexts_and_safe_events(self) -> None:
        evaluated = asyncio.run(
            run_scripted_report(version="1", outputs=APPROVED_OUTPUTS)
        )

        self.assertEqual(
            tuple(case.run_id for case in evaluated.run.cases),
            (RunId("run-2"), RunId("run-4")),
        )
        self.assertEqual(
            tuple(event.kind for event in evaluated.events),
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
            tuple(call.context.run_id for call in evaluated.target.calls),
            (RunId("run-2"), RunId("run-4")),
        )
        self.assertEqual(
            tuple(call.input for call in evaluated.target.calls),
            tuple(case.input for case in RELEASE_DATASET.cases),
        )


if __name__ == "__main__":
    unittest.main()
