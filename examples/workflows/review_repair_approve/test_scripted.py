from __future__ import annotations

import asyncio
import unittest

from agentrig.core import EventKind, ExecutionStatus, FailureKind, GradeStatus
from agentrig.workflow import ApprovalDecision

from examples.workflows.review_repair_approve.scripted import (
    run_scripted_example,
)


class ReviewRepairApproveExampleTest(unittest.TestCase):
    def test_repairs_then_approves_and_publishes(self) -> None:
        run = asyncio.run(run_scripted_example())

        self.assertEqual(run.outcome.status, ExecutionStatus.SUCCEEDED)
        result = run.outcome.unwrap()
        self.assertEqual(result.candidate.draft.revision, 1)
        self.assertIn("Rollback:", result.candidate.draft.body)
        self.assertEqual(result.candidate.repairs, 1)
        self.assertEqual(result.candidate.grades[0].status, GradeStatus.PASS)
        self.assertEqual(
            result.approval.decision,
            ApprovalDecision.APPROVED,
        )
        self.assertEqual(
            result.publication.destination,
            "releases/feature-launch.md",
        )
        self.assertEqual(run.grader_calls, 2)
        self.assertEqual(run.approval_calls, 1)
        self.assertEqual(run.publisher_calls, 1)

        event_kinds = tuple(event.kind for event in run.events)
        self.assertEqual(event_kinds.count(EventKind.GRADE_PRODUCED), 2)
        self.assertIn(EventKind.PROGRESS_REPORTED, event_kinds)
        self.assertLess(
            event_kinds.index(EventKind.APPROVAL_REQUESTED),
            event_kinds.index(EventKind.APPROVAL_RESOLVED),
        )

    def test_denial_never_invokes_the_publisher(self) -> None:
        run = asyncio.run(
            run_scripted_example(approval=ApprovalDecision.DENIED)
        )

        self.assertEqual(run.outcome.status, ExecutionStatus.FAILED)
        self.assertIsNotNone(run.outcome.failure)
        if run.outcome.failure is None:
            raise AssertionError("failed outcome has no failure")
        self.assertEqual(
            run.outcome.failure.kind,
            FailureKind.APPROVAL_DENIED,
        )
        self.assertEqual(run.outcome.failure.code, "approval.denied")
        self.assertEqual(run.grader_calls, 2)
        self.assertEqual(run.approval_calls, 1)
        self.assertEqual(run.publisher_calls, 0)
        self.assertIn(
            EventKind.APPROVAL_RESOLVED,
            tuple(event.kind for event in run.events),
        )


if __name__ == "__main__":
    unittest.main()
