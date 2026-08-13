from __future__ import annotations

import unittest

from agentrig.core import (
    AgentRigError,
    ArtifactId,
    ArtifactRef,
    ExecutionOutcome,
    ExecutionStatus,
    Failure,
    FailureKind,
    RunId,
)


def create_artifact(artifact_id: str = "artifact-1") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(artifact_id),
        kind="report",
        media_type="text/plain",
        producer_run_id=RunId("run-1"),
        workspace_path=f"outputs/{artifact_id}.txt",
    )


def create_failure(kind: FailureKind) -> Failure:
    return Failure(kind=kind, message=f"safe {kind.value} message")


def unwrap_text(outcome: ExecutionOutcome[str]) -> str:
    return outcome.unwrap()


class ExecutionStatusTest(unittest.TestCase):
    def test_vocabulary_has_stable_wire_values(self) -> None:
        self.assertEqual(
            {status.value for status in ExecutionStatus},
            {"blocked", "cancelled", "failed", "succeeded"},
        )


class ExecutionOutcomeTest(unittest.TestCase):
    def test_success_preserves_typed_output_and_artifacts(self) -> None:
        artifacts = [create_artifact()]

        outcome = ExecutionOutcome.succeeded(
            "draft complete",
            artifacts=artifacts,
        )
        artifacts.append(create_artifact("artifact-2"))

        self.assertEqual(outcome.status, ExecutionStatus.SUCCEEDED)
        self.assertTrue(outcome.is_success)
        self.assertIsNone(outcome.failure)
        self.assertEqual(unwrap_text(outcome), "draft complete")
        self.assertEqual(
            tuple(artifact.artifact_id for artifact in outcome.artifacts),
            (ArtifactId("artifact-1"),),
        )

    def test_none_is_a_valid_successful_output(self) -> None:
        outcome = ExecutionOutcome.succeeded(None)

        self.assertTrue(outcome.is_success)
        self.assertIsNone(outcome.unwrap())

    def test_failure_categories_map_to_unambiguous_statuses(self) -> None:
        for kind in FailureKind:
            with self.subTest(kind=kind):
                outcome = ExecutionOutcome.from_failure(create_failure(kind))
                if kind is FailureKind.CANCELLED:
                    expected = ExecutionStatus.CANCELLED
                elif kind in (
                    FailureKind.APPROVAL_REQUIRED,
                    FailureKind.WORKFLOW_BLOCKED,
                ):
                    expected = ExecutionStatus.BLOCKED
                else:
                    expected = ExecutionStatus.FAILED

                self.assertEqual(outcome.status, expected)
                self.assertFalse(outcome.is_success)

    def test_non_successful_unwrap_raises_normalized_error(self) -> None:
        failure = create_failure(FailureKind.PERMANENT_PROVIDER)
        outcome: ExecutionOutcome[str] = ExecutionOutcome.from_failure(
            failure,
            artifacts=(create_artifact(),),
        )

        with self.assertRaises(AgentRigError) as raised:
            outcome.unwrap()

        self.assertIs(raised.exception.failure, failure)
        self.assertEqual(outcome.artifacts, (create_artifact(),))

    def test_direct_construction_rejects_impossible_states(self) -> None:
        failure = create_failure(FailureKind.UNEXPECTED)
        invalid_values = (
            {
                "status": ExecutionStatus.SUCCEEDED,
                "output": "value",
                "failure": failure,
            },
            {"status": ExecutionStatus.FAILED},
            {
                "status": ExecutionStatus.FAILED,
                "output": "partial",
                "failure": failure,
            },
            {
                "status": ExecutionStatus.BLOCKED,
                "failure": failure,
            },
            {
                "status": ExecutionStatus.CANCELLED,
                "failure": failure,
            },
        )
        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    ExecutionOutcome(**values)  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            ExecutionOutcome(status="failed")  # type: ignore[arg-type]

    def test_from_failure_requires_a_normalized_failure(self) -> None:
        with self.assertRaises(TypeError):
            ExecutionOutcome.from_failure(  # type: ignore[arg-type]
                RuntimeError("raw exception")
            )

    def test_artifacts_require_refs_with_unique_ids(self) -> None:
        artifact = create_artifact()
        with self.assertRaises(ValueError):
            ExecutionOutcome.succeeded(
                "value",
                artifacts=(artifact, artifact),
            )
        with self.assertRaises(TypeError):
            ExecutionOutcome.succeeded(
                "value",
                artifacts=("not-an-artifact",),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
