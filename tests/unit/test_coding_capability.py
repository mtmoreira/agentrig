from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    CapabilityRequirements,
    ChangedFileEvidence,
    CodingAgent,
    CodingChangeKind,
    CodingResult,
    CodingStatus,
    CodingTask,
    CodingValidationStatus,
    DataRetention,
    ValidationEvidence,
    WorkspaceAuthorization,
)
from agentrig.core import (
    ArtifactId,
    ArtifactRef,
    CancellationSource,
    Failure,
    FailureKind,
    RunContext,
    RunId,
)


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 14, 4, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


@dataclass
class SequentialRunIdGenerator:
    next_value: int = 1

    def generate(self) -> RunId:
        run_id = RunId(f"run-{self.next_value}")
        self.next_value += 1
        return run_id


def create_context() -> RunContext:
    return RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialRunIdGenerator(),
        cancellation=CancellationSource().token,
    )


def create_workspace(
    *,
    writable_roots: tuple[str, ...] = ("src", "tests"),
) -> WorkspaceAuthorization:
    return WorkspaceAuthorization(
        workspace_id="storyworld",
        root_path="/work/storyworld",
        writable_roots=writable_roots,
    )


def create_requirements() -> CapabilityRequirements:
    return CapabilityRequirements(
        kind=CapabilityKind.CODING,
        features=frozenset({CapabilityFeature.TOOL_USE}),
        allowed_data_retention=frozenset({DataRetention.NOT_RETAINED}),
    )


def create_task(
    *,
    workspace: WorkspaceAuthorization | None = None,
    requirements: CapabilityRequirements | None = None,
    max_changed_files: int = 2,
) -> CodingTask:
    return CodingTask(
        task_id="task-1",
        workspace=workspace if workspace is not None else create_workspace(),
        objective="  Implement the bounded change.\n",
        acceptance_criteria=(
            "The requested behavior is covered.",
            "The relevant validation passes.",
        ),
        max_changed_files=max_changed_files,
        requirements=(
            requirements if requirements is not None else create_requirements()
        ),
    )


def create_descriptor(
    *,
    features: frozenset[CapabilityFeature] | None = None,
    limits: dict[CapabilityLimit, int] | None = None,
) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id="example.coding",
        version="1",
        kind=CapabilityKind.CODING,
        features=(
            features
            if features is not None
            else frozenset({CapabilityFeature.TOOL_USE})
        ),
        limits=(
            limits
            if limits is not None
            else {CapabilityLimit.MAX_CHANGED_FILES: 4}
        ),
        data_retention=DataRetention.NOT_RETAINED,
    )


def create_artifact(artifact_id: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(artifact_id),
        kind="report",
        media_type="text/plain",
        producer_run_id=RunId("run-provider"),
        workspace_path=f"evidence/{artifact_id}.txt",
    )


def passing_validation(
    validation_id: str = "unit",
) -> ValidationEvidence:
    return ValidationEvidence(
        validation_id=validation_id,
        status=CodingValidationStatus.PASSED,
        summary="  Relevant tests passed.\n",
        exit_code=0,
        output_artifact=create_artifact(f"{validation_id}-log"),
    )


def blocked_failure() -> Failure:
    return Failure(
        kind=FailureKind.WORKFLOW_BLOCKED,
        message="external approval is required",
        code="coding.approval_required",
    )


@dataclass
class ScriptedCodingAgent:
    descriptor: CapabilityDescriptor
    changed_files: tuple[ChangedFileEvidence, ...]
    validations: tuple[ValidationEvidence, ...]
    blocker: Failure | None = None
    calls: list[tuple[CodingTask, RunContext]] = field(default_factory=list)

    async def execute(
        self,
        task: CodingTask,
        context: RunContext,
    ) -> CodingResult:
        task.require_supported_by(self.descriptor)
        context.cancellation.raise_if_cancelled()
        if context.deadline is not None:
            context.deadline.raise_if_expired(context.clock)
        self.calls.append((task, context))
        if self.blocker is not None:
            return CodingResult.blocked(
                task=task,
                blocker=self.blocker,
                changed_files=self.changed_files,
                validations=self.validations,
            )
        return CodingResult.succeeded(
            task=task,
            changed_files=self.changed_files,
            validations=self.validations,
        )


async def execute_typed(
    agent: CodingAgent,
    task: CodingTask,
    context: RunContext,
) -> CodingResult:
    return await agent.execute(task, context)


class WorkspaceAuthorizationTest(unittest.TestCase):
    def test_enforces_canonical_explicit_writable_scopes(self) -> None:
        workspace = create_workspace()

        self.assertTrue(workspace.permits("src/story.py"))
        self.assertTrue(workspace.permits("tests"))
        self.assertFalse(workspace.permits("docs/plan.md"))
        self.assertNotIn("/work/storyworld", repr(workspace))
        self.assertNotIn("writable_roots", repr(workspace))

        whole_workspace = create_workspace(writable_roots=(".",))
        self.assertTrue(whole_workspace.permits("docs/plan.md"))

    def test_rejects_broad_or_noncanonical_authorization(self) -> None:
        invalid_roots = (
            "/",
            "relative/workspace",
            "/work/../storyworld",
            "/work//storyworld",
            "/work/storyworld/",
            "//server/workspace",
            "/work\\storyworld",
        )
        for root_path in invalid_roots:
            with self.subTest(root_path=root_path):
                with self.assertRaises(ValueError):
                    WorkspaceAuthorization(
                        workspace_id="workspace",
                        root_path=root_path,
                        writable_roots=("src",),
                    )

        invalid_scopes = (
            (),
            ("../src",),
            ("/src",),
            ("src/../tests",),
            ("src/",),
            ("src", "src"),
            (".", "src"),
        )
        for writable_roots in invalid_scopes:
            with self.subTest(writable_roots=writable_roots):
                with self.assertRaises(ValueError):
                    create_workspace(writable_roots=writable_roots)

        with self.assertRaises(ValueError):
            create_workspace().permits("../outside.txt")


class CodingTaskTest(unittest.TestCase):
    def test_preserves_private_content_and_preflights_requirements(self) -> None:
        criteria = [
            "The requested behavior is covered.",
            "The relevant validation passes.",
        ]
        task = CodingTask(
            task_id="task-1",
            workspace=create_workspace(),
            objective="  Implement the bounded change.\n",
            acceptance_criteria=criteria,  # type: ignore[arg-type]
            max_changed_files=2,
            requirements=create_requirements(),
        )
        criteria.clear()

        self.assertEqual(task.objective, "  Implement the bounded change.\n")
        self.assertEqual(len(task.acceptance_criteria), 2)
        self.assertNotIn("Implement the bounded change", repr(task))
        self.assertNotIn("requested behavior", repr(task))
        self.assertEqual(
            task.capability_requirements.minimum_limits,
            {CapabilityLimit.MAX_CHANGED_FILES: 2},
        )
        task.require_supported_by(create_descriptor())

        with self.assertRaisesRegex(ValueError, "tool_use"):
            task.require_supported_by(
                create_descriptor(features=frozenset())
            )
        with self.assertRaisesRegex(ValueError, "max_changed_files"):
            task.require_supported_by(
                create_descriptor(
                    limits={CapabilityLimit.MAX_CHANGED_FILES: 1}
                )
            )

    def test_rejects_invalid_identity_workspace_objective_and_criteria(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            CodingTask(
                task_id=" padded ",
                workspace=create_workspace(),
                objective="Implement change",
                acceptance_criteria=("Pass tests",),
                max_changed_files=1,
            )
        with self.assertRaises(TypeError):
            CodingTask(
                task_id="task",
                workspace="invalid",  # type: ignore[arg-type]
                objective="Implement change",
                acceptance_criteria=("Pass tests",),
                max_changed_files=1,
            )
        for criteria in (
            (),
            (" ",),
            ("Pass tests", "Pass tests"),
        ):
            with self.subTest(criteria=criteria):
                with self.assertRaises(ValueError):
                    CodingTask(
                        task_id="task",
                        workspace=create_workspace(),
                        objective="Implement change",
                        acceptance_criteria=criteria,
                        max_changed_files=1,
                    )
        with self.assertRaises(ValueError):
            CodingTask(
                task_id="task",
                workspace=create_workspace(),
                objective=" ",
                acceptance_criteria=("Pass tests",),
                max_changed_files=1,
            )
        with self.assertRaises(ValueError):
            CodingTask(
                task_id="task",
                workspace=create_workspace(),
                objective="Implement change",
                acceptance_criteria=("Pass tests",),
                max_changed_files=1,
                requirements=CapabilityRequirements(
                    kind=CapabilityKind.SEARCH,
                ),
            )
        with self.assertRaises(ValueError):
            CodingTask(
                task_id="task",
                workspace=create_workspace(),
                objective="Implement change",
                acceptance_criteria=("Pass tests",),
                max_changed_files=0,
            )
        with self.assertRaises(ValueError):
            CodingTask(
                task_id="task",
                workspace=create_workspace(),
                objective="Implement change",
                acceptance_criteria=("Pass tests",),
                max_changed_files=1,
                requirements=CapabilityRequirements(
                    kind=CapabilityKind.CODING,
                    minimum_limits={
                        CapabilityLimit.MAX_CHANGED_FILES: 1,
                    },
                ),
            )


class CodingEvidenceTest(unittest.TestCase):
    def test_coding_vocabularies_have_stable_wire_values(self) -> None:
        self.assertEqual(
            tuple(item.value for item in CodingChangeKind),
            ("added", "modified", "deleted"),
        )
        self.assertEqual(
            tuple(item.value for item in CodingValidationStatus),
            ("passed", "failed", "not_run"),
        )
        self.assertEqual(
            tuple(item.value for item in CodingStatus),
            ("succeeded", "blocked"),
        )

    def test_changed_file_and_validation_evidence_are_sanitized(self) -> None:
        change = ChangedFileEvidence(
            path="src/story.py",
            change_kind=CodingChangeKind.MODIFIED,
            evidence_artifact=create_artifact("patch"),
        )
        validation = passing_validation()

        self.assertEqual(change.path, "src/story.py")
        self.assertEqual(change.change_kind, CodingChangeKind.MODIFIED)
        self.assertEqual(validation.summary, "  Relevant tests passed.\n")
        self.assertNotIn("Relevant tests passed", repr(validation))
        self.assertNotIn("unit-log", repr(validation))

    def test_validation_status_and_exit_code_must_agree(self) -> None:
        invalid_values = (
            (CodingValidationStatus.NOT_RUN, 1),
            (CodingValidationStatus.PASSED, 1),
            (CodingValidationStatus.FAILED, 0),
            (CodingValidationStatus.FAILED, -1),
        )
        for status, exit_code in invalid_values:
            with self.subTest(status=status, exit_code=exit_code):
                with self.assertRaises(ValueError):
                    ValidationEvidence(
                        validation_id="unit",
                        status=status,
                        summary="Validation evidence.",
                        exit_code=exit_code,
                    )
        with self.assertRaises(TypeError):
            ValidationEvidence(
                validation_id="unit",
                status="passed",  # type: ignore[arg-type]
                summary="Validation evidence.",
            )
        with self.assertRaises(ValueError):
            ChangedFileEvidence(
                path="../outside.py",
                change_kind=CodingChangeKind.MODIFIED,
            )


class CodingResultTest(unittest.TestCase):
    def test_success_requires_authorized_changes_and_passing_evidence(self) -> None:
        task = create_task()
        changes = [
            ChangedFileEvidence(
                path="src/story.py",
                change_kind=CodingChangeKind.MODIFIED,
            )
        ]
        validations = [passing_validation()]

        result = CodingResult.succeeded(
            task=task,
            changed_files=changes,
            validations=validations,
        )
        changes.clear()
        validations.clear()

        self.assertEqual(result.status, CodingStatus.SUCCEEDED)
        self.assertTrue(result.is_success)
        self.assertEqual(result.task_id, "task-1")
        self.assertEqual(result.workspace_id, "storyworld")
        self.assertEqual(len(result.changed_files), 1)
        self.assertEqual(len(result.validations), 1)

    def test_blocked_result_preserves_partial_evidence_and_failure(self) -> None:
        failure = blocked_failure()
        failed_validation = ValidationEvidence(
            validation_id="unit",
            status=CodingValidationStatus.FAILED,
            summary="Tests expose an external blocker.",
            exit_code=1,
        )

        result = CodingResult.blocked(
            task=create_task(),
            blocker=failure,
            changed_files=(
                ChangedFileEvidence(
                    path="src/story.py",
                    change_kind=CodingChangeKind.MODIFIED,
                ),
            ),
            validations=(failed_validation,),
        )

        self.assertEqual(result.status, CodingStatus.BLOCKED)
        self.assertFalse(result.is_success)
        self.assertIs(result.blocker, failure)
        self.assertEqual(result.validations, (failed_validation,))
        self.assertNotIn(failure.message, repr(result))

    def test_rejects_impossible_unauthorized_or_duplicate_results(self) -> None:
        task = create_task()
        passed = passing_validation()
        failed = ValidationEvidence(
            validation_id="failed",
            status=CodingValidationStatus.FAILED,
            summary="Validation failed.",
            exit_code=1,
        )
        with self.assertRaises(ValueError):
            CodingResult.succeeded(task=task, validations=())
        with self.assertRaises(ValueError):
            CodingResult.succeeded(task=task, validations=(failed,))
        with self.assertRaises(ValueError):
            CodingResult(
                task=task,
                status=CodingStatus.SUCCEEDED,
                validations=(passed,),
                blocker=blocked_failure(),
            )
        with self.assertRaises(ValueError):
            CodingResult(
                task=task,
                status=CodingStatus.BLOCKED,
            )
        with self.assertRaises(ValueError):
            CodingResult.blocked(
                task=task,
                blocker=Failure(
                    kind=FailureKind.UNEXPECTED,
                    message="unexpected implementation failure",
                ),
            )
        with self.assertRaises(ValueError):
            CodingResult.succeeded(
                task=task,
                changed_files=(
                    ChangedFileEvidence(
                        path="docs/plan.md",
                        change_kind=CodingChangeKind.MODIFIED,
                    ),
                ),
                validations=(passed,),
            )
        repeated = ChangedFileEvidence(
            path="src/story.py",
            change_kind=CodingChangeKind.MODIFIED,
        )
        with self.assertRaises(ValueError):
            CodingResult.succeeded(
                task=task,
                changed_files=(repeated, repeated),
                validations=(passed,),
            )
        with self.assertRaises(ValueError):
            CodingResult.succeeded(
                task=task,
                validations=(passed, passed),
            )
        with self.assertRaises(ValueError):
            CodingResult.succeeded(
                task=task,
                changed_files=(
                    ChangedFileEvidence(
                        path="src/one.py",
                        change_kind=CodingChangeKind.ADDED,
                    ),
                    ChangedFileEvidence(
                        path="src/two.py",
                        change_kind=CodingChangeKind.ADDED,
                    ),
                    ChangedFileEvidence(
                        path="src/three.py",
                        change_kind=CodingChangeKind.ADDED,
                    ),
                ),
                validations=(passed,),
            )


class CodingAgentContractTest(unittest.TestCase):
    def test_protocol_executes_one_typed_authorized_task(self) -> None:
        agent = ScriptedCodingAgent(
            descriptor=create_descriptor(),
            changed_files=(
                ChangedFileEvidence(
                    path="src/story.py",
                    change_kind=CodingChangeKind.MODIFIED,
                ),
            ),
            validations=(passing_validation(),),
        )
        task = create_task()
        context = create_context()

        result = asyncio.run(execute_typed(agent, task, context))

        self.assertIsInstance(agent, CodingAgent)
        self.assertTrue(result.is_success)
        self.assertEqual(agent.calls, [(task, context)])

    def test_preflight_failure_does_not_record_or_execute_task(self) -> None:
        agent = ScriptedCodingAgent(
            descriptor=create_descriptor(features=frozenset()),
            changed_files=(),
            validations=(passing_validation(),),
        )

        with self.assertRaisesRegex(ValueError, "tool_use"):
            asyncio.run(agent.execute(create_task(), create_context()))

        self.assertEqual(agent.calls, [])


if __name__ == "__main__":
    unittest.main()
