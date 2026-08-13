from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agentrig.core import (
    AgentRigError,
    CancellationSource,
    Deadline,
    EffectProfile,
    EventId,
    EventKind,
    ExecutionStatus,
    Failure,
    FailureKind,
    InMemoryEventSink,
    NoOpRedactionPolicy,
    RunContext,
    RunId,
)
from agentrig.workflow import (
    ApprovalAuthority,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResolution,
    ApprovalStep,
    ApprovalStepResult,
    Step,
    StepDescriptor,
    execute_step,
)


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 14, 2, 0, tzinfo=UTC)

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
    deadline: Deadline | None = None,
) -> tuple[RunContext, InMemoryEventSink]:
    owned_source = source if source is not None else CancellationSource()
    sink = InMemoryEventSink(redaction_policy=NoOpRedactionPolicy())
    context = RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialRunIdGenerator(),
        cancellation=owned_source.token,
        event_sink=sink,
        event_id_generator=SequentialEventIdGenerator(),
        deadline=deadline,
        correlation={"request_id": "request-1"},
    )
    return context, sink


def action_descriptor() -> StepDescriptor:
    return StepDescriptor(
        step_id="workspace.write",
        version="2",
        effect_profile=EffectProfile.NON_REPEATABLE,
    )


@dataclass
class RecordingAction:
    descriptor: StepDescriptor = field(default_factory=action_descriptor)
    output: str = "written"
    failure: Failure | None = None
    calls: list[tuple[str, RunContext]] = field(default_factory=list)

    async def run(self, input: str, context: RunContext) -> str:
        self.calls.append((input, context))
        if self.failure is not None:
            raise AgentRigError(self.failure)
        return self.output


@dataclass
class RecordingAuthority:
    resolution: ApprovalResolution | None = None
    failure: Failure | None = None
    calls: list[tuple[ApprovalRequest[str], RunContext]] = field(
        default_factory=list
    )

    async def resolve(
        self,
        request: ApprovalRequest[str],
        context: RunContext,
    ) -> ApprovalResolution:
        self.calls.append((request, context))
        if self.failure is not None:
            raise AgentRigError(self.failure)
        if self.resolution is None:
            raise AssertionError("recording authority has no resolution")
        return self.resolution


def create_resolution(
    decision: ApprovalDecision = ApprovalDecision.APPROVED,
    *,
    approval_id: str = "approval-1",
) -> ApprovalResolution:
    return ApprovalResolution(
        approval_id=approval_id,
        decision=decision,
        resolver="reviewer-private",
        reason="private operational reason",
    )


def create_request(
    action: StepDescriptor | None = None,
) -> ApprovalRequest[str]:
    return ApprovalRequest(
        approval_id="approval-1",
        action=action if action is not None else action_descriptor(),
        summary="Write password=private into the workspace",
        proposed_input="private prepared payload",
    )


class ApprovalStepTest(unittest.TestCase):
    def test_approved_request_runs_the_scoped_action_and_records_resolution(
        self,
    ) -> None:
        action = RecordingAction()
        authority = RecordingAuthority(resolution=create_resolution())
        step = ApprovalStep[str, str](action=action, authority=authority)
        request = create_request(action.descriptor)
        context, sink = create_context()

        outcome = asyncio.run(execute_step(step, request, context))

        self.assertIsInstance(step, Step)
        self.assertIsInstance(authority, ApprovalAuthority)
        self.assertEqual(
            step.descriptor,
            StepDescriptor(
                step_id="workspace.write.approval",
                version="2",
                effect_profile=EffectProfile.NON_REPEATABLE,
            ),
        )
        self.assertEqual(outcome.status, ExecutionStatus.SUCCEEDED)
        result = outcome.unwrap()
        self.assertIs(result.request, request)
        self.assertIs(result.resolution, authority.resolution)
        self.assertEqual(result.output, "written")
        self.assertEqual(authority.calls, [(request, context)])
        self.assertEqual(len(action.calls), 1)
        self.assertEqual(action.calls[0][0], "private prepared payload")
        self.assertEqual(action.calls[0][1].run_id, RunId("run-2"))
        self.assertEqual(action.calls[0][1].parent_run_id, RunId("run-1"))
        self.assertEqual(
            action.calls[0][1].correlation["approval_id"],
            "approval-1",
        )
        self.assertEqual(
            [event.kind for event in sink.events],
            [
                EventKind.STEP_STARTED,
                EventKind.APPROVAL_REQUESTED,
                EventKind.APPROVAL_RESOLVED,
                EventKind.STEP_STARTED,
                EventKind.STEP_COMPLETED,
                EventKind.STEP_COMPLETED,
            ],
        )
        self.assertEqual(
            sink.events[1].attributes,
            {
                "approval_id": "approval-1",
                "action_id": "workspace.write",
                "action_version": "2",
                "effect_profile": "non_repeatable",
            },
        )
        self.assertEqual(sink.events[2].attributes["decision"], "approved")
        self.assertNotIn("summary", sink.events[1].attributes)
        self.assertNotIn("resolver", sink.events[2].attributes)
        self.assertNotIn("reason", sink.events[2].attributes)
        self.assertNotIn("private", repr(sink.events))

    def test_denial_is_terminal_and_never_invokes_the_action(self) -> None:
        action = RecordingAction()
        authority = RecordingAuthority(
            resolution=create_resolution(ApprovalDecision.DENIED)
        )
        step = ApprovalStep[str, str](action=action, authority=authority)
        request = create_request(action.descriptor)
        context, sink = create_context()

        outcome = asyncio.run(execute_step(step, request, context))

        self.assertEqual(outcome.status, ExecutionStatus.FAILED)
        self.assertIsNotNone(outcome.failure)
        if outcome.failure is None:
            raise AssertionError("denied outcome has no failure")
        self.assertEqual(outcome.failure.kind, FailureKind.APPROVAL_DENIED)
        self.assertEqual(outcome.failure.code, "approval.denied")
        self.assertEqual(
            outcome.failure.metadata,
            {
                "approval_id": "approval-1",
                "action_id": "workspace.write",
                "action_version": "2",
            },
        )
        self.assertEqual(action.calls, [])
        self.assertEqual(
            [event.kind for event in sink.events],
            [
                EventKind.STEP_STARTED,
                EventKind.APPROVAL_REQUESTED,
                EventKind.APPROVAL_RESOLVED,
                EventKind.STEP_COMPLETED,
            ],
        )
        self.assertEqual(sink.events[2].attributes["decision"], "denied")

    def test_unresolved_request_preserves_normalized_blocking_failure(self) -> None:
        action = RecordingAction()
        failure = Failure(
            kind=FailureKind.APPROVAL_REQUIRED,
            message="approval is pending",
            code="approval.pending",
            metadata={"approval_id": "approval-1"},
        )
        authority = RecordingAuthority(failure=failure)
        step = ApprovalStep[str, str](action=action, authority=authority)
        request = create_request(action.descriptor)
        context, sink = create_context()

        outcome = asyncio.run(execute_step(step, request, context))

        self.assertEqual(outcome.status, ExecutionStatus.BLOCKED)
        self.assertIs(outcome.failure, failure)
        self.assertEqual(action.calls, [])
        self.assertEqual(
            [event.kind for event in sink.events],
            [
                EventKind.STEP_STARTED,
                EventKind.APPROVAL_REQUESTED,
                EventKind.STEP_COMPLETED,
            ],
        )

    def test_constraints_prevent_resolution_or_action_execution(self) -> None:
        source = CancellationSource()
        source.cancel("caller stopped")
        action = RecordingAction()
        authority = RecordingAuthority(resolution=create_resolution())
        step = ApprovalStep[str, str](action=action, authority=authority)
        request = create_request(action.descriptor)
        context, sink = create_context(source=source)

        cancelled = asyncio.run(execute_step(step, request, context))

        self.assertEqual(cancelled.status, ExecutionStatus.CANCELLED)
        self.assertEqual(authority.calls, [])
        self.assertEqual(action.calls, [])
        self.assertEqual(
            [event.kind for event in sink.events],
            [EventKind.STEP_STARTED, EventKind.STEP_COMPLETED],
        )

        deadline = Deadline(
            expires_at=datetime(2026, 8, 14, 2, 0, tzinfo=UTC),
            monotonic_deadline=100.0,
        )
        action = RecordingAction()
        authority = RecordingAuthority(resolution=create_resolution())
        step = ApprovalStep[str, str](action=action, authority=authority)
        request = create_request(action.descriptor)
        context, sink = create_context(deadline=deadline)

        expired = asyncio.run(execute_step(step, request, context))

        self.assertEqual(expired.status, ExecutionStatus.FAILED)
        self.assertIsNotNone(expired.failure)
        if expired.failure is None:
            raise AssertionError("expired outcome has no failure")
        self.assertEqual(expired.failure.kind, FailureKind.DEADLINE_EXCEEDED)
        self.assertEqual(authority.calls, [])
        self.assertEqual(action.calls, [])
        self.assertEqual(
            [event.kind for event in sink.events],
            [EventKind.STEP_STARTED, EventKind.STEP_COMPLETED],
        )

        post_source = CancellationSource()

        @dataclass
        class CancellingAuthority:
            async def resolve(
                self,
                request: ApprovalRequest[str],
                context: RunContext,
            ) -> ApprovalResolution:
                del context
                post_source.cancel("stopped after approval")
                return create_resolution(approval_id=request.approval_id)

        action = RecordingAction()
        step = ApprovalStep[str, str](
            action=action,
            authority=CancellingAuthority(),
        )
        request = create_request(action.descriptor)
        context, sink = create_context(source=post_source)

        post_cancelled = asyncio.run(execute_step(step, request, context))

        self.assertEqual(post_cancelled.status, ExecutionStatus.CANCELLED)
        self.assertEqual(action.calls, [])
        self.assertEqual(
            [event.kind for event in sink.events],
            [
                EventKind.STEP_STARTED,
                EventKind.APPROVAL_REQUESTED,
                EventKind.APPROVAL_RESOLVED,
                EventKind.STEP_COMPLETED,
            ],
        )

    def test_mismatched_request_is_rejected_before_requesting_approval(self) -> None:
        action = RecordingAction()
        authority = RecordingAuthority(resolution=create_resolution())
        step = ApprovalStep[str, str](action=action, authority=authority)
        mismatched = create_request(
            StepDescriptor(
                step_id="workspace.delete",
                version="1",
                effect_profile=EffectProfile.NON_REPEATABLE,
            )
        )
        context, sink = create_context()

        outcome = asyncio.run(execute_step(step, mismatched, context))

        self.assertEqual(outcome.status, ExecutionStatus.FAILED)
        self.assertIsNotNone(outcome.failure)
        if outcome.failure is None:
            raise AssertionError("mismatched outcome has no failure")
        self.assertEqual(outcome.failure.kind, FailureKind.INVALID_INPUT)
        self.assertEqual(outcome.failure.code, "approval.action_mismatch")
        self.assertEqual(authority.calls, [])
        self.assertEqual(action.calls, [])
        self.assertEqual(
            [event.kind for event in sink.events],
            [EventKind.STEP_STARTED, EventKind.STEP_COMPLETED],
        )

    def test_sanitizes_invalid_authority_and_preserves_action_failure(self) -> None:
        @dataclass(frozen=True)
        class BrokenAuthority:
            async def resolve(
                self,
                request: ApprovalRequest[str],
                context: RunContext,
            ) -> ApprovalResolution:
                del request, context
                raise RuntimeError("password=private")

        action = RecordingAction()
        step = ApprovalStep[str, str](
            action=action,
            authority=BrokenAuthority(),
        )
        context, _ = create_context()

        broken = asyncio.run(
            execute_step(step, create_request(action.descriptor), context)
        )

        self.assertIsNotNone(broken.failure)
        if broken.failure is None:
            raise AssertionError("broken outcome has no failure")
        self.assertEqual(broken.failure.kind, FailureKind.UNEXPECTED)
        self.assertEqual(broken.failure.code, "approval.resolution_failed")
        self.assertNotIn("private", broken.failure.message)
        self.assertEqual(action.calls, [])

        @dataclass(frozen=True)
        class InvalidAuthority:
            async def resolve(
                self,
                request: ApprovalRequest[str],
                context: RunContext,
            ) -> object:
                del request, context
                return "approved"

        step = ApprovalStep[str, str](
            action=action,
            authority=InvalidAuthority(),  # type: ignore[arg-type]
        )
        context, _ = create_context()
        invalid = asyncio.run(
            execute_step(step, create_request(action.descriptor), context)
        )

        self.assertIsNotNone(invalid.failure)
        if invalid.failure is None:
            raise AssertionError("invalid outcome has no failure")
        self.assertEqual(invalid.failure.code, "approval.invalid_resolution")
        self.assertEqual(action.calls, [])

        action_failure = Failure(
            kind=FailureKind.PERMANENT_PROVIDER,
            message="protected action failed",
            code="action.failed",
        )
        failing_action = RecordingAction(failure=action_failure)
        step = ApprovalStep[str, str](
            action=failing_action,
            authority=RecordingAuthority(resolution=create_resolution()),
        )
        context, _ = create_context()
        failed_action = asyncio.run(
            execute_step(
                step,
                create_request(failing_action.descriptor),
                context,
            )
        )

        self.assertIs(failed_action.failure, action_failure)
        self.assertEqual(len(failing_action.calls), 1)

    def test_validates_request_resolution_configuration_and_result(self) -> None:
        action = RecordingAction()
        authority = RecordingAuthority(resolution=create_resolution())
        request = create_request(action.descriptor)
        self.assertNotIn("private prepared payload", repr(request))

        with self.assertRaises((TypeError, ValueError)):
            ApprovalRequest(
                approval_id=" ",
                action=action.descriptor,
                summary="Write",
                proposed_input="draft",
            )
        with self.assertRaises(TypeError):
            ApprovalResolution(
                approval_id="approval-1",
                decision="approved",  # type: ignore[arg-type]
                resolver="reviewer",
            )
        with self.assertRaises(TypeError):
            ApprovalStep[str, str](
                action="invalid",  # type: ignore[arg-type]
                authority=authority,
            )
        with self.assertRaises(TypeError):
            ApprovalStep[str, str](
                action=action,
                authority="invalid",  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            asyncio.run(
                ApprovalStep[str, str](
                    action=action,
                    authority=authority,
                ).run("invalid", create_context()[0])  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            ApprovalStepResult(
                request=request,
                resolution=create_resolution(ApprovalDecision.DENIED),
                output="unreachable",
            )


if __name__ == "__main__":
    unittest.main()
