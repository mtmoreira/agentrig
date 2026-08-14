"""Deterministic executable for the review, repair, and approval example."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agentrig.core import (
    CancellationSource,
    EffectProfile,
    Event,
    EventId,
    ExecutionOutcome,
    Grade,
    GradeClassification,
    GradeDecision,
    GradeEvidence,
    GradePolicyDescriptor,
    GradeReference,
    GradeStatus,
    GradeThreshold,
    GraderDescriptor,
    InMemoryEventSink,
    RunContext,
    RunId,
    ScoreRange,
    ThresholdGradePolicy,
)
from agentrig.testing import ScriptedGrader
from agentrig.workflow import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResolution,
    FunctionStep,
    RepairRequest,
    Step,
    StepDescriptor,
    Workflow,
)

from examples.workflows.review_repair_approve.workflow import (
    Draft,
    Publication,
    ReleaseCandidate,
    ReleaseResult,
    build_release_workflow,
)

READINESS = GraderDescriptor(
    grader_id="release.readiness",
    version="1",
)
READINESS_METRIC = "rollback_documented"


@dataclass(frozen=True, slots=True)
class FixedClock:
    """Keep example event timestamps reproducible."""

    def now(self) -> datetime:
        return datetime(2026, 8, 13, 12, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


@dataclass(slots=True)
class SequentialRunIdGenerator:
    next_value: int = 1

    def generate(self) -> RunId:
        run_id = RunId(f"run-{self.next_value}")
        self.next_value += 1
        return run_id


@dataclass(slots=True)
class SequentialEventIdGenerator:
    next_value: int = 1

    def generate(self) -> EventId:
        event_id = EventId(f"event-{self.next_value}")
        self.next_value += 1
        return event_id


@dataclass(slots=True)
class ScriptedApprovalAuthority:
    """Resolve the example request with one configured deterministic decision."""

    decision: ApprovalDecision = ApprovalDecision.APPROVED
    calls: list[ApprovalRequest[ReleaseCandidate]] = field(default_factory=list)

    async def resolve(
        self,
        request: ApprovalRequest[ReleaseCandidate],
        context: RunContext,
    ) -> ApprovalResolution:
        del context
        self.calls.append(request)
        return ApprovalResolution(
            approval_id=request.approval_id,
            decision=self.decision,
            resolver="scripted-release-reviewer",
        )


@dataclass(slots=True)
class ScriptedPublisher:
    """Record publication only when the approval boundary invokes it."""

    calls: list[ReleaseCandidate] = field(default_factory=list)

    async def publish(
        self,
        candidate: ReleaseCandidate,
        context: RunContext,
    ) -> Publication:
        del context
        self.calls.append(candidate)
        return Publication(
            destination=(
                f"releases/{candidate.draft.title.lower().replace(' ', '-')}.md"
            ),
            revision=candidate.draft.revision,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedExample:
    workflow: Workflow[Draft, ReleaseResult]
    grader: ScriptedGrader[Draft]
    authority: ScriptedApprovalAuthority
    publisher: ScriptedPublisher


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedRun:
    outcome: ExecutionOutcome[ReleaseResult]
    events: tuple[Event, ...]
    grader_calls: int
    approval_calls: int
    publisher_calls: int


def _grade(*, status: GradeStatus, score: float) -> Grade:
    return Grade(
        grader=READINESS,
        metric=READINESS_METRIC,
        status=status,
        classification=GradeClassification.HARD,
        explanation="A releasable draft documents how to roll back the change.",
        evidence=(GradeEvidence(field_path=("body",)),),
        score=score,
        score_range=ScoreRange(minimum=0, maximum=1),
    )


def _release_policy() -> ThresholdGradePolicy:
    return ThresholdGradePolicy(
        descriptor=GradePolicyDescriptor(
            policy_id="release.review",
            version="1",
        ),
        thresholds=(
            GradeThreshold(
                grade=GradeReference(
                    grader_id=READINESS.grader_id,
                    grader_version=READINESS.version,
                    metric=READINESS_METRIC,
                ),
                minimum_score=0.8,
                failure_decision=GradeDecision.REPAIR,
            ),
        ),
        hard_failure_decision=GradeDecision.REPAIR,
        soft_failure_decision=GradeDecision.REPAIR,
    )


async def _repair_draft(
    request: RepairRequest[Draft],
    context: RunContext,
) -> Draft:
    del context
    return Draft(
        title=request.current_subject.title,
        body=(
            f"{request.current_subject.body}\n\n"
            "Rollback: disable the release feature flag."
        ),
        revision=request.current_subject.revision + 1,
    )


def create_scripted_example(
    *,
    approval: ApprovalDecision = ApprovalDecision.APPROVED,
) -> ScriptedExample:
    """Build the example with deterministic grades, repair, and publication."""
    grader = ScriptedGrader[Draft](
        descriptor=READINESS,
        outcomes=(
            _grade(status=GradeStatus.FAILURE, score=0.4),
            _grade(status=GradeStatus.PASS, score=0.95),
        ),
    )
    repair_step: Step[RepairRequest[Draft], Draft] = FunctionStep(
        descriptor=StepDescriptor(
            step_id="release.add-rollback",
            version="1",
            effect_profile=EffectProfile.IDEMPOTENT,
        ),
        function=_repair_draft,
    )
    authority = ScriptedApprovalAuthority(decision=approval)
    publisher = ScriptedPublisher()
    publish_step: Step[ReleaseCandidate, Publication] = FunctionStep(
        descriptor=StepDescriptor(
            step_id="release.publish",
            version="1",
            effect_profile=EffectProfile.NON_REPEATABLE,
        ),
        function=publisher.publish,
    )
    return ScriptedExample(
        workflow=build_release_workflow(
            grader=grader,
            policy=_release_policy(),
            repair_step=repair_step,
            approval_authority=authority,
            publish_step=publish_step,
            max_attempts=2,
            max_grading_cost=0,
        ),
        grader=grader,
        authority=authority,
        publisher=publisher,
    )


def create_context() -> tuple[RunContext, InMemoryEventSink]:
    sink = InMemoryEventSink()
    context = RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialRunIdGenerator(),
        cancellation=CancellationSource().token,
        event_sink=sink,
        event_id_generator=SequentialEventIdGenerator(),
        correlation={"example": "review-repair-approve"},
    )
    return context, sink


async def run_scripted_example(
    *,
    approval: ApprovalDecision = ApprovalDecision.APPROVED,
) -> ScriptedRun:
    configured = create_scripted_example(approval=approval)
    context, sink = create_context()
    outcome = await configured.workflow.execute(
        Draft(
            title="Feature Launch",
            body="Enable the feature for ten percent of accounts.",
        ),
        context,
    )
    return ScriptedRun(
        outcome=outcome,
        events=sink.events,
        grader_calls=len(configured.grader.calls),
        approval_calls=len(configured.authority.calls),
        publisher_calls=len(configured.publisher.calls),
    )


def main() -> None:
    run = asyncio.run(run_scripted_example())
    result = run.outcome.unwrap()
    summary = {
        "approval": result.approval.decision.value,
        "destination": result.publication.destination,
        "event_kinds": [event.kind.value for event in run.events],
        "final_grade": result.candidate.grades[0].status.value,
        "final_revision": result.candidate.draft.revision,
        "repairs": result.candidate.repairs,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
