"""Provider-neutral composition for a reviewed publication workflow."""

from __future__ import annotations

from dataclasses import dataclass

from agentrig.core import (
    EffectProfile,
    Grade,
    GradeDecision,
    GradePolicy,
    Grader,
    RunContext,
)
from agentrig.workflow import (
    ApprovalAuthority,
    ApprovalRequest,
    ApprovalResolution,
    ApprovalStep,
    ApprovalStepResult,
    FunctionStep,
    GradeStep,
    RepairBudget,
    RepairLoop,
    RepairLoopResult,
    RepairRequest,
    Sequence,
    Step,
    StepDescriptor,
    Workflow,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Draft:
    """One versioned draft entering release review."""

    title: str
    body: str
    revision: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class ReleaseCandidate:
    """The accepted draft and evidence presented for publication approval."""

    draft: Draft
    repairs: int
    grades: tuple[Grade, ...]
    decision: GradeDecision


@dataclass(frozen=True, slots=True, kw_only=True)
class Publication:
    """Portable publication record returned by the protected action."""

    destination: str
    revision: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ReleaseResult:
    """Final typed output spanning review, approval, and publication."""

    candidate: ReleaseCandidate
    publication: Publication
    approval: ApprovalResolution


def build_release_workflow(
    *,
    grader: Grader[Draft],
    policy: GradePolicy,
    repair_step: Step[RepairRequest[Draft], Draft],
    approval_authority: ApprovalAuthority[ReleaseCandidate],
    publish_step: Step[ReleaseCandidate, Publication],
    max_attempts: int = 2,
    max_grading_cost: float = 0,
) -> Workflow[Draft, ReleaseResult]:
    """Compose review, bounded repair, approval, and publication boundaries."""
    grade_step = GradeStep[Draft](graders=(grader,), policy=policy)
    repair_loop: Step[Draft, RepairLoopResult[Draft]] = RepairLoop(
        repair_step=repair_step,
        grade_step=grade_step,
        max_attempts=max_attempts,
        budget=RepairBudget(max_grading_cost=max_grading_cost),
    )

    async def prepare_approval(
        reviewed: RepairLoopResult[Draft],
        context: RunContext,
    ) -> ApprovalRequest[ReleaseCandidate]:
        del context
        candidate = ReleaseCandidate(
            draft=reviewed.subject,
            repairs=reviewed.repairs,
            grades=reviewed.grading.grades,
            decision=reviewed.grading.decision,
        )
        return ApprovalRequest(
            approval_id=f"publish-revision-{candidate.draft.revision}",
            action=publish_step.descriptor,
            summary="Publish the reviewed release draft",
            proposed_input=candidate,
        )

    prepare_step: Step[
        RepairLoopResult[Draft],
        ApprovalRequest[ReleaseCandidate],
    ] = FunctionStep(
        descriptor=StepDescriptor(
            step_id="release.prepare-approval",
            version="1",
            effect_profile=EffectProfile.READ_ONLY,
        ),
        function=prepare_approval,
    )
    approval_step: Step[
        ApprovalRequest[ReleaseCandidate],
        ApprovalStepResult[ReleaseCandidate, Publication],
    ] = ApprovalStep(
        action=publish_step,
        authority=approval_authority,
    )

    async def finalize(
        approved: ApprovalStepResult[ReleaseCandidate, Publication],
        context: RunContext,
    ) -> ReleaseResult:
        del context
        return ReleaseResult(
            candidate=approved.request.proposed_input,
            publication=approved.output,
            approval=approved.resolution,
        )

    finalize_step: Step[
        ApprovalStepResult[ReleaseCandidate, Publication],
        ReleaseResult,
    ] = FunctionStep(
        descriptor=StepDescriptor(
            step_id="release.finalize",
            version="1",
            effect_profile=EffectProfile.READ_ONLY,
        ),
        function=finalize,
    )
    workflow: Workflow[Draft, ReleaseResult] = Sequence(
        repair_loop,
        prepare_step,
        approval_step,
        finalize_step,
    )
    return workflow
