"""Positive fixture for typed approval-gated workflow composition."""

from dataclasses import dataclass

from agentrig.core import EffectProfile, RunContext
from agentrig.workflow import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResolution,
    ApprovalStep,
    ApprovalStepResult,
    FunctionStep,
    Step,
    StepDescriptor,
)


async def write_text(input: str, context: RunContext) -> int:
    del context
    return len(input)


@dataclass(frozen=True)
class TextApprovalAuthority:
    async def resolve(
        self,
        request: ApprovalRequest[str],
        context: RunContext,
    ) -> ApprovalResolution:
        del context
        return ApprovalResolution(
            approval_id=request.approval_id,
            decision=ApprovalDecision.APPROVED,
            resolver="fixture",
        )


action = FunctionStep(
    descriptor=StepDescriptor(
        step_id="text.write",
        version="1",
        effect_profile=EffectProfile.NON_REPEATABLE,
    ),
    function=write_text,
)
step: Step[
    ApprovalRequest[str],
    ApprovalStepResult[str, int],
] = ApprovalStep(action=action, authority=TextApprovalAuthority())
request = ApprovalRequest(
    approval_id="approval-1",
    action=action.descriptor,
    summary="Write the prepared text",
    proposed_input="prepared text",
)
