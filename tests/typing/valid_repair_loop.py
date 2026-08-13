"""Positive fixture for typed bounded repair composition."""

from collections.abc import Sequence as GradeSequence
from dataclasses import dataclass

from agentrig.core import (
    EffectProfile,
    Grade,
    GradeClassification,
    GradeDecision,
    GradePolicyDescriptor,
    GradeStatus,
    GraderDescriptor,
    GradingContext,
    RunContext,
)
from agentrig.workflow import (
    FunctionStep,
    GradeStep,
    RepairBudget,
    RepairLoop,
    RepairLoopResult,
    RepairRequest,
    Step,
    StepDescriptor,
)


@dataclass(frozen=True)
class TextGrader:
    descriptor: GraderDescriptor

    async def grade(self, subject: str, context: GradingContext) -> Grade:
        del context
        return Grade(
            grader=self.descriptor,
            metric="nonempty",
            status=GradeStatus.PASS if subject else GradeStatus.FAILURE,
            classification=GradeClassification.HARD,
            explanation="The text must be nonempty.",
        )


@dataclass(frozen=True)
class RepairPolicy:
    descriptor: GradePolicyDescriptor

    def decide(self, grades: GradeSequence[Grade]) -> GradeDecision:
        if any(grade.status is GradeStatus.FAILURE for grade in grades):
            return GradeDecision.REPAIR
        return GradeDecision.CONTINUE


async def repair_text(
    request: RepairRequest[str],
    context: RunContext,
) -> str:
    del context
    return request.current_subject or "repaired"


repair_step = FunctionStep(
    descriptor=StepDescriptor(
        step_id="text.repair",
        version="1",
        effect_profile=EffectProfile.READ_ONLY,
    ),
    function=repair_text,
)
grade_step = GradeStep[str](
    graders=(
        TextGrader(
            descriptor=GraderDescriptor(grader_id="text.present", version="1")
        ),
    ),
    policy=RepairPolicy(
        descriptor=GradePolicyDescriptor(policy_id="text.release", version="1")
    ),
)
loop: Step[str, RepairLoopResult[str]] = RepairLoop(
    repair_step=repair_step,
    grade_step=grade_step,
    max_attempts=2,
    budget=RepairBudget(max_grading_cost=0),
)
