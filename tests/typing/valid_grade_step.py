"""Positive fixture for typed grading within workflow composition."""

from collections.abc import Sequence as GradeSequence
from dataclasses import dataclass

from agentrig.core import (
    Grade,
    GradeClassification,
    GradeDecision,
    GradePolicyDescriptor,
    GradeStatus,
    GraderDescriptor,
    GradingContext,
)
from agentrig.workflow import GradeStep, GradeStepResult, Step


@dataclass(frozen=True)
class TextGrader:
    descriptor: GraderDescriptor

    async def grade(self, subject: str, context: GradingContext) -> Grade:
        del subject, context
        return Grade(
            grader=self.descriptor,
            metric="nonempty",
            status=GradeStatus.PASS,
            classification=GradeClassification.HARD,
            explanation="The text is present.",
        )


@dataclass(frozen=True)
class ContinuePolicy:
    descriptor: GradePolicyDescriptor

    def decide(self, grades: GradeSequence[Grade]) -> GradeDecision:
        del grades
        return GradeDecision.CONTINUE


step: Step[str, GradeStepResult[str]] = GradeStep(
    graders=(
        TextGrader(
            descriptor=GraderDescriptor(grader_id="text.present", version="1")
        ),
    ),
    policy=ContinuePolicy(
        descriptor=GradePolicyDescriptor(policy_id="text.release", version="1")
    ),
)
