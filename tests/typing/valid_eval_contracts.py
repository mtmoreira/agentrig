"""Positive fixture for typed evaluation datasets and targets."""

from dataclasses import dataclass

from agentrig.core import (
    ExecutionOutcome,
    Grade,
    GradeClassification,
    GradeStatus,
    GraderDescriptor,
    GradingContext,
    RunContext,
)
from agentrig.evals import (
    EvalCase,
    EvalDataset,
    EvalRunner,
    EvalSubject,
    EvalTarget,
    EvalTargetDescriptor,
    EvalTargetKind,
)


@dataclass(frozen=True)
class LengthTarget:
    descriptor: EvalTargetDescriptor

    async def run(
        self,
        input: str,
        context: RunContext,
    ) -> ExecutionOutcome[int]:
        del context
        return ExecutionOutcome.succeeded(len(input))


case = EvalCase[str](
    case_id="text.length",
    version="1",
    input="draft",
    expected_constraints=("Return the number of characters.",),
)
dataset: EvalDataset[str] = EvalDataset(
    dataset_id="text-basics",
    version="2026-08-13",
    cases=(case,),
)
target: EvalTarget[str, int] = LengthTarget(
    descriptor=EvalTargetDescriptor(
        target_id="length",
        version="1",
        kind=EvalTargetKind.CAPABILITY,
    )
)


@dataclass(frozen=True)
class LengthGrader:
    descriptor: GraderDescriptor

    async def grade(
        self,
        subject: EvalSubject[str, int],
        context: GradingContext,
    ) -> Grade:
        del context
        return Grade(
            grader=self.descriptor,
            metric="nonnegative",
            status=(
                GradeStatus.PASS
                if subject.output >= 0
                else GradeStatus.FAILURE
            ),
            classification=GradeClassification.HARD,
            explanation="The length must be nonnegative.",
        )


runner: EvalRunner[str, int] = EvalRunner(
    target=target,
    graders=(
        LengthGrader(
            descriptor=GraderDescriptor(
                grader_id="length.nonnegative",
                version="1",
            )
        ),
    ),
)
