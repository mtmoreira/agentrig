"""Positive fixture for typed evaluation datasets and targets."""

from dataclasses import dataclass

from agentrig.core import ExecutionOutcome, RunContext
from agentrig.evals import (
    EvalCase,
    EvalDataset,
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
