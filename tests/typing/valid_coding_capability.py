"""Positive fixture for typed authorized coding-agent execution."""

from dataclasses import dataclass

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityKind,
    ChangedFileEvidence,
    CodingAgent,
    CodingChangeKind,
    CodingResult,
    CodingTask,
    CodingValidationStatus,
    DataRetention,
    ValidationEvidence,
)
from agentrig.core import RunContext


@dataclass(frozen=True)
class DeterministicCodingAgent:
    descriptor: CapabilityDescriptor

    async def execute(
        self,
        task: CodingTask,
        context: RunContext,
    ) -> CodingResult:
        del context
        task.require_supported_by(self.descriptor)
        return CodingResult.succeeded(
            task=task,
            changed_files=(
                ChangedFileEvidence(
                    path="src/example.py",
                    change_kind=CodingChangeKind.MODIFIED,
                ),
            ),
            validations=(
                ValidationEvidence(
                    validation_id="unit",
                    status=CodingValidationStatus.PASSED,
                    summary="Unit tests passed.",
                    exit_code=0,
                ),
            ),
        )


agent: CodingAgent = DeterministicCodingAgent(
    descriptor=CapabilityDescriptor(
        capability_id="example.coding",
        version="1",
        kind=CapabilityKind.CODING,
        data_retention=DataRetention.NOT_RETAINED,
    )
)


async def execute_task(task: CodingTask, context: RunContext) -> CodingResult:
    return await agent.execute(task, context)
