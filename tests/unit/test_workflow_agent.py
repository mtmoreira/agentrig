from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agentrig.agents import (
    Agent,
    AgentContract,
    AgentLimits,
    AgentResult,
    AgentStatus,
)
from agentrig.core import (
    ArtifactId,
    ArtifactRef,
    CancellationSource,
    Deadline,
    EffectProfile,
    ExecutionOutcome,
    Failure,
    FailureKind,
    RunContext,
    RunId,
)
from agentrig.workflow import (
    FunctionStep,
    Sequence,
    StepDescriptor,
    Workflow,
    WorkflowAgent,
)


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 14, 0, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


@dataclass
class SequentialRunIdGenerator:
    next_value: int = 1

    def generate(self) -> RunId:
        run_id = RunId(f"run-{self.next_value}")
        self.next_value += 1
        return run_id


def create_context(
    source: CancellationSource | None = None,
    *,
    deadline: Deadline | None = None,
) -> RunContext:
    owned_source = source if source is not None else CancellationSource()
    return RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialRunIdGenerator(),
        cancellation=owned_source.token,
        deadline=deadline,
    )


def create_contract() -> AgentContract[str, str]:
    return AgentContract(
        agent_id="text-workflow",
        version="3",
        purpose="Run one configured text workflow",
        input_schema="example.text.v1",
        output_schema="example.text.v1",
        prompt_version="workflow-3",
        effect_profile=EffectProfile.READ_ONLY,
        limits=AgentLimits(max_turns=2, max_tool_calls=0),
        stopping_policy="workflow_completed",
    )


def create_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId("artifact-1"),
        kind="report",
        media_type="text/plain",
        producer_run_id=RunId("run-producer"),
        workspace_path="outputs/report.txt",
    )


@dataclass
class RecordingWorkflow:
    outcomes: tuple[ExecutionOutcome[str], ...]
    calls: list[tuple[str, RunContext]] = field(default_factory=list)

    async def execute(
        self,
        input: str,
        context: RunContext,
    ) -> ExecutionOutcome[str]:
        index = len(self.calls)
        self.calls.append((input, context))
        return self.outcomes[min(index, len(self.outcomes) - 1)]


async def run_typed_agent(
    agent: Agent[str, str],
    input: str,
    context: RunContext,
) -> AgentResult[str]:
    return await agent.run(input, context)


class WorkflowAgentTest(unittest.TestCase):
    def test_configured_sequence_satisfies_an_agent_contract(self) -> None:
        async def uppercase(input: str, context: RunContext) -> str:
            del context
            return input.upper()

        async def decorate(input: str, context: RunContext) -> str:
            del context
            return f"[{input}]"

        sequence: Sequence[str, str] = Sequence(
            FunctionStep(
                descriptor=StepDescriptor(
                    step_id="uppercase",
                    version="1",
                    effect_profile=EffectProfile.READ_ONLY,
                ),
                function=uppercase,
            ),
            FunctionStep(
                descriptor=StepDescriptor(
                    step_id="decorate",
                    version="1",
                    effect_profile=EffectProfile.READ_ONLY,
                ),
                function=decorate,
            ),
        )
        agent = WorkflowAgent(workflow=sequence, contract=create_contract())

        result = asyncio.run(
            run_typed_agent(agent, "draft", create_context())
        )

        self.assertIsInstance(sequence, Workflow)
        self.assertIsInstance(agent, Agent)
        self.assertIs(agent.workflow, sequence)
        self.assertEqual(agent.contract, create_contract())
        self.assertEqual(result.unwrap(), "[DRAFT]")

    def test_preserves_success_and_failure_artifacts(self) -> None:
        artifact = create_artifact()
        success_workflow = RecordingWorkflow(
            outcomes=(
                ExecutionOutcome.succeeded(
                    "complete",
                    artifacts=(artifact,),
                ),
            )
        )

        success = asyncio.run(
            WorkflowAgent(
                workflow=success_workflow,
                contract=create_contract(),
            ).run("draft", create_context())
        )

        self.assertEqual(success.unwrap(), "complete")
        self.assertEqual(success.artifacts, (artifact,))

        failure = Failure(
            kind=FailureKind.WORKFLOW_BLOCKED,
            message="workflow requires external state",
            code="workflow.waiting",
        )
        failure_workflow = RecordingWorkflow(
            outcomes=(
                ExecutionOutcome.from_failure(
                    failure,
                    artifacts=(artifact,),
                ),
            )
        )

        blocked = asyncio.run(
            WorkflowAgent(
                workflow=failure_workflow,
                contract=create_contract(),
            ).run("draft", create_context())
        )

        self.assertEqual(blocked.status, AgentStatus.BLOCKED)
        self.assertIs(blocked.failure, failure)
        self.assertEqual(blocked.artifacts, (artifact,))

    def test_constraints_prevent_or_override_workflow_success(self) -> None:
        source = CancellationSource()
        source.cancel("caller stopped")
        cancelled_workflow = RecordingWorkflow(
            outcomes=(ExecutionOutcome.succeeded("unreachable"),)
        )

        cancelled = asyncio.run(
            WorkflowAgent(
                workflow=cancelled_workflow,
                contract=create_contract(),
            ).run("draft", create_context(source))
        )

        self.assertEqual(cancelled.status, AgentStatus.CANCELLED)
        self.assertEqual(cancelled_workflow.calls, [])

        expired = Deadline(
            expires_at=datetime(2026, 8, 14, 0, 0, tzinfo=UTC),
            monotonic_deadline=100.0,
        )
        expired_workflow = RecordingWorkflow(
            outcomes=(ExecutionOutcome.succeeded("unreachable"),)
        )

        deadline_result = asyncio.run(
            WorkflowAgent(
                workflow=expired_workflow,
                contract=create_contract(),
            ).run("draft", create_context(deadline=expired))
        )

        self.assertEqual(deadline_result.status, AgentStatus.FAILED)
        self.assertIsNotNone(deadline_result.failure)
        if deadline_result.failure is None:
            raise AssertionError("failed result has no failure")
        self.assertEqual(
            deadline_result.failure.kind,
            FailureKind.DEADLINE_EXCEEDED,
        )
        self.assertEqual(expired_workflow.calls, [])

        post_source = CancellationSource()
        artifact = create_artifact()

        @dataclass(frozen=True)
        class CancellingWorkflow:
            async def execute(
                self,
                input: str,
                context: RunContext,
            ) -> ExecutionOutcome[str]:
                del input, context
                post_source.cancel("stopped during workflow")
                return ExecutionOutcome.succeeded(
                    "unreachable",
                    artifacts=(artifact,),
                )

        post_cancelled = asyncio.run(
            WorkflowAgent(
                workflow=CancellingWorkflow(),
                contract=create_contract(),
            ).run("draft", create_context(post_source))
        )

        self.assertEqual(post_cancelled.status, AgentStatus.CANCELLED)
        self.assertEqual(post_cancelled.artifacts, (artifact,))

    def test_raw_workflow_exception_is_safely_normalized(self) -> None:
        @dataclass(frozen=True)
        class BrokenWorkflow:
            async def execute(
                self,
                input: str,
                context: RunContext,
            ) -> ExecutionOutcome[str]:
                del input, context
                raise RuntimeError("password=private")

        result = asyncio.run(
            WorkflowAgent(
                workflow=BrokenWorkflow(),
                contract=create_contract(),
            ).run("draft", create_context())
        )

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIsNotNone(result.failure)
        if result.failure is None:
            raise AssertionError("failed result has no failure")
        self.assertEqual(result.failure.kind, FailureKind.UNEXPECTED)
        self.assertNotIn("private", result.failure.message)

    def test_rejects_invalid_configuration_context_and_outcome(self) -> None:
        with self.assertRaises(TypeError):
            WorkflowAgent(
                workflow="not-a-workflow",  # type: ignore[arg-type]
                contract=create_contract(),
            )
        with self.assertRaises(TypeError):
            WorkflowAgent(
                workflow=RecordingWorkflow(
                    outcomes=(ExecutionOutcome.succeeded("complete"),)
                ),
                contract="not-a-contract",  # type: ignore[arg-type]
            )

        agent = WorkflowAgent(
            workflow=RecordingWorkflow(
                outcomes=(ExecutionOutcome.succeeded("complete"),)
            ),
            contract=create_contract(),
        )
        with self.assertRaises(TypeError):
            asyncio.run(
                agent.run(
                    "draft",
                    "not-a-context",  # type: ignore[arg-type]
                )
            )

        @dataclass(frozen=True)
        class InvalidOutcomeWorkflow:
            async def execute(
                self,
                input: str,
                context: RunContext,
            ) -> object:
                del input, context
                return "not-an-outcome"

        invalid = asyncio.run(
            WorkflowAgent(
                workflow=InvalidOutcomeWorkflow(),  # type: ignore[arg-type]
                contract=create_contract(),
            ).run("draft", create_context())
        )

        self.assertEqual(invalid.status, AgentStatus.FAILED)
        self.assertIsNotNone(invalid.failure)
        if invalid.failure is None:
            raise AssertionError("failed result has no failure")
        self.assertEqual(invalid.failure.kind, FailureKind.UNEXPECTED)


if __name__ == "__main__":
    unittest.main()
