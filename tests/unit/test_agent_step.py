from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agentrig.agents import AgentContract, AgentLimits, AgentResult
from agentrig.core import (
    CancellationSource,
    Deadline,
    DeadlineExceeded,
    EffectProfile,
    EventId,
    EventKind,
    ExecutionStatus,
    Failure,
    FailureKind,
    InMemoryEventSink,
    NoOpRedactionPolicy,
    RunCancelled,
    RunContext,
    RunId,
)
from agentrig.workflow import (
    AgentStep,
    FunctionStep,
    RetryPolicy,
    Sequence,
    Step,
    StepDescriptor,
    execute_step,
    execute_step_with_retry,
)


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 13, 23, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


@dataclass
class SequentialIdGenerator:
    prefix: str
    next_value: int = 1

    def generate(self) -> RunId | EventId:
        value = f"{self.prefix}-{self.next_value}"
        self.next_value += 1
        if self.prefix == "run":
            return RunId(value)
        return EventId(value)


def create_contract(
    *,
    effect_profile: EffectProfile = EffectProfile.READ_ONLY,
) -> AgentContract[str, str]:
    return AgentContract(
        agent_id="writer",
        version="2",
        purpose="Write one bounded response",
        input_schema="example.prompt.v1",
        output_schema="example.response.v1",
        prompt_version="prompt-2",
        effect_profile=effect_profile,
        limits=AgentLimits(max_turns=3, max_tool_calls=1),
        stopping_policy="output_schema_satisfied",
    )


def create_context(
    sink: InMemoryEventSink | None = None,
    source: CancellationSource | None = None,
    *,
    deadline: Deadline | None = None,
) -> RunContext:
    owned_sink = (
        sink
        if sink is not None
        else InMemoryEventSink(redaction_policy=NoOpRedactionPolicy())
    )
    owned_source = source if source is not None else CancellationSource()
    return RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialIdGenerator("run"),  # type: ignore[arg-type]
        cancellation=owned_source.token,
        event_sink=owned_sink,
        event_id_generator=SequentialIdGenerator("event"),  # type: ignore[arg-type]
        deadline=deadline,
    )


@dataclass
class RecordingAgent:
    contract: AgentContract[str, str]
    results: tuple[AgentResult[str], ...]
    calls: list[tuple[str, RunContext]] = field(default_factory=list)

    async def run(
        self,
        input: str,
        context: RunContext,
    ) -> AgentResult[str]:
        index = len(self.calls)
        self.calls.append((input, context))
        return self.results[min(index, len(self.results) - 1)]


async def invoke_typed_step(
    step: Step[str, str],
    input: str,
    context: RunContext,
) -> str:
    return await step.run(input, context)


def transient_failure() -> Failure:
    return Failure(
        kind=FailureKind.TRANSIENT_PROVIDER,
        message="provider temporarily unavailable",
        code="provider.busy",
    )


class AgentStepTest(unittest.TestCase):
    def test_derives_descriptor_and_runs_a_typed_agent(self) -> None:
        agent = RecordingAgent(
            contract=create_contract(),
            results=(AgentResult.succeeded("complete"),),
        )
        step = AgentStep(agent)
        context = create_context()

        result = asyncio.run(invoke_typed_step(step, "draft", context))

        self.assertIsInstance(step, Step)
        self.assertEqual(result, "complete")
        self.assertEqual(agent.calls, [("draft", context)])
        self.assertEqual(
            step.descriptor,
            StepDescriptor(
                step_id="writer",
                version="2",
                effect_profile=EffectProfile.READ_ONLY,
            ),
        )

    def test_preserves_normalized_agent_failure_at_workflow_boundary(self) -> None:
        sink = InMemoryEventSink(redaction_policy=NoOpRedactionPolicy())
        failure = Failure(
            kind=FailureKind.WORKFLOW_BLOCKED,
            message="workflow requires external state",
            code="agent.waiting",
        )
        step = AgentStep(
            RecordingAgent(
                contract=create_contract(),
                results=(AgentResult.from_failure(failure),),
            )
        )

        outcome = asyncio.run(
            execute_step(step, "draft", create_context(sink).derive_child())
        )

        self.assertEqual(outcome.status, ExecutionStatus.BLOCKED)
        self.assertIs(outcome.failure, failure)
        self.assertEqual(
            [event.kind for event in sink.events],
            [EventKind.STEP_STARTED, EventKind.STEP_COMPLETED],
        )
        self.assertEqual(sink.events[-1].attributes["step_id"], "writer")
        self.assertEqual(
            sink.events[-1].attributes["failure_code"],
            "agent.waiting",
        )

    def test_contract_effect_profile_controls_transient_retry(self) -> None:
        retryable_sink = InMemoryEventSink(
            redaction_policy=NoOpRedactionPolicy()
        )
        retryable_agent = RecordingAgent(
            contract=create_contract(effect_profile=EffectProfile.IDEMPOTENT),
            results=(
                AgentResult.from_failure(transient_failure()),
                AgentResult.succeeded("complete"),
            ),
        )

        retried = asyncio.run(
            execute_step_with_retry(
                AgentStep(retryable_agent),
                "draft",
                create_context(retryable_sink).derive_child(),
                retry_policy=RetryPolicy(max_attempts=2),
            )
        )

        self.assertEqual(retried.unwrap(), "complete")
        self.assertEqual(len(retryable_agent.calls), 2)
        self.assertIn(
            EventKind.RETRY_SCHEDULED,
            [event.kind for event in retryable_sink.events],
        )

        non_repeatable_sink = InMemoryEventSink(
            redaction_policy=NoOpRedactionPolicy()
        )
        non_repeatable_agent = RecordingAgent(
            contract=create_contract(
                effect_profile=EffectProfile.NON_REPEATABLE
            ),
            results=(
                AgentResult.from_failure(transient_failure()),
                AgentResult.succeeded("unreachable"),
            ),
        )

        not_retried = asyncio.run(
            execute_step_with_retry(
                AgentStep(non_repeatable_agent),
                "draft",
                create_context(non_repeatable_sink).derive_child(),
                retry_policy=RetryPolicy(max_attempts=2),
            )
        )

        self.assertEqual(not_retried.status, ExecutionStatus.FAILED)
        self.assertEqual(len(non_repeatable_agent.calls), 1)
        self.assertNotIn(
            EventKind.RETRY_SCHEDULED,
            [event.kind for event in non_repeatable_sink.events],
        )

    def test_composes_with_function_steps_in_a_typed_sequence(self) -> None:
        agent = RecordingAgent(
            contract=create_contract(),
            results=(AgentResult.succeeded("COMPLETE"),),
        )

        async def length(input: str, context: RunContext) -> int:
            del context
            return len(input)

        sequence: Sequence[str, int] = Sequence(
            AgentStep(agent),
            FunctionStep(
                descriptor=StepDescriptor(
                    step_id="text.length",
                    version="1",
                    effect_profile=EffectProfile.READ_ONLY,
                ),
                function=length,
            ),
        )

        outcome = asyncio.run(sequence.execute("draft", create_context()))

        self.assertEqual(outcome.unwrap(), 8)
        self.assertEqual(agent.calls[0][0], "draft")

    def test_constraints_are_checked_before_invoking_the_agent(self) -> None:
        agent = RecordingAgent(
            contract=create_contract(),
            results=(AgentResult.succeeded("unreachable"),),
        )
        step = AgentStep(agent)
        source = CancellationSource()
        source.cancel("caller stopped")

        with self.assertRaises(RunCancelled):
            asyncio.run(step.run("cancelled", create_context(source=source)))

        expired = Deadline(
            expires_at=datetime(2026, 8, 13, 23, 0, tzinfo=UTC),
            monotonic_deadline=100.0,
        )
        with self.assertRaises(DeadlineExceeded):
            asyncio.run(
                step.run("expired", create_context(deadline=expired))
            )

        self.assertEqual(agent.calls, [])

    def test_rejects_invalid_agent_context_contract_and_result(self) -> None:
        with self.assertRaises(TypeError):
            AgentStep("not-an-agent")  # type: ignore[arg-type]

        @dataclass(frozen=True)
        class InvalidContractAgent:
            contract: object = "not-a-contract"

            async def run(self, input: str, context: RunContext) -> object:
                del input, context
                return AgentResult.succeeded("unreachable")

        with self.assertRaises(TypeError):
            AgentStep(InvalidContractAgent())  # type: ignore[arg-type]

        @dataclass(frozen=True)
        class InvalidResultAgent:
            contract: AgentContract[str, str]

            async def run(self, input: str, context: RunContext) -> object:
                del input, context
                return "not-a-result"

        invalid_result_step = AgentStep(
            InvalidResultAgent(create_contract())  # type: ignore[arg-type]
        )
        with self.assertRaises(TypeError):
            asyncio.run(invalid_result_step.run("draft", create_context()))

        valid_step = AgentStep(
            RecordingAgent(
                contract=create_contract(),
                results=(AgentResult.succeeded("complete"),),
            )
        )
        with self.assertRaises(TypeError):
            asyncio.run(
                valid_step.run(
                    "draft",
                    "not-a-context",  # type: ignore[arg-type]
                )
            )


if __name__ == "__main__":
    unittest.main()
