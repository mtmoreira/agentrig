from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass
from datetime import UTC, datetime

from agentrig.agents import (
    AgentContract,
    AgentExecutionResult,
    AgentLimits,
    ConfiguredAgent,
)
from agentrig.core import (
    CancellationSource,
    EffectProfile,
    EventId,
    Failure,
    FailureKind,
    GradeDecision,
    GradePolicyDescriptor,
    GradeReference,
    GradeThreshold,
    InMemoryEventSink,
    JsonValue,
    RunContext,
    RunId,
    ThresholdGradePolicy,
)
from agentrig.evals import (
    AgentEvalTarget,
    EvalChangeKind,
    EvalInconclusiveReason,
    EvalMetric,
    EvalReport,
    EvalRunner,
    EvalSubject,
    PromotionDecision,
    compare_to_baseline,
)
from agentrig.testing import (
    ScriptedAgentProgress,
    ScriptedAgentRuntime,
    ScriptedAgentScenario,
)
from evals.baselines import (
    SCRIPTED_AGENT_PROMOTION_POLICY,
    create_scripted_agent_baseline,
)
from evals.datasets import SCRIPTED_AGENT_DATASET
from agentrig.workflow import GradeStep
from evals.graders import (
    SCRIPTED_AGENT_GRADER,
    ScriptedAgentExactOutputGrader,
)


@dataclass
class DeterministicClock:
    monotonic_time: float = 100.0
    increment: float = 0.25

    def now(self) -> datetime:
        return datetime(2026, 8, 13, 22, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        current = self.monotonic_time
        self.monotonic_time += self.increment
        return current


@dataclass
class SequentialRunIdGenerator:
    next_value: int = 1

    def generate(self) -> RunId:
        value = RunId(f"run-{self.next_value}")
        self.next_value += 1
        return value


@dataclass
class SequentialEventIdGenerator:
    next_value: int = 1

    def generate(self) -> EventId:
        value = EventId(f"event-{self.next_value}")
        self.next_value += 1
        return value


@dataclass(frozen=True, slots=True)
class TextInputCodec:
    schema_id: str = "example.private-prompt.v1"

    def encode(self, value: str) -> JsonValue:
        if not isinstance(value, str):
            raise TypeError("scripted eval input must be a string")
        return value


@dataclass(frozen=True, slots=True)
class TextOutputCodec:
    schema_id: str = "example.answer.v1"

    def decode(self, value: JsonValue) -> str:
        if not isinstance(value, str):
            raise TypeError("scripted eval output must be a string")
        return value


def create_context() -> tuple[RunContext, InMemoryEventSink]:
    sink = InMemoryEventSink()
    context = RunContext.create_root(
        clock=DeterministicClock(),
        id_generator=SequentialRunIdGenerator(),
        cancellation=CancellationSource().token,
        event_sink=sink,
        event_id_generator=SequentialEventIdGenerator(),
        correlation={"suite_id": "scripted-agent"},
    )
    return context, sink


def create_agent(
    *,
    version: str,
    results: tuple[AgentExecutionResult, ...],
) -> tuple[ConfiguredAgent[str, str], ScriptedAgentRuntime]:
    runtime = ScriptedAgentRuntime(
        scenarios=tuple(
            ScriptedAgentScenario(
                result=result,
                actions=(ScriptedAgentProgress(message="Evaluating case"),),
            )
            for result in results
        )
    )
    agent = ConfiguredAgent[str, str](
        runtime=runtime,
        contract=AgentContract(
            agent_id="scripted-answer",
            version=version,
            purpose="Return deterministic answers for the eval harness",
            input_schema="example.private-prompt.v1",
            output_schema="example.answer.v1",
            prompt_version=f"prompt-{version}",
            effect_profile=EffectProfile.READ_ONLY,
            limits=AgentLimits(max_turns=1, max_tool_calls=0),
            stopping_policy="output_schema_satisfied",
        ),
        instructions="Return the configured deterministic answer.",
        input_codec=TextInputCodec(),
        output_codec=TextOutputCodec(),
    )
    return agent, runtime


def run_report(
    *,
    version: str,
    results: tuple[AgentExecutionResult, ...],
) -> tuple[EvalReport, ScriptedAgentRuntime]:
    agent, runtime = create_agent(version=version, results=results)
    runner = EvalRunner[str, str](
        target=AgentEvalTarget(agent),
        graders=(ScriptedAgentExactOutputGrader(),),
    )
    context, _ = create_context()
    result = asyncio.run(runner.run(SCRIPTED_AGENT_DATASET, context))
    report = EvalReport.from_run(
        result,
        environment={"mode": "offline", "api_key": "report-private"},
    )
    return report, runtime


class ScriptedAgentEvalTest(unittest.TestCase):
    def test_same_grader_runs_in_workflow_and_eval_suite(self) -> None:
        grader = ScriptedAgentExactOutputGrader()
        agent, _ = create_agent(
            version="1",
            results=(
                AgentExecutionResult.succeeded("ALPHA"),
                AgentExecutionResult.succeeded("BETA"),
            ),
        )
        target = AgentEvalTarget(agent)
        runner = EvalRunner[str, str](
            target=target,
            graders=(grader,),
        )
        eval_context, _ = create_context()

        eval_result = asyncio.run(
            runner.run(SCRIPTED_AGENT_DATASET, eval_context)
        )

        subject = EvalSubject(
            case=SCRIPTED_AGENT_DATASET.cases[0],
            target=target.descriptor,
            output="ALPHA",
        )
        grade_step = GradeStep[EvalSubject[str, str]](
            graders=(grader,),
            policy=ThresholdGradePolicy(
                descriptor=GradePolicyDescriptor(
                    policy_id="scripted-agent.release",
                    version="1",
                ),
                thresholds=(
                    GradeThreshold(
                        grade=GradeReference(
                            grader_id=SCRIPTED_AGENT_GRADER.grader_id,
                            grader_version=SCRIPTED_AGENT_GRADER.version,
                            metric="exact_output",
                        ),
                        minimum_score=1.0,
                    ),
                ),
            ),
        )
        workflow_context, _ = create_context()

        workflow_result = asyncio.run(
            grade_step.run(subject, workflow_context)
        )

        self.assertEqual(workflow_result.decision, GradeDecision.CONTINUE)
        self.assertEqual(
            workflow_result.grades,
            eval_result.cases[0].grades,
        )

    def test_multiple_cases_produce_a_private_deterministic_report(self) -> None:
        report, runtime = run_report(
            version="1",
            results=(
                AgentExecutionResult.succeeded("ALPHA"),
                AgentExecutionResult.succeeded("BETA"),
            ),
        )
        baseline = create_scripted_agent_baseline(report)
        comparison = compare_to_baseline(baseline, report)

        self.assertEqual(report.summary.case_count, 2)
        self.assertEqual(report.summary.succeeded_cases, 2)
        self.assertEqual(report.summary.passing_grades, 2)
        self.assertEqual(len(runtime.calls), 2)
        self.assertEqual(comparison.changes, ())
        self.assertEqual(
            SCRIPTED_AGENT_PROMOTION_POLICY.decide(comparison),
            PromotionDecision.PROMOTE,
        )

        serialized = report.to_json()
        for private_value in (
            "private prompt alpha",
            "private prompt beta",
            "ALPHA",
            "BETA",
            "report-private",
        ):
            self.assertNotIn(private_value, serialized)

    def test_known_output_regression_is_rejected(self) -> None:
        approved, _ = run_report(
            version="1",
            results=(
                AgentExecutionResult.succeeded("ALPHA"),
                AgentExecutionResult.succeeded("BETA"),
            ),
        )
        candidate, _ = run_report(
            version="2",
            results=(
                AgentExecutionResult.succeeded("ALPHA"),
                AgentExecutionResult.succeeded("REGRESSION"),
            ),
        )

        comparison = compare_to_baseline(
            create_scripted_agent_baseline(approved),
            candidate,
        )

        self.assertTrue(
            all(
                change.kind is EvalChangeKind.REGRESSION
                for change in comparison.changes
            )
        )
        self.assertEqual(
            {change.metric for change in comparison.regressions},
            {EvalMetric.GRADE_STATUS, EvalMetric.GRADE_SCORE},
        )
        self.assertEqual(
            SCRIPTED_AGENT_PROMOTION_POLICY.decide(comparison),
            PromotionDecision.REJECT,
        )

    def test_unavailable_live_runtime_is_inconclusive(self) -> None:
        approved, _ = run_report(
            version="1",
            results=(
                AgentExecutionResult.succeeded("ALPHA"),
                AgentExecutionResult.succeeded("BETA"),
            ),
        )
        unavailable = Failure(
            kind=FailureKind.WORKFLOW_BLOCKED,
            message="live provider credentials are unavailable",
            code="provider.credentials_unavailable",
        )
        candidate, _ = run_report(
            version="2",
            results=(
                AgentExecutionResult.from_failure(unavailable),
                AgentExecutionResult.from_failure(unavailable),
            ),
        )

        comparison = compare_to_baseline(
            create_scripted_agent_baseline(approved),
            candidate,
        )

        self.assertEqual(candidate.summary.blocked_cases, 2)
        self.assertEqual(
            {item.reason for item in comparison.inconclusive},
            {EvalInconclusiveReason.BLOCKED_CASE},
        )
        self.assertEqual(
            SCRIPTED_AGENT_PROMOTION_POLICY.decide(comparison),
            PromotionDecision.INCONCLUSIVE,
        )


if __name__ == "__main__":
    unittest.main()
