"""Deterministic executable for the structured-agent example."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime

from agentrig.agents import Agent, AgentExecutionResult, AgentResult
from agentrig.core import (
    CancellationSource,
    Event,
    EventId,
    InMemoryEventSink,
    RunContext,
    RunId,
)
from agentrig.testing import ScriptedAgentRuntime, ScriptedAgentScenario

from examples.agents.codex_structured_agent.workflow import (
    DecisionBrief,
    DecisionRequest,
    configure_decision_agent,
)

SCRIPTED_CAPABILITY_ID = "example.scripted.agent_runtime"


@dataclass(frozen=True, slots=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 14, 15, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


@dataclass(slots=True)
class SequentialRunIdGenerator:
    next_value: int = 1

    def generate(self) -> RunId:
        run_id = RunId(f"run-{self.next_value}")
        self.next_value += 1
        return run_id


@dataclass(slots=True)
class SequentialEventIdGenerator:
    next_value: int = 1

    def generate(self) -> EventId:
        event_id = EventId(f"event-{self.next_value}")
        self.next_value += 1
        return event_id


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedExample:
    runtime: ScriptedAgentRuntime
    agent: Agent[DecisionRequest, DecisionBrief]


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedRun:
    result: AgentResult[DecisionBrief]
    events: tuple[Event, ...]
    runtime: ScriptedAgentRuntime


def example_request() -> DecisionRequest:
    return DecisionRequest(
        question="Should the team automate this release check?",
        constraints=(
            "The check must be deterministic offline.",
            "Live provider execution must remain explicit.",
        ),
    )


def create_scripted_example(
    result: AgentExecutionResult | None = None,
) -> ScriptedExample:
    runtime = ScriptedAgentRuntime(
        scenarios=(
            ScriptedAgentScenario(
                result=(
                    result
                    if result is not None
                    else AgentExecutionResult.succeeded(
                        {
                            "summary": (
                                "Automate the deterministic gate and keep the "
                                "provider run separately opted in."
                            ),
                            "risks": (
                                "A live check can fail when authentication is "
                                "unavailable.",
                            ),
                            "recommendation": "proceed",
                        },
                        provider_metadata={"session_id": "scripted-session-1"},
                    )
                )
            ),
        )
    )
    return ScriptedExample(
        runtime=runtime,
        agent=configure_decision_agent(
            runtime,
            runtime_capability_id=SCRIPTED_CAPABILITY_ID,
        ),
    )


def create_context() -> tuple[RunContext, InMemoryEventSink]:
    sink = InMemoryEventSink()
    return (
        RunContext.create_root(
            clock=FixedClock(),
            id_generator=SequentialRunIdGenerator(),
            cancellation=CancellationSource().token,
            event_sink=sink,
            event_id_generator=SequentialEventIdGenerator(),
            correlation={"example": "codex-structured-agent"},
        ),
        sink,
    )


async def run_scripted_example(
    configured: ScriptedExample | None = None,
) -> ScriptedRun:
    owned = configured if configured is not None else create_scripted_example()
    context, sink = create_context()
    result = await owned.agent.run(example_request(), context)
    return ScriptedRun(
        result=result,
        events=sink.events,
        runtime=owned.runtime,
    )


def main() -> None:
    run = asyncio.run(run_scripted_example())
    brief = run.result.unwrap()
    print(
        json.dumps(
            {
                "event_kinds": [event.kind.value for event in run.events],
                "recommendation": brief.recommendation,
                "risk_count": len(brief.risks),
                "runtime": "scripted",
                "summary": brief.summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
