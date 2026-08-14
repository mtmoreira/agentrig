"""Deterministic executable for configured-agent workflow substitution."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
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
from agentrig.testing import (
    ScriptedAgentProgress,
    ScriptedAgentRuntime,
    ScriptedAgentScenario,
    ScriptedToolRequest,
)
from agentrig.workflow import Workflow

from examples.agents.configured_workflow.workflow import (
    DeliveredBrief,
    ResearchRequest,
    build_delivery_workflow,
    configure_researcher,
    expose_delivery_agent,
)


@dataclass(frozen=True, slots=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 14, 13, 0, tzinfo=UTC)

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
    workflow: Workflow[ResearchRequest, DeliveredBrief]
    agent: Agent[ResearchRequest, DeliveredBrief]


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedRun:
    result: AgentResult[DeliveredBrief]
    events: tuple[Event, ...]
    runtime: ScriptedAgentRuntime


def create_scripted_example(
    result: AgentExecutionResult | None = None,
    *,
    tool_name: str = "search",
) -> ScriptedExample:
    runtime = ScriptedAgentRuntime(
        scenarios=(
            ScriptedAgentScenario(
                actions=(
                    ScriptedAgentProgress(message="Searching approved sources"),
                    ScriptedToolRequest(tool_name=tool_name),
                ),
                result=(
                    result
                    if result is not None
                    else AgentExecutionResult.succeeded(
                        {
                            "answer": (
                                "Typed agent contracts keep runtime details "
                                "behind a portable boundary."
                            ),
                            "sources": ("https://example.test/contracts",),
                        },
                        provider_metadata={"session_id": "scripted-session-1"},
                    )
                ),
            ),
        )
    )
    researcher = configure_researcher(runtime)
    workflow = build_delivery_workflow(researcher)
    return ScriptedExample(
        runtime=runtime,
        workflow=workflow,
        agent=expose_delivery_agent(workflow),
    )


def create_context() -> tuple[RunContext, InMemoryEventSink]:
    sink = InMemoryEventSink()
    context = RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialRunIdGenerator(),
        cancellation=CancellationSource().token,
        event_sink=sink,
        event_id_generator=SequentialEventIdGenerator(),
        correlation={"example": "configured-workflow"},
    )
    return context, sink


async def run_scripted_example(
    configured: ScriptedExample | None = None,
) -> ScriptedRun:
    owned = configured if configured is not None else create_scripted_example()
    context, sink = create_context()
    result = await owned.agent.run(
        ResearchRequest(topic="portable agent contracts"),
        context,
    )
    return ScriptedRun(result=result, events=sink.events, runtime=owned.runtime)


def main() -> None:
    run = asyncio.run(run_scripted_example())
    delivered = run.result.unwrap()
    runtime_call = run.runtime.calls[0]
    encoded_input = runtime_call.request.input
    if not isinstance(encoded_input, Mapping):
        raise AssertionError("configured input did not encode as an object")
    summary = {
        "answer": delivered.brief.answer,
        "delivery_agent": "research-delivery",
        "encoded_input": dict(encoded_input),
        "event_kinds": [event.kind.value for event in run.events],
        "research_agent": runtime_call.request.contract.agent_id,
        "source_count": len(delivered.brief.sources),
        "word_count": delivered.word_count,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
