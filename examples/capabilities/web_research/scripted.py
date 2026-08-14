"""Deterministic executable for runtime-backed web research."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime

from agentrig.agents import AgentExecutionResult
from agentrig.capabilities import (
    DataRetention,
    SearchProvider,
    SearchRequest,
    SearchResult,
)
from agentrig.core import (
    CancellationSource,
    Event,
    EventId,
    InMemoryEventSink,
    RunContext,
    RunId,
)
from agentrig.testing import ScriptedAgentRuntime, ScriptedAgentScenario

from examples.capabilities.web_research.workflow import (
    configure_runtime_search_provider,
)

SCRIPTED_RUNTIME_CAPABILITY_ID = "example.scripted.agent_runtime"
SCRIPTED_WEB_SEARCH_TOOL_ID = "example.scripted.web_search"
SOURCE_ONE = "https://example.test/runtime-boundary"
SOURCE_TWO = "https://example.test/search-contract"


@dataclass(frozen=True, slots=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 14, 17, 0, tzinfo=UTC)

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
    provider: SearchProvider


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedRun:
    result: SearchResult
    events: tuple[Event, ...]
    runtime: ScriptedAgentRuntime


def example_request(*, max_results: int = 2) -> SearchRequest:
    return SearchRequest(
        query="portable autonomous runtime and search contracts",
        max_results=max_results,
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
                            "hits": (
                                {
                                    "source_uri": SOURCE_ONE,
                                    "title": "Portable runtime boundaries",
                                    "summary": (
                                        "Injected runtime contracts keep "
                                        "application policy provider-neutral."
                                    ),
                                },
                                {
                                    "source_uri": SOURCE_TWO,
                                    "title": "Bounded search contracts",
                                    "summary": (
                                        "Result limits and citation identity "
                                        "make search evidence portable."
                                    ),
                                },
                            )
                        }
                    )
                )
            ),
        )
    )
    return ScriptedExample(
        runtime=runtime,
        provider=configure_runtime_search_provider(
            runtime,
            runtime_capability_id=SCRIPTED_RUNTIME_CAPABILITY_ID,
            web_search_tool_id=SCRIPTED_WEB_SEARCH_TOOL_ID,
            search_capability_id="example.scripted.search",
            search_capability_version="1",
            data_retention=DataRetention.NOT_RETAINED,
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
            correlation={"example": "web-research"},
        ),
        sink,
    )


async def run_scripted_example(
    configured: ScriptedExample | None = None,
    *,
    request: SearchRequest | None = None,
) -> ScriptedRun:
    owned = configured if configured is not None else create_scripted_example()
    context, sink = create_context()
    result = await owned.provider.search(
        request if request is not None else example_request(),
        context,
    )
    return ScriptedRun(
        result=result,
        events=sink.events,
        runtime=owned.runtime,
    )


def main() -> None:
    run = asyncio.run(run_scripted_example())
    print(
        json.dumps(
            {
                "citations": [
                    {
                        "source_uri": citation.source_uri,
                        "title": citation.title,
                    }
                    for citation in run.result.citations
                ],
                "event_kinds": [event.kind.value for event in run.events],
                "result_count": len(run.result.hits),
                "runtime": "scripted",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
