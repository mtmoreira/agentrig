"""Deterministic executable for the bounded coding example."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime

from agentrig.agents import AgentExecutionResult
from agentrig.capabilities import (
    CapabilityFeature,
    CapabilityKind,
    CapabilityRequirements,
    CodingAgent,
    CodingResult,
    CodingTask,
    DataRetention,
    WorkspaceAuthorization,
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

from examples.capabilities.bounded_coding.workflow import (
    configure_runtime_coding_agent,
)

SCRIPTED_RUNTIME_CAPABILITY_ID = "example.scripted.agent_runtime"
SCRIPTED_TOOL_ID = "example.scripted.workspace_tool"


@dataclass(frozen=True, slots=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 14, 16, 0, tzinfo=UTC)

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
    agent: CodingAgent


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedRun:
    result: CodingResult
    events: tuple[Event, ...]
    runtime: ScriptedAgentRuntime


def example_task(
    *,
    root_path: str = "/work/example",
    max_changed_files: int = 1,
) -> CodingTask:
    return CodingTask(
        task_id="add-greeting",
        workspace=WorkspaceAuthorization(
            workspace_id="bounded-example",
            root_path=root_path,
            writable_roots=(".",),
        ),
        objective=(
            "Create greeting.py with greet(name: str) returning "
            "the exact text 'Hello, {name}!' using the supplied name."
        ),
        acceptance_criteria=(
            "Only greeting.py is changed.",
            "python -m py_compile greeting.py exits successfully.",
        ),
        max_changed_files=max_changed_files,
        requirements=CapabilityRequirements(
            kind=CapabilityKind.CODING,
            features=frozenset({CapabilityFeature.TOOL_USE}),
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
                            "changed_files": (
                                {
                                    "path": "greeting.py",
                                    "change_kind": "added",
                                },
                            ),
                            "validations": (
                                {
                                    "validation_id": "python-compile",
                                    "summary": "The module compiled successfully.",
                                    "exit_code": 0,
                                },
                            ),
                        }
                    )
                )
            ),
        )
    )
    return ScriptedExample(
        runtime=runtime,
        agent=configure_runtime_coding_agent(
            runtime,
            runtime_capability_id=SCRIPTED_RUNTIME_CAPABILITY_ID,
            tool_id=SCRIPTED_TOOL_ID,
            coding_capability_id="example.scripted.coding",
            coding_capability_version="1",
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
            correlation={"example": "bounded-coding"},
        ),
        sink,
    )


async def run_scripted_example(
    configured: ScriptedExample | None = None,
    *,
    task: CodingTask | None = None,
) -> ScriptedRun:
    owned = configured if configured is not None else create_scripted_example()
    context, sink = create_context()
    result = await owned.agent.execute(
        task if task is not None else example_task(),
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
                "changed_files": [
                    change.path for change in run.result.changed_files
                ],
                "event_kinds": [event.kind.value for event in run.events],
                "runtime": "scripted",
                "status": run.result.status.value,
                "validations": [
                    validation.validation_id
                    for validation in run.result.validations
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
