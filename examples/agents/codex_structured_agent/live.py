"""Explicit live Codex composition root for the structured-agent example."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from agentrig.core import (
    CancellationSource,
    Deadline,
    EventId,
    InMemoryEventSink,
    RunContext,
    RunId,
    SystemClock,
    Uuid4IdGenerator,
)
from agentrig.integrations.openai import (
    CODEX_AGENT_RUNTIME_CAPABILITY,
    CodexAgentRuntime,
    CodexSandboxMode,
    CodexSandboxPolicy,
)
from agentrig.integrations.openai.sdk import CodexSdkClientFactory

from examples.agents.codex_structured_agent.scripted import example_request
from examples.agents.codex_structured_agent.workflow import (
    BRIEF_JSON_SCHEMA,
    BRIEF_SCHEMA,
    configure_decision_agent,
)

_LIVE_OPT_IN = "AGENTRIG_RUN_LIVE"
_MODEL_ENVIRONMENT_VARIABLE = "AGENTRIG_CODEX_LIVE_MODEL"
_DEFAULT_MODEL = "gpt-5.6-terra"


def _require_live_opt_in() -> None:
    if os.environ.get(_LIVE_OPT_IN) != "1":
        raise RuntimeError("live example requires AGENTRIG_RUN_LIVE=1")


def _model() -> str:
    model = os.environ.get(_MODEL_ENVIRONMENT_VARIABLE, _DEFAULT_MODEL)
    if not model or model != model.strip():
        raise ValueError(
            f"{_MODEL_ENVIRONMENT_VARIABLE} must be nonempty and trimmed"
        )
    return model


def _context() -> tuple[RunContext, InMemoryEventSink]:
    clock = SystemClock()
    sink = InMemoryEventSink()
    return (
        RunContext.create_root(
            clock=clock,
            id_generator=Uuid4IdGenerator(RunId),
            cancellation=CancellationSource().token,
            event_sink=sink,
            event_id_generator=Uuid4IdGenerator(EventId),
            deadline=Deadline.after(120.0, clock),
            labels={"example_mode": "live"},
        ),
        sink,
    )


async def run_live_example() -> dict[str, object]:
    _require_live_opt_in()
    runtime = CodexAgentRuntime(
        client_factory=CodexSdkClientFactory(),
        model=_model(),
        sandbox=CodexSandboxPolicy(
            mode=CodexSandboxMode.READ_ONLY,
            cwd=str(Path.cwd().resolve()),
            network_access=False,
        ),
        output_schemas={BRIEF_SCHEMA: BRIEF_JSON_SCHEMA},
        ephemeral=True,
    )
    agent = configure_decision_agent(
        runtime,
        runtime_capability_id=CODEX_AGENT_RUNTIME_CAPABILITY.capability_id,
    )
    context, sink = _context()
    result = await agent.run(example_request(), context)
    if not result.is_success:
        failure = result.failure
        if failure is None:
            raise RuntimeError("live decision agent failed without a failure")
        exception_type = failure.metadata.get("exception_type")
        exception_detail = (
            f" exception_type={exception_type}"
            if isinstance(exception_type, str)
            else ""
        )
        raise RuntimeError(
            "live decision agent failed with normalized "
            f"kind={failure.kind.value} code={failure.code or 'none'}"
            f"{exception_detail}"
        )
    brief = result.unwrap()
    return {
        "event_kinds": [event.kind.value for event in sink.events],
        "model": _model(),
        "recommendation": brief.recommendation,
        "risk_count": len(brief.risks),
        "runtime": "openai.codex",
        "summary": brief.summary,
    }


def main() -> None:
    print(json.dumps(asyncio.run(run_live_example()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
