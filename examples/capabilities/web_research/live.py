"""Explicit live Codex composition root for read-only web research."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from agentrig.capabilities import DataRetention, SearchRequest
from agentrig.core import (
    CancellationSource,
    Deadline,
    EventId,
    EventKind,
    InMemoryEventSink,
    RunContext,
    RunId,
    SystemClock,
    Uuid4IdGenerator,
)
from agentrig.integrations.openai import (
    CODEX_AGENT_RUNTIME_CAPABILITY,
    CODEX_WEB_SEARCH_TOOL,
    CodexAgentRuntime,
    CodexApprovalMode,
    CodexSandboxMode,
    CodexSandboxPolicy,
)
from agentrig.integrations.openai.sdk import CodexSdkClientFactory

from examples.capabilities.web_research.workflow import (
    REPORT_JSON_SCHEMA,
    REPORT_SCHEMA,
    configure_runtime_search_provider,
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
            network_access=True,
        ),
        output_schemas={REPORT_SCHEMA: REPORT_JSON_SCHEMA},
        approval_mode=CodexApprovalMode.DENY_ALL,
        ephemeral=True,
    )
    provider = configure_runtime_search_provider(
        runtime,
        runtime_capability_id=CODEX_AGENT_RUNTIME_CAPABILITY.capability_id,
        web_search_tool_id=CODEX_WEB_SEARCH_TOOL,
        search_capability_id="example.openai.codex.search",
        search_capability_version=CODEX_AGENT_RUNTIME_CAPABILITY.version,
        data_retention=DataRetention.PROVIDER_MANAGED,
    )
    context, sink = _context()
    try:
        result = await provider.search(
            SearchRequest(
                query="official OpenAI Codex app server documentation",
                max_results=2,
            ),
            context,
        )
    except Exception as error:
        failure = getattr(error, "failure", None)
        if failure is None:
            raise RuntimeError(
                "live research provider failed with a sanitized exception"
            ) from error
        exception_type = failure.metadata.get("exception_type")
        exception_detail = (
            f" exception_type={exception_type}"
            if isinstance(exception_type, str)
            else ""
        )
        raise RuntimeError(
            "live research provider failed with normalized "
            f"kind={failure.kind.value} code={failure.code or 'none'}"
            f"{exception_detail}"
        ) from error
    event_kinds = tuple(event.kind for event in sink.events)
    if (
        EventKind.TOOL_CALL_STARTED not in event_kinds
        or EventKind.TOOL_CALL_COMPLETED not in event_kinds
    ):
        raise RuntimeError("live research provider did not use web search")
    return {
        "citations": [
            {
                "source_uri": citation.source_uri,
                "title": citation.title,
            }
            for citation in result.citations
        ],
        "event_kinds": [event_kind.value for event_kind in event_kinds],
        "model": _model(),
        "result_count": len(result.hits),
        "runtime": "openai.codex",
    }


def main() -> None:
    print(json.dumps(asyncio.run(run_live_example()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
