"""Explicit live Codex composition root for bounded coding."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from agentrig.capabilities import DataRetention
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
    CODEX_SHELL_TOOL,
    CodexAgentRuntime,
    CodexApprovalMode,
    CodexSandboxMode,
    CodexSandboxPolicy,
)
from agentrig.integrations.openai.sdk import CodexSdkClientFactory

from examples.capabilities.bounded_coding.scripted import example_task
from examples.capabilities.bounded_coding.workflow import (
    REPORT_JSON_SCHEMA,
    REPORT_SCHEMA,
    configure_runtime_coding_agent,
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
    with tempfile.TemporaryDirectory(prefix="agentrig-bounded-coding-") as raw:
        workspace = Path(raw).resolve()
        runtime = CodexAgentRuntime(
            client_factory=CodexSdkClientFactory(),
            model=_model(),
            sandbox=CodexSandboxPolicy(
                mode=CodexSandboxMode.WORKSPACE_WRITE,
                cwd=str(workspace),
                writable_roots=(str(workspace),),
                network_access=False,
            ),
            output_schemas={REPORT_SCHEMA: REPORT_JSON_SCHEMA},
            approval_mode=CodexApprovalMode.DENY_ALL,
            ephemeral=True,
        )
        agent = configure_runtime_coding_agent(
            runtime,
            runtime_capability_id=(
                CODEX_AGENT_RUNTIME_CAPABILITY.capability_id
            ),
            tool_id=CODEX_SHELL_TOOL,
            coding_capability_id="example.openai.codex.coding",
            coding_capability_version=(
                CODEX_AGENT_RUNTIME_CAPABILITY.version
            ),
            data_retention=DataRetention.PROVIDER_MANAGED,
        )
        context, sink = _context()
        try:
            result = await agent.execute(
                example_task(root_path=str(workspace)),
                context,
            )
        except Exception as error:
            failure = getattr(error, "failure", None)
            if failure is None:
                raise RuntimeError(
                    "live coding agent failed with a sanitized exception"
                ) from error
            exception_type = failure.metadata.get("exception_type")
            exception_detail = (
                f" exception_type={exception_type}"
                if isinstance(exception_type, str)
                else ""
            )
            raise RuntimeError(
                "live coding agent failed with normalized "
                f"kind={failure.kind.value} code={failure.code or 'none'}"
                f"{exception_detail}"
            ) from error

        changed_paths = tuple(change.path for change in result.changed_files)
        if changed_paths != ("greeting.py",):
            raise RuntimeError("live coding agent reported unexpected changes")
        generated = workspace / "greeting.py"
        if not generated.is_file():
            raise RuntimeError("live coding agent did not create its report file")
        actual_files = tuple(
            sorted(
                candidate.relative_to(workspace).as_posix()
                for candidate in workspace.rglob("*")
                if candidate.is_file()
            )
        )
        if actual_files != ("greeting.py",):
            raise RuntimeError("live coding agent changed unexpected files")
        validation = subprocess.run(
            [sys.executable, "-m", "py_compile", str(generated)],
            cwd=workspace,
            capture_output=True,
            check=False,
            text=True,
        )
        if validation.returncode != 0:
            raise RuntimeError("host verification rejected the generated module")
        return {
            "changed_files": list(changed_paths),
            "event_kinds": [event.kind.value for event in sink.events],
            "host_validation": "passed",
            "model": _model(),
            "runtime": "openai.codex",
            "status": result.status.value,
        }


def main() -> None:
    print(json.dumps(asyncio.run(run_live_example()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
