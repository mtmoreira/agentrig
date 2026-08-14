from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agentrig.agents import (
    AgentContract,
    AgentExecutionRequest,
    AgentLimits,
    AgentRuntime,
)
from agentrig.core import (
    CancellationSource,
    Deadline,
    EffectProfile,
    EventId,
    EventKind,
    FailureKind,
    InMemoryEventSink,
    NoOpRedactionPolicy,
    RunContext,
    RunId,
)
from agentrig.integrations.openai import (
    CODEX_AGENT_RUNTIME_CAPABILITY,
    CODEX_SHELL_TOOL,
    CodexAgentRuntime,
    CodexApprovalMode,
    CodexApprovalRequested,
    CodexClient,
    CodexProgressKind,
    CodexProgressReported,
    CodexSandboxMode,
    CodexSandboxPolicy,
    CodexThread,
    CodexThreadRequest,
    CodexToolCallCompleted,
    CodexToolCallStarted,
    CodexTurn,
    CodexTurnCompleted,
    CodexTurnEvent,
    CodexTurnRequest,
    CodexTurnStarted,
    CodexTurnStatus,
    CodexUsageReported,
)


@dataclass(frozen=True)
class FixedClock:
    monotonic_value: float = 100.0

    def now(self) -> datetime:
        return datetime(2026, 8, 14, 18, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return self.monotonic_value


@dataclass
class SequentialIdGenerator:
    prefix: str
    next_value: int = 1

    def generate(self) -> RunId | EventId:
        value = f"{self.prefix}-{self.next_value}"
        self.next_value += 1
        return RunId(value) if self.prefix == "run" else EventId(value)


def create_context(
    *,
    source: CancellationSource | None = None,
    deadline: Deadline | None = None,
) -> tuple[RunContext, InMemoryEventSink]:
    sink = InMemoryEventSink(redaction_policy=NoOpRedactionPolicy())
    owned_source = source if source is not None else CancellationSource()
    return (
        RunContext.create_root(
            clock=FixedClock(),
            id_generator=SequentialIdGenerator("run"),  # type: ignore[arg-type]
            cancellation=owned_source.token,
            event_sink=sink,
            event_id_generator=SequentialIdGenerator("event"),  # type: ignore[arg-type]
            deadline=deadline,
        ),
        sink,
    )


def create_contract(
    *,
    tools: tuple[str, ...] = (),
    capabilities: tuple[str, ...] | None = None,
    permissions: dict[str, str] | None = None,
    max_tool_calls: int = 2,
) -> AgentContract[object, object]:
    return AgentContract(
        agent_id="codex-test",
        version="1",
        purpose="Exercise the Codex runtime",
        input_schema="example.input.v1",
        output_schema="example.output.v1",
        prompt_version="prompt-1",
        effect_profile=EffectProfile.READ_ONLY,
        limits=AgentLimits(max_turns=1, max_tool_calls=max_tool_calls),
        stopping_policy="structured_output",
        allowed_tools=tools,
        allowed_capabilities=(
            capabilities
            if capabilities is not None
            else (CODEX_AGENT_RUNTIME_CAPABILITY.capability_id,)
        ),
        permissions=(
            permissions
            if permissions is not None
            else {"workspace": "read_only", "network": "denied"}
        ),
    )


def create_request(
    *,
    contract: AgentContract[object, object] | None = None,
    provider_options: dict[str, object] | None = None,
) -> AgentExecutionRequest:
    return AgentExecutionRequest(
        contract=contract if contract is not None else create_contract(),
        instructions="Return one structured result.",
        input={"value": "draft"},
        provider_options=provider_options or {},  # type: ignore[arg-type]
    )


@dataclass
class FakeTurn:
    scripted_events: tuple[CodexTurnEvent, ...]
    thread_id: str = "thread-1"
    turn_id: str = "turn-1"
    interrupted: bool = False
    block_after_events: bool = False

    async def events(self) -> AsyncIterator[CodexTurnEvent]:
        for event in self.scripted_events:
            yield event
        if self.block_after_events:
            await asyncio.Event().wait()

    async def interrupt(self) -> None:
        self.interrupted = True


@dataclass
class FakeThread:
    turn: FakeTurn
    thread_id: str = "thread-1"
    requests: list[CodexTurnRequest] = field(default_factory=list)

    async def start_turn(self, request: CodexTurnRequest) -> CodexTurn:
        self.requests.append(request)
        return self.turn


@dataclass
class FakeClient:
    thread: FakeThread
    starts: list[CodexThreadRequest] = field(default_factory=list)
    resumes: list[tuple[str, CodexThreadRequest]] = field(default_factory=list)
    closed: bool = False
    start_error: Exception | None = None

    async def start_thread(self, request: CodexThreadRequest) -> CodexThread:
        self.starts.append(request)
        if self.start_error is not None:
            raise self.start_error
        return self.thread

    async def resume_thread(
        self,
        thread_id: str,
        request: CodexThreadRequest,
    ) -> CodexThread:
        self.resumes.append((thread_id, request))
        return self.thread

    async def close(self) -> None:
        self.closed = True


@dataclass
class FakeFactory:
    client: FakeClient
    calls: int = 0

    def create(self) -> CodexClient:
        self.calls += 1
        return self.client


def create_runtime(
    client: FakeClient,
    *,
    sandbox: CodexSandboxPolicy | None = None,
    approval_mode: CodexApprovalMode = CodexApprovalMode.DENY_ALL,
) -> CodexAgentRuntime:
    return CodexAgentRuntime(
        client_factory=FakeFactory(client),
        model="gpt-5.3-codex",
        sandbox=(
            sandbox
            if sandbox is not None
            else CodexSandboxPolicy(
                mode=CodexSandboxMode.READ_ONLY,
                cwd="/workspace",
            )
        ),
        output_schemas={
            "example.output.v1": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
                "additionalProperties": False,
            }
        },
        approval_mode=approval_mode,
    )


class CodexAgentRuntimeTest(unittest.TestCase):
    def test_streams_safe_lifecycle_and_returns_structured_output(self) -> None:
        turn = FakeTurn(
            scripted_events=(
                CodexTurnStarted(turn_id="turn-1"),
                CodexProgressReported(
                    turn_id="turn-1",
                    kind=CodexProgressKind.PLAN,
                ),
                CodexToolCallStarted(
                    turn_id="turn-1",
                    call_id="call-1",
                    tool_name=CODEX_SHELL_TOOL,
                ),
                CodexToolCallCompleted(
                    turn_id="turn-1",
                    call_id="call-1",
                    tool_name=CODEX_SHELL_TOOL,
                    succeeded=True,
                ),
                CodexUsageReported(
                    turn_id="turn-1",
                    input_tokens=10,
                    cached_input_tokens=3,
                    output_tokens=5,
                ),
                CodexTurnCompleted(
                    turn_id="turn-1",
                    status=CodexTurnStatus.SUCCEEDED,
                    output={"result": "complete"},
                ),
            )
        )
        client = FakeClient(FakeThread(turn))
        runtime = create_runtime(client)
        context, sink = create_context()
        request = create_request(
            contract=create_contract(tools=(CODEX_SHELL_TOOL,))
        )

        result = asyncio.run(runtime.execute(request, context))

        self.assertIsInstance(runtime, AgentRuntime)
        self.assertEqual(result.result.output, {"result": "complete"})
        self.assertEqual(
            result.provider_metadata,
            {
                "provider": "openai.codex",
                "thread_id": "thread-1",
                "turn_id": "turn-1",
            },
        )
        self.assertTrue(client.closed)
        self.assertEqual(client.starts[0].allowed_tools, (CODEX_SHELL_TOOL,))
        self.assertEqual(client.thread.requests[0].prompt, '{"value":"draft"}')
        self.assertEqual(
            [event.kind for event in sink.events],
            [
                EventKind.PROVIDER_CALL_STARTED,
                EventKind.PROGRESS_REPORTED,
                EventKind.TOOL_CALL_STARTED,
                EventKind.TOOL_CALL_COMPLETED,
                EventKind.USAGE_REPORTED,
                EventKind.PROVIDER_CALL_COMPLETED,
            ],
        )
        self.assertNotIn("draft", repr(sink.events))

    def test_resumes_an_explicit_thread_without_expanding_authority(self) -> None:
        turn = FakeTurn(
            scripted_events=(
                CodexTurnCompleted(
                    turn_id="turn-1",
                    status=CodexTurnStatus.SUCCEEDED,
                    output={"result": "resumed"},
                ),
            )
        )
        client = FakeClient(FakeThread(turn))
        runtime = create_runtime(client)
        context, _ = create_context()

        result = asyncio.run(
            runtime.execute(
                create_request(provider_options={"thread_id": "thread-existing"}),
                context,
            )
        )

        self.assertEqual(result.result.output, {"result": "resumed"})
        self.assertEqual(client.starts, [])
        self.assertEqual(client.resumes[0][0], "thread-existing")
        self.assertEqual(client.resumes[0][1].sandbox.mode, CodexSandboxMode.READ_ONLY)

    def test_rejects_contract_authority_before_starting_the_client(self) -> None:
        turn = FakeTurn(scripted_events=())
        client = FakeClient(FakeThread(turn))
        factory = FakeFactory(client)
        runtime = CodexAgentRuntime(
            client_factory=factory,
            model="gpt-5.3-codex",
            sandbox=CodexSandboxPolicy(
                mode=CodexSandboxMode.READ_ONLY,
                cwd="/workspace",
            ),
            output_schemas={"example.output.v1": {"type": "object"}},
        )
        context, sink = create_context()

        missing_capability = asyncio.run(
            runtime.execute(
                create_request(contract=create_contract(capabilities=())),
                context,
            )
        )
        bad_permission = asyncio.run(
            runtime.execute(
                create_request(
                    contract=create_contract(
                        permissions={"workspace": "read_write", "network": "denied"}
                    )
                ),
                context,
            )
        )

        self.assertEqual(
            missing_capability.result.failure.kind,  # type: ignore[union-attr]
            FailureKind.INVALID_INPUT,
        )
        self.assertEqual(
            bad_permission.result.failure.code,  # type: ignore[union-attr]
            "codex.workspace_permission_mismatch",
        )
        self.assertEqual(factory.calls, 0)
        self.assertEqual(sink.events, ())

    def test_cancellation_interrupts_the_turn_and_closes_the_client(self) -> None:
        source = CancellationSource()
        turn = FakeTurn(scripted_events=(), block_after_events=True)
        client = FakeClient(FakeThread(turn))
        runtime = create_runtime(client)
        context, _ = create_context(source=source)

        async def run_and_cancel() -> object:
            task = asyncio.create_task(runtime.execute(create_request(), context))
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            source.cancel("caller stopped")
            return await task

        result = asyncio.run(run_and_cancel())

        self.assertEqual(result.result.failure.kind, FailureKind.CANCELLED)  # type: ignore[attr-defined,union-attr]
        self.assertEqual(result.result.failure.message, "caller stopped")  # type: ignore[attr-defined,union-attr]
        self.assertTrue(turn.interrupted)
        self.assertTrue(client.closed)

    def test_expired_deadline_prevents_client_construction(self) -> None:
        turn = FakeTurn(scripted_events=())
        client = FakeClient(FakeThread(turn))
        factory = FakeFactory(client)
        runtime = CodexAgentRuntime(
            client_factory=factory,
            model="gpt-5.3-codex",
            sandbox=CodexSandboxPolicy(
                mode=CodexSandboxMode.READ_ONLY,
                cwd="/workspace",
            ),
            output_schemas={"example.output.v1": {"type": "object"}},
        )
        deadline = Deadline(
            expires_at=datetime(2026, 8, 14, 18, 0, tzinfo=UTC),
            monotonic_deadline=100.0,
        )
        context, _ = create_context(deadline=deadline)

        result = asyncio.run(runtime.execute(create_request(), context))

        self.assertEqual(
            result.result.failure.kind,  # type: ignore[union-attr]
            FailureKind.DEADLINE_EXCEEDED,
        )
        self.assertEqual(factory.calls, 0)

    def test_manual_approval_blocks_and_deny_all_fails(self) -> None:
        events = (
            CodexApprovalRequested(
                turn_id="turn-1",
                approval_id="call-1",
                tool_name=CODEX_SHELL_TOOL,
            ),
            CodexTurnCompleted(
                turn_id="turn-1",
                status=CodexTurnStatus.FAILED,
                error_code="codex.failed",
            ),
        )
        context, _ = create_context()
        manual = asyncio.run(
            create_runtime(
                FakeClient(FakeThread(FakeTurn(events))),
                approval_mode=CodexApprovalMode.MANUAL,
            ).execute(
                create_request(contract=create_contract(tools=(CODEX_SHELL_TOOL,))),
                context,
            )
        )
        context, _ = create_context()
        denied = asyncio.run(
            create_runtime(
                FakeClient(FakeThread(FakeTurn(events))),
                approval_mode=CodexApprovalMode.DENY_ALL,
            ).execute(
                create_request(contract=create_contract(tools=(CODEX_SHELL_TOOL,))),
                context,
            )
        )

        self.assertEqual(manual.result.failure.kind, FailureKind.APPROVAL_REQUIRED)  # type: ignore[union-attr]
        self.assertEqual(denied.result.failure.kind, FailureKind.APPROVAL_DENIED)  # type: ignore[union-attr]

    def test_maps_safe_provider_failures_and_sanitizes_transport_errors(self) -> None:
        context, _ = create_context()
        overloaded = asyncio.run(
            create_runtime(
                FakeClient(
                    FakeThread(
                        FakeTurn(
                            (
                                CodexTurnCompleted(
                                    turn_id="turn-1",
                                    status=CodexTurnStatus.FAILED,
                                    error_code="codex.server_overloaded",
                                ),
                            )
                        )
                    )
                )
            ).execute(create_request(), context)
        )
        context, _ = create_context()
        transport = asyncio.run(
            create_runtime(
                FakeClient(
                    FakeThread(FakeTurn(())),
                    start_error=RuntimeError("password=private"),
                )
            ).execute(create_request(), context)
        )

        self.assertEqual(
            overloaded.result.failure.kind,  # type: ignore[union-attr]
            FailureKind.TRANSIENT_PROVIDER,
        )
        self.assertEqual(transport.result.failure.code, "codex.transport_failed")  # type: ignore[union-attr]
        self.assertNotIn("private", transport.result.failure.message)  # type: ignore[union-attr]

    def test_interrupts_when_provider_exceeds_tool_authority(self) -> None:
        turn = FakeTurn(
            scripted_events=(
                CodexToolCallStarted(
                    turn_id="turn-1",
                    call_id="call-1",
                    tool_name=CODEX_SHELL_TOOL,
                ),
            ),
            block_after_events=True,
        )
        context, _ = create_context()

        result = asyncio.run(
            create_runtime(FakeClient(FakeThread(turn))).execute(
                create_request(),
                context,
            )
        )

        self.assertEqual(result.result.failure.code, "codex.tool_authority_exceeded")  # type: ignore[union-attr]
        self.assertTrue(turn.interrupted)


if __name__ == "__main__":
    unittest.main()
