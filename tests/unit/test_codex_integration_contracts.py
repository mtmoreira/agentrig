from __future__ import annotations

import asyncio
import math
import unittest
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from agentrig.capabilities import (
    CapabilityFeature,
    CapabilityKind,
    DataRetention,
)
from agentrig.integrations.openai import (
    CODEX_AGENT_RUNTIME_CAPABILITY,
    CODEX_SHELL_TOOL,
    CODEX_SDK_VERSION,
    CODEX_WEB_SEARCH_TOOL,
    CodexApprovalMode,
    CodexApprovalRequested,
    CodexClient,
    CodexClientFactory,
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


def read_only_sandbox() -> CodexSandboxPolicy:
    return CodexSandboxPolicy(
        mode=CodexSandboxMode.READ_ONLY,
        cwd="/workspace/project",
    )


def write_sandbox() -> CodexSandboxPolicy:
    return CodexSandboxPolicy(
        mode=CodexSandboxMode.WORKSPACE_WRITE,
        cwd="/workspace/project",
        writable_roots=("/workspace/project", "/workspace/shared"),
    )


def create_thread_request() -> CodexThreadRequest:
    return CodexThreadRequest(
        model="gpt-codex",
        instructions="Return one bounded structured response.",
        sandbox=read_only_sandbox(),
        approval_mode=CodexApprovalMode.DENY_ALL,
    )


def create_turn_request() -> CodexTurnRequest:
    return CodexTurnRequest(
        prompt="Execute the bounded request.",
        output_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
        sandbox=read_only_sandbox(),
        approval_mode=CodexApprovalMode.DENY_ALL,
    )


@dataclass
class FakeTurn:
    thread_id: str
    turn_id: str
    scripted_events: tuple[CodexTurnEvent, ...]
    interruptions: int = 0

    def events(self) -> AsyncIterator[CodexTurnEvent]:
        return self._events()

    async def _events(self) -> AsyncIterator[CodexTurnEvent]:
        for event in self.scripted_events:
            yield event

    async def interrupt(self) -> None:
        self.interruptions += 1


@dataclass
class FakeThread:
    thread_id: str
    turn: FakeTurn
    requests: list[CodexTurnRequest] = field(default_factory=list)

    async def start_turn(self, request: CodexTurnRequest) -> CodexTurn:
        self.requests.append(request)
        return self.turn


@dataclass
class FakeClient:
    thread: FakeThread
    starts: list[CodexThreadRequest] = field(default_factory=list)
    resumes: list[tuple[str, CodexThreadRequest]] = field(default_factory=list)
    closes: int = 0

    async def start_thread(self, request: CodexThreadRequest) -> CodexThread:
        self.starts.append(request)
        return self.thread

    async def resume_thread(
        self,
        thread_id: str,
        request: CodexThreadRequest,
    ) -> CodexThread:
        self.resumes.append((thread_id, request))
        return self.thread

    async def close(self) -> None:
        self.closes += 1


@dataclass(frozen=True)
class FakeFactory:
    client: FakeClient

    def create(self) -> CodexClient:
        return self.client


class CodexCapabilityTest(unittest.TestCase):
    def test_declares_the_pinned_runtime_surface(self) -> None:
        descriptor = CODEX_AGENT_RUNTIME_CAPABILITY

        self.assertEqual(CODEX_SDK_VERSION, "0.144.4")
        self.assertEqual(descriptor.capability_id, "openai.codex.agent_runtime")
        self.assertEqual(descriptor.version, CODEX_SDK_VERSION)
        self.assertEqual(descriptor.kind, CapabilityKind.CODING)
        self.assertEqual(
            descriptor.features,
            frozenset(
                {
                    CapabilityFeature.STREAMING,
                    CapabilityFeature.CANCELLATION,
                    CapabilityFeature.STRUCTURED_OUTPUT,
                    CapabilityFeature.SESSION_CONTINUATION,
                    CapabilityFeature.APPROVAL_REQUESTS,
                    CapabilityFeature.TOOL_USE,
                }
            ),
        )
        self.assertEqual(
            descriptor.data_retention,
            DataRetention.PROVIDER_MANAGED,
        )


class CodexSandboxPolicyTest(unittest.TestCase):
    def test_preserves_explicit_bounded_authority(self) -> None:
        read_only = read_only_sandbox()
        writable = write_sandbox()

        self.assertEqual(read_only.writable_roots, ())
        self.assertFalse(read_only.network_access)
        self.assertEqual(
            writable.writable_roots,
            ("/workspace/project", "/workspace/shared"),
        )

    def test_rejects_unbounded_or_implicit_write_authority(self) -> None:
        invalid_paths = (
            "relative/project",
            "/",
            "//workspace/project",
            "/workspace/../private",
            "/workspace/project/",
            "C:\\workspace",
        )
        for path in invalid_paths:
            with self.subTest(path=path), self.assertRaises(ValueError):
                CodexSandboxPolicy(
                    mode=CodexSandboxMode.READ_ONLY,
                    cwd=path,
                )

        with self.assertRaises(ValueError):
            CodexSandboxPolicy(
                mode=CodexSandboxMode.READ_ONLY,
                cwd="/workspace/project",
                writable_roots=("/workspace/project",),
            )
        with self.assertRaises(ValueError):
            CodexSandboxPolicy(
                mode=CodexSandboxMode.WORKSPACE_WRITE,
                cwd="/workspace/project",
            )
        with self.assertRaises(ValueError):
            CodexSandboxPolicy(
                mode=CodexSandboxMode.WORKSPACE_WRITE,
                cwd="/workspace/project",
                writable_roots=("/workspace/other",),
            )
        with self.assertRaises(ValueError):
            CodexSandboxPolicy(
                mode=CodexSandboxMode.WORKSPACE_WRITE,
                cwd="/workspace/project",
                writable_roots=(
                    "/workspace/project",
                    "/workspace/project",
                ),
            )
        with self.assertRaises(TypeError):
            CodexSandboxPolicy(
                mode=CodexSandboxMode.READ_ONLY,
                cwd="/workspace/project",
                network_access=1,  # type: ignore[arg-type]
            )


class CodexRequestTest(unittest.TestCase):
    def test_requires_explicit_thread_and_turn_configuration(self) -> None:
        thread = create_thread_request()
        turn = create_turn_request()

        self.assertTrue(thread.ephemeral)
        self.assertEqual(thread.service_name, "agentrig")
        self.assertIs(thread.approval_mode, CodexApprovalMode.DENY_ALL)
        self.assertEqual(thread.allowed_tools, ())
        self.assertIs(turn.sandbox.mode, CodexSandboxMode.READ_ONLY)
        self.assertEqual(turn.output_schema["required"], ("answer",))
        with self.assertRaises(TypeError):
            turn.output_schema["type"] = "array"  # type: ignore[index]

    def test_rejects_invalid_or_unsafe_request_values(self) -> None:
        with self.assertRaises(ValueError):
            CodexThreadRequest(
                model=" padded ",
                instructions="Run",
                sandbox=read_only_sandbox(),
                approval_mode=CodexApprovalMode.DENY_ALL,
            )
        with self.assertRaises(TypeError):
            CodexThreadRequest(
                model="gpt-codex",
                instructions="Run",
                sandbox=read_only_sandbox(),
                approval_mode="never",  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            CodexThreadRequest(
                model="gpt-codex",
                instructions="Run",
                sandbox=read_only_sandbox(),
                approval_mode=CodexApprovalMode.DENY_ALL,
                allowed_tools=("unsupported",),
            )
        with self.assertRaises(ValueError):
            CodexThreadRequest(
                model="gpt-codex",
                instructions="Run",
                sandbox=read_only_sandbox(),
                approval_mode=CodexApprovalMode.DENY_ALL,
                allowed_tools=(CODEX_SHELL_TOOL, CODEX_SHELL_TOOL),
            )
        with self.assertRaises(ValueError):
            CodexThreadRequest(
                model="gpt-codex",
                instructions="Run",
                sandbox=read_only_sandbox(),
                approval_mode=CodexApprovalMode.DENY_ALL,
                allowed_tools=(CODEX_WEB_SEARCH_TOOL,),
            )
        with self.assertRaises(ValueError):
            CodexTurnRequest(
                prompt="Run",
                output_schema={},
                sandbox=read_only_sandbox(),
                approval_mode=CodexApprovalMode.DENY_ALL,
            )
        with self.assertRaises(ValueError):
            CodexTurnRequest(
                prompt="Run",
                output_schema={"minimum": math.inf},
                sandbox=read_only_sandbox(),
                approval_mode=CodexApprovalMode.DENY_ALL,
            )


class CodexEventTest(unittest.TestCase):
    def test_models_only_allowlisted_lifecycle_and_usage_data(self) -> None:
        events: tuple[CodexTurnEvent, ...] = (
            CodexTurnStarted(turn_id="turn-1"),
            CodexProgressReported(
                turn_id="turn-1",
                kind=CodexProgressKind.PLAN,
            ),
            CodexToolCallStarted(
                turn_id="turn-1",
                call_id="call-1",
                tool_name="shell",
            ),
            CodexToolCallCompleted(
                turn_id="turn-1",
                call_id="call-1",
                tool_name="shell",
                succeeded=True,
            ),
            CodexApprovalRequested(
                turn_id="turn-1",
                approval_id="approval-1",
                tool_name="shell",
            ),
            CodexUsageReported(
                turn_id="turn-1",
                input_tokens=13,
                cached_input_tokens=5,
                output_tokens=8,
            ),
            CodexTurnCompleted(
                turn_id="turn-1",
                status=CodexTurnStatus.SUCCEEDED,
                output={"answer": ["complete"]},
            ),
        )

        self.assertEqual(len(events), 7)
        completion = events[-1]
        self.assertIsInstance(completion, CodexTurnCompleted)
        if not isinstance(completion, CodexTurnCompleted):
            raise AssertionError("final event is not a CodexTurnCompleted")
        self.assertEqual(completion.output, {"answer": ("complete",)})
        self.assertNotIn("prompt", repr(events[:-1]))
        self.assertNotIn("arguments", repr(events[:-1]))

    def test_rejects_impossible_terminal_and_usage_values(self) -> None:
        with self.assertRaises(ValueError):
            CodexUsageReported(
                turn_id="turn-1",
                input_tokens=2,
                cached_input_tokens=3,
                output_tokens=1,
            )
        with self.assertRaises(ValueError):
            CodexUsageReported(
                turn_id="turn-1",
                input_tokens=-1,
                cached_input_tokens=0,
                output_tokens=1,
            )
        with self.assertRaises(ValueError):
            CodexTurnCompleted(
                turn_id="turn-1",
                status=CodexTurnStatus.SUCCEEDED,
                output={"answer": "complete"},
                error_code="provider.failed",
            )
        with self.assertRaises(ValueError):
            CodexTurnCompleted(
                turn_id="turn-1",
                status=CodexTurnStatus.FAILED,
                output={"answer": "unreachable"},
            )


class CodexClientContractTest(unittest.TestCase):
    def test_injected_protocol_supports_start_resume_stream_and_interrupt(self) -> None:
        completion = CodexTurnCompleted(
            turn_id="turn-1",
            status=CodexTurnStatus.SUCCEEDED,
            output={"answer": "complete"},
        )
        turn = FakeTurn(
            thread_id="thread-1",
            turn_id="turn-1",
            scripted_events=(completion,),
        )
        thread = FakeThread(thread_id="thread-1", turn=turn)
        client = FakeClient(thread=thread)
        factory = FakeFactory(client=client)

        async def exercise() -> tuple[CodexTurnEvent, ...]:
            created = factory.create()
            started = await created.start_thread(create_thread_request())
            resumed = await created.resume_thread(
                "thread-1",
                create_thread_request(),
            )
            self.assertIs(started, resumed)
            active = await started.start_turn(create_turn_request())
            await active.interrupt()
            events = tuple([event async for event in active.events()])
            await created.close()
            return events

        events = asyncio.run(exercise())

        self.assertIsInstance(factory, CodexClientFactory)
        self.assertIsInstance(client, CodexClient)
        self.assertIsInstance(thread, CodexThread)
        self.assertIsInstance(turn, CodexTurn)
        self.assertEqual(events, (completion,))
        self.assertEqual(len(client.starts), 1)
        self.assertEqual(len(client.resumes), 1)
        self.assertEqual(thread.requests, [create_turn_request()])
        self.assertEqual(turn.interruptions, 1)
        self.assertEqual(client.closes, 1)


if __name__ == "__main__":
    unittest.main()
