"""Positive fixture for the injected Codex client boundary."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from agentrig.integrations.openai import (
    CodexApprovalMode,
    CodexClient,
    CodexClientFactory,
    CodexSandboxMode,
    CodexSandboxPolicy,
    CodexThread,
    CodexThreadRequest,
    CodexTurn,
    CodexTurnCompleted,
    CodexTurnEvent,
    CodexTurnRequest,
    CodexTurnStatus,
)


@dataclass(frozen=True)
class FakeTurn:
    thread_id: str
    turn_id: str

    def events(self) -> AsyncIterator[CodexTurnEvent]:
        return _events(self.turn_id)

    async def interrupt(self) -> None:
        return None


async def _events(turn_id: str) -> AsyncIterator[CodexTurnEvent]:
    yield CodexTurnCompleted(
        turn_id=turn_id,
        status=CodexTurnStatus.SUCCEEDED,
        output={"answer": "complete"},
    )


@dataclass(frozen=True)
class FakeThread:
    thread_id: str

    async def start_turn(self, request: CodexTurnRequest) -> CodexTurn:
        del request
        return FakeTurn(thread_id=self.thread_id, turn_id="turn-1")


class FakeClient:
    async def start_thread(self, request: CodexThreadRequest) -> CodexThread:
        del request
        return FakeThread(thread_id="thread-1")

    async def resume_thread(
        self,
        thread_id: str,
        request: CodexThreadRequest,
    ) -> CodexThread:
        del request
        return FakeThread(thread_id=thread_id)

    async def close(self) -> None:
        return None


@dataclass(frozen=True)
class FakeClientFactory:
    def create(self) -> CodexClient:
        return FakeClient()


sandbox = CodexSandboxPolicy(
    mode=CodexSandboxMode.READ_ONLY,
    cwd="/workspace/project",
)
thread_request = CodexThreadRequest(
    model="gpt-codex",
    instructions="Return one structured result.",
    sandbox=sandbox,
    approval_mode=CodexApprovalMode.DENY_ALL,
)
turn_request = CodexTurnRequest(
    prompt="Run the bounded task.",
    output_schema={"type": "object"},
    sandbox=sandbox,
    approval_mode=CodexApprovalMode.DENY_ALL,
)
factory: CodexClientFactory = FakeClientFactory()
client: CodexClient = factory.create()
