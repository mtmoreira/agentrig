"""Optional openai-codex SDK bridge for AgentRig's injected contracts."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from queue import Empty, Queue
from typing import Any, Protocol, cast

from openai_codex.client import CodexClient as RawCodexClient
from openai_codex.client import CodexConfig
from openai_codex.generated.v2_all import (
    AgentMessageThreadItem,
    AbsolutePathBuf,
    AskForApproval,
    AskForApprovalValue,
    CodexErrorInfoValue,
    CommandExecutionStatus,
    CommandExecutionThreadItem,
    ContextCompactionThreadItem,
    FileChangeThreadItem,
    ItemCompletedNotification,
    ItemStartedNotification,
    MessagePhase,
    PatchApplyStatus,
    PlanThreadItem,
    ReadOnlySandboxPolicy,
    ReasoningThreadItem,
    SandboxMode,
    SandboxPolicy,
    ThreadResumeParams,
    ThreadStartParams,
    ThreadTokenUsageUpdatedNotification,
    TurnCompletedNotification,
    TurnStartParams,
    TurnStartedNotification,
    TurnStatus,
    WebSearchThreadItem,
    WorkspaceWriteSandboxPolicy,
)
from openai_codex.models import JsonObject, Notification

from agentrig.core._json import JsonValue, thaw_json_value
from agentrig.integrations.openai.codex import (
    CODEX_SHELL_TOOL,
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

_COMMAND_APPROVAL = "item/commandExecution/requestApproval"
_FILE_APPROVAL = "item/fileChange/requestApproval"
_PERMISSIONS_APPROVAL = "item/permissions/requestApproval"
_MCP_ELICITATION = "mcpServer/elicitation/request"


class _RawClient(Protocol):
    def start(self) -> None: ...
    def initialize(self) -> object: ...
    def close(self) -> None: ...
    def thread_start(self, params: Any) -> Any: ...
    def thread_resume(self, thread_id: str, params: Any) -> Any: ...
    def turn_start(
        self,
        thread_id: str,
        input_items: str,
        params: Any,
    ) -> Any: ...
    def turn_interrupt(self, thread_id: str, turn_id: str) -> Any: ...
    def next_turn_notification(self, turn_id: str) -> Notification: ...
    def unregister_turn_notifications(self, turn_id: str) -> None: ...


RawClientBuilder = Callable[
    [CodexConfig, Callable[[str, JsonObject | None], JsonObject]],
    _RawClient,
]


class CodexSdkClientFactory:
    """Create stable SDK-backed clients with experimental APIs disabled."""

    def __init__(self, *, raw_client_builder: RawClientBuilder | None = None) -> None:
        self._builder = raw_client_builder or _default_raw_client_builder

    def create(self) -> CodexClient:
        return _SdkClient(self._builder)


class _SdkClient:
    def __init__(self, builder: RawClientBuilder) -> None:
        self._approvals: Queue[CodexApprovalRequested] = Queue()
        config = CodexConfig(
            experimental_api=False,
            config_overrides=(
                "features.shell_tool=false",
                'web_search="disabled"',
                "tools.view_image=false",
                "agents.enabled=false",
                "features.skill_mcp_dependency_install=false",
                'history.persistence="none"',
            ),
            client_name="agentrig",
            client_title="AgentRig",
        )
        self._raw = builder(config, self._handle_approval)
        self._started = False
        self._start_lock = asyncio.Lock()

    async def start_thread(self, request: CodexThreadRequest) -> CodexThread:
        await self._ensure_started()
        response = await asyncio.to_thread(
            self._raw.thread_start,
            ThreadStartParams(
                model=request.model,
                base_instructions=request.instructions,
                cwd=request.sandbox.cwd,
                sandbox=_sandbox_mode(request.sandbox),
                approval_policy=_approval_policy(request.approval_mode),
                ephemeral=request.ephemeral,
                service_name=request.service_name,
                config=_thread_config(request),
            ),
        )
        thread_id = _response_thread_id(response)
        return _SdkThread(self, thread_id)

    async def resume_thread(
        self,
        thread_id: str,
        request: CodexThreadRequest,
    ) -> CodexThread:
        await self._ensure_started()
        response = await asyncio.to_thread(
            self._raw.thread_resume,
            thread_id,
            ThreadResumeParams(
                thread_id=thread_id,
                model=request.model,
                base_instructions=request.instructions,
                cwd=request.sandbox.cwd,
                sandbox=_sandbox_mode(request.sandbox),
                approval_policy=_approval_policy(request.approval_mode),
                config=_thread_config(request),
            ),
        )
        return _SdkThread(self, _response_thread_id(response))

    async def close(self) -> None:
        if self._started:
            await asyncio.to_thread(self._raw.close)
            self._started = False

    async def _ensure_started(self) -> None:
        async with self._start_lock:
            if self._started:
                return
            await asyncio.to_thread(self._raw.start)
            await asyncio.to_thread(self._raw.initialize)
            self._started = True

    async def _start_turn(
        self,
        thread_id: str,
        request: CodexTurnRequest,
    ) -> CodexTurn:
        response = await asyncio.to_thread(
            self._raw.turn_start,
            thread_id,
            request.prompt,
            TurnStartParams(
                thread_id=thread_id,
                input=[],
                output_schema=cast(Any, thaw_json_value(request.output_schema)),
                sandbox_policy=_sandbox_policy(request.sandbox),
                approval_policy=_approval_policy(request.approval_mode),
            ),
        )
        turn_id = _response_turn_id(response)
        return _SdkTurn(self, thread_id, turn_id)

    def _handle_approval(
        self,
        method: str,
        params: JsonObject | None,
    ) -> JsonObject:
        values = params or {}
        turn_id = values.get("turnId")
        item_id = values.get("itemId")
        if method in {_COMMAND_APPROVAL, _FILE_APPROVAL}:
            if isinstance(turn_id, str) and isinstance(item_id, str):
                self._approvals.put(
                    CodexApprovalRequested(
                        turn_id=turn_id,
                        approval_id=item_id,
                        tool_name=CODEX_SHELL_TOOL,
                    )
                )
            return {"decision": "decline"}
        if method == _PERMISSIONS_APPROVAL:
            return {"permissions": []}
        if method == _MCP_ELICITATION:
            return {"action": "decline", "content": None}
        return {}

    def _take_approvals(self, turn_id: str) -> tuple[CodexApprovalRequested, ...]:
        matching: list[CodexApprovalRequested] = []
        deferred: list[CodexApprovalRequested] = []
        while True:
            try:
                approval = self._approvals.get_nowait()
            except Empty:
                break
            if approval.turn_id == turn_id:
                matching.append(approval)
            else:
                deferred.append(approval)
        for approval in deferred:
            self._approvals.put(approval)
        return tuple(matching)


class _SdkThread:
    def __init__(self, client: _SdkClient, thread_id: str) -> None:
        self._client = client
        self._thread_id = thread_id

    @property
    def thread_id(self) -> str:
        return self._thread_id

    async def start_turn(self, request: CodexTurnRequest) -> CodexTurn:
        return await self._client._start_turn(self._thread_id, request)


class _SdkTurn:
    def __init__(self, client: _SdkClient, thread_id: str, turn_id: str) -> None:
        self._client = client
        self._thread_id = thread_id
        self._turn_id = turn_id
        self._final_response: str | None = None

    @property
    def thread_id(self) -> str:
        return self._thread_id

    @property
    def turn_id(self) -> str:
        return self._turn_id

    async def interrupt(self) -> None:
        await asyncio.to_thread(
            self._client._raw.turn_interrupt,
            self._thread_id,
            self._turn_id,
        )

    async def events(self) -> AsyncIterator[CodexTurnEvent]:
        try:
            while True:
                for approval in self._client._take_approvals(self._turn_id):
                    yield approval
                notification = await asyncio.to_thread(
                    self._client._raw.next_turn_notification,
                    self._turn_id,
                )
                for approval in self._client._take_approvals(self._turn_id):
                    yield approval
                translated = self._translate(notification)
                for event in translated:
                    yield event
                if any(isinstance(event, CodexTurnCompleted) for event in translated):
                    break
        finally:
            await asyncio.to_thread(
                self._client._raw.unregister_turn_notifications,
                self._turn_id,
            )

    def _translate(self, notification: Notification) -> tuple[CodexTurnEvent, ...]:
        payload = notification.payload
        if isinstance(payload, TurnStartedNotification):
            return (CodexTurnStarted(turn_id=self._turn_id),)
        if isinstance(payload, ItemStartedNotification):
            return self._translate_item(payload.item.root, started=True)
        if isinstance(payload, ItemCompletedNotification):
            item = payload.item.root
            if isinstance(item, AgentMessageThreadItem):
                if item.phase in (None, MessagePhase.final_answer):
                    self._final_response = item.text
                return ()
            return self._translate_item(item, started=False)
        if isinstance(payload, ThreadTokenUsageUpdatedNotification):
            usage = payload.token_usage.last
            return (
                CodexUsageReported(
                    turn_id=self._turn_id,
                    input_tokens=usage.input_tokens,
                    cached_input_tokens=usage.cached_input_tokens,
                    output_tokens=usage.output_tokens,
                ),
            )
        if isinstance(payload, TurnCompletedNotification):
            return (self._translate_terminal(payload),)
        return ()

    def _translate_item(
        self,
        item: object,
        *,
        started: bool,
    ) -> tuple[CodexTurnEvent, ...]:
        progress_kind: CodexProgressKind | None = None
        if isinstance(item, PlanThreadItem):
            progress_kind = CodexProgressKind.PLAN
        elif isinstance(item, ReasoningThreadItem):
            progress_kind = CodexProgressKind.REASONING
        elif isinstance(item, ContextCompactionThreadItem):
            progress_kind = CodexProgressKind.CONTEXT_COMPACTION
        if progress_kind is not None:
            if not started:
                return ()
            return (
                CodexProgressReported(
                    turn_id=self._turn_id,
                    kind=progress_kind,
                ),
            )

        if isinstance(item, (CommandExecutionThreadItem, FileChangeThreadItem)):
            succeeded = (
                item.status is CommandExecutionStatus.completed
                if isinstance(item, CommandExecutionThreadItem)
                else item.status is PatchApplyStatus.completed
            )
            return self._tool_event(
                call_id=item.id,
                tool_name=CODEX_SHELL_TOOL,
                started=started,
                succeeded=succeeded,
            )
        if isinstance(item, WebSearchThreadItem):
            return self._tool_event(
                call_id=item.id,
                tool_name=CODEX_WEB_SEARCH_TOOL,
                started=started,
                succeeded=True,
            )
        return ()

    def _tool_event(
        self,
        *,
        call_id: str,
        tool_name: str,
        started: bool,
        succeeded: bool,
    ) -> tuple[CodexTurnEvent, ...]:
        if started:
            return (
                CodexToolCallStarted(
                    turn_id=self._turn_id,
                    call_id=call_id,
                    tool_name=tool_name,
                ),
            )
        return (
            CodexToolCallCompleted(
                turn_id=self._turn_id,
                call_id=call_id,
                tool_name=tool_name,
                succeeded=succeeded,
            ),
        )

    def _translate_terminal(
        self,
        payload: TurnCompletedNotification,
    ) -> CodexTurnCompleted:
        turn = payload.turn
        if turn.status is TurnStatus.interrupted:
            return CodexTurnCompleted(
                turn_id=self._turn_id,
                status=CodexTurnStatus.INTERRUPTED,
                error_code="codex.interrupted",
            )
        if turn.status is TurnStatus.failed:
            error_code = _safe_error_code(turn.error)
            status = (
                CodexTurnStatus.REFUSED
                if error_code == "codex.cyber_policy"
                else CodexTurnStatus.FAILED
            )
            return CodexTurnCompleted(
                turn_id=self._turn_id,
                status=status,
                error_code=error_code,
            )
        if turn.status is not TurnStatus.completed:
            return CodexTurnCompleted(
                turn_id=self._turn_id,
                status=CodexTurnStatus.FAILED,
                error_code="codex.invalid_terminal_status",
            )
        try:
            output = json.loads(self._final_response or "")
        except (json.JSONDecodeError, TypeError):
            return CodexTurnCompleted(
                turn_id=self._turn_id,
                status=CodexTurnStatus.INVALID_OUTPUT,
                error_code="codex.invalid_output",
            )
        try:
            return CodexTurnCompleted(
                turn_id=self._turn_id,
                status=CodexTurnStatus.SUCCEEDED,
                output=cast(JsonValue, output),
            )
        except (TypeError, ValueError):
            return CodexTurnCompleted(
                turn_id=self._turn_id,
                status=CodexTurnStatus.INVALID_OUTPUT,
                error_code="codex.invalid_output",
            )


def _default_raw_client_builder(
    config: CodexConfig,
    approval_handler: Callable[[str, JsonObject | None], JsonObject],
) -> _RawClient:
    return RawCodexClient(config=config, approval_handler=approval_handler)


def _thread_config(request: CodexThreadRequest) -> dict[str, Any]:
    tools = set(request.allowed_tools)
    return {
        "features": {
            "shell_tool": CODEX_SHELL_TOOL in tools,
            "skill_mcp_dependency_install": False,
        },
        "web_search": (
            "live" if CODEX_WEB_SEARCH_TOOL in tools else "disabled"
        ),
        "tools": {"view_image": False},
        "agents": {"enabled": False},
        "mcp_servers": {},
        "hooks": {},
        "history": {"persistence": "none"},
    }


def _sandbox_mode(policy: CodexSandboxPolicy) -> SandboxMode:
    return (
        SandboxMode.read_only
        if policy.mode is CodexSandboxMode.READ_ONLY
        else SandboxMode.workspace_write
    )


def _sandbox_policy(policy: CodexSandboxPolicy) -> SandboxPolicy:
    if policy.mode is CodexSandboxMode.READ_ONLY:
        return SandboxPolicy(
            root=ReadOnlySandboxPolicy(
                type="readOnly",
                network_access=policy.network_access,
            )
        )
    return SandboxPolicy(
        root=WorkspaceWriteSandboxPolicy(
            type="workspaceWrite",
            network_access=policy.network_access,
            writable_roots=[
                AbsolutePathBuf(root=root) for root in policy.writable_roots
            ],
            exclude_slash_tmp=True,
            exclude_tmpdir_env_var=True,
        )
    )


def _approval_policy(mode: CodexApprovalMode) -> AskForApproval:
    value = (
        AskForApprovalValue.never
        if mode is CodexApprovalMode.DENY_ALL
        else AskForApprovalValue.on_request
    )
    return AskForApproval(root=value)


def _response_thread_id(response: object) -> str:
    thread = getattr(response, "thread", None)
    thread_id = getattr(thread, "id", None)
    if not isinstance(thread_id, str) or not thread_id:
        raise TypeError("Codex SDK returned an invalid thread response")
    return thread_id


def _response_turn_id(response: object) -> str:
    turn = getattr(response, "turn", None)
    turn_id = getattr(turn, "id", None)
    if not isinstance(turn_id, str) or not turn_id:
        raise TypeError("Codex SDK returned an invalid turn response")
    return turn_id


def _safe_error_code(error: object) -> str:
    info = getattr(error, "codex_error_info", None)
    root = getattr(info, "root", None)
    if isinstance(root, CodexErrorInfoValue):
        return "codex." + _camel_to_snake(root.value)
    class_name = type(root).__name__
    known = {
        "HttpConnectionFailedCodexErrorInfo": "codex.http_connection_failed",
        "ResponseStreamConnectionFailedCodexErrorInfo": (
            "codex.response_stream_connection_failed"
        ),
        "ResponseStreamDisconnectedCodexErrorInfo": (
            "codex.response_stream_disconnected"
        ),
        "ResponseTooManyFailedAttemptsCodexErrorInfo": (
            "codex.response_too_many_failed_attempts"
        ),
    }
    return known.get(class_name, "codex.failed")


def _camel_to_snake(value: str) -> str:
    characters: list[str] = []
    for character in value:
        if character.isupper():
            characters.append("_")
            characters.append(character.lower())
        else:
            characters.append(character)
    return "".join(characters)


assert isinstance(CodexSdkClientFactory(), CodexClientFactory)
