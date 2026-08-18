"""Provider-neutral AgentRuntime adapter for injected Codex clients."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping

from agentrig.agents import (
    AgentExecutionRequest,
    AgentExecutionResult,
    AgentRuntimeUsage,
)
from agentrig.core._json import JsonValue, freeze_json_object, thaw_json_value
from agentrig.core.context import RunContext
from agentrig.core.deadline import DeadlineExceeded
from agentrig.core.errors import Failure, FailureKind, normalize_exception
from agentrig.core.events import Event, EventKind
from agentrig.integrations.openai.codex import (
    CODEX_AGENT_RUNTIME_CAPABILITY,
    CODEX_SUPPORTED_TOOLS,
    CODEX_WEB_SEARCH_TOOL,
    CodexApprovalMode,
    CodexApprovalRequested,
    CodexClientFactory,
    CodexProgressReported,
    CodexSandboxMode,
    CodexSandboxPolicy,
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

_PROVIDER = "openai.codex"


class CodexAgentRuntime:
    """Execute portable agent requests through one bounded Codex runtime."""

    def __init__(
        self,
        *,
        client_factory: CodexClientFactory,
        model: str,
        sandbox: CodexSandboxPolicy,
        output_schemas: Mapping[str, Mapping[str, JsonValue]],
        approval_mode: CodexApprovalMode = CodexApprovalMode.DENY_ALL,
        ephemeral: bool = True,
    ) -> None:
        if not isinstance(client_factory, CodexClientFactory):
            raise TypeError("Codex runtime factory must satisfy CodexClientFactory")
        if not isinstance(model, str) or not model or model != model.strip():
            raise ValueError("Codex runtime model must be nonempty and trimmed")
        if not isinstance(sandbox, CodexSandboxPolicy):
            raise TypeError("Codex runtime sandbox must be a CodexSandboxPolicy")
        if not isinstance(approval_mode, CodexApprovalMode):
            raise TypeError("Codex runtime approval mode must be a CodexApprovalMode")
        if not isinstance(ephemeral, bool):
            raise TypeError("Codex runtime ephemeral flag must be a bool")

        schemas: dict[str, Mapping[str, JsonValue]] = {}
        for schema_id, schema in output_schemas.items():
            if not schema_id or schema_id != schema_id.strip():
                raise ValueError("Codex schema IDs must be nonempty and trimmed")
            frozen = freeze_json_object("Codex runtime output schema", schema)
            if not frozen:
                raise ValueError("Codex runtime output schemas must not be empty")
            schemas[schema_id] = frozen
        if not schemas:
            raise ValueError("Codex runtime requires at least one output schema")

        self._client_factory = client_factory
        self._model = model
        self._sandbox = sandbox
        self._output_schemas = schemas
        self._approval_mode = approval_mode
        self._ephemeral = ephemeral

    async def execute(
        self,
        request: AgentExecutionRequest,
        context: RunContext,
    ) -> AgentExecutionResult:
        """Validate authority, stream one turn, and normalize its result."""
        if not isinstance(request, AgentExecutionRequest):
            raise TypeError("Codex runtime request must be an AgentExecutionRequest")
        if not isinstance(context, RunContext):
            raise TypeError("Codex runtime context must be a RunContext")

        try:
            _check_constraints(context)
        except BaseException as error:
            return AgentExecutionResult.from_failure(normalize_exception(error))

        configuration_failure = self._validate_request(request)
        if configuration_failure is not None:
            return AgentExecutionResult.from_failure(configuration_failure)

        schema = self._output_schemas[request.contract.output_schema]
        thread_id = _optional_string_option(request, "thread_id")
        _emit(
            context,
            EventKind.PROVIDER_CALL_STARTED,
            _base_attributes(request),
        )

        client = None
        turn: CodexTurn | None = None
        try:
            client = self._client_factory.create()
            thread_request = CodexThreadRequest(
                model=self._model,
                instructions=request.instructions,
                sandbox=self._sandbox,
                approval_mode=self._approval_mode,
                allowed_tools=request.contract.allowed_tools,
                ephemeral=self._ephemeral,
            )
            thread = (
                await client.resume_thread(thread_id, thread_request)
                if thread_id is not None
                else await client.start_thread(thread_request)
            )
            turn = await thread.start_turn(
                CodexTurnRequest(
                    prompt=_encode_prompt(request.input),
                    output_schema=schema,
                    sandbox=self._sandbox,
                    approval_mode=self._approval_mode,
                )
            )
            result = await self._stream_result(request, context, turn)
        except asyncio.CancelledError as error:
            if turn is not None:
                await _interrupt_safely(turn)
            result = AgentExecutionResult.from_failure(normalize_exception(error))
        except Exception as error:
            result = AgentExecutionResult.from_failure(
                _provider_exception_failure(error)
            )
        finally:
            if client is not None:
                try:
                    await client.close()
                except Exception:
                    pass

        _emit_completed(context, request, result)
        return result

    def _validate_request(
        self,
        request: AgentExecutionRequest,
    ) -> Failure | None:
        contract = request.contract
        if (
            CODEX_AGENT_RUNTIME_CAPABILITY.capability_id
            not in contract.allowed_capabilities
        ):
            return _invalid_configuration(
                "agent contract does not authorize the Codex runtime",
                "codex.capability_not_allowed",
            )
        if contract.output_schema not in self._output_schemas:
            return _invalid_configuration(
                "agent output schema is not configured for Codex",
                "codex.output_schema_not_configured",
            )
        unsupported_tools = set(contract.allowed_tools) - CODEX_SUPPORTED_TOOLS
        if unsupported_tools:
            return _invalid_configuration(
                "agent contract authorizes unsupported Codex tools",
                "codex.tool_not_supported",
            )
        if set(request.provider_options) - {"thread_id"}:
            return _invalid_configuration(
                "Codex provider options contain unsupported keys",
                "codex.provider_options_invalid",
            )
        thread_id = request.provider_options.get("thread_id")
        if thread_id is not None and (
            not isinstance(thread_id, str)
            or not thread_id
            or thread_id != thread_id.strip()
        ):
            return _invalid_configuration(
                "Codex thread ID must be nonempty and trimmed",
                "codex.thread_id_invalid",
            )

        required_workspace = (
            "read_only"
            if self._sandbox.mode is CodexSandboxMode.READ_ONLY
            else "read_write"
        )
        if contract.permissions.get("workspace") != required_workspace:
            return _invalid_configuration(
                "agent workspace permission does not match the Codex sandbox",
                "codex.workspace_permission_mismatch",
            )
        required_network = "allowed" if self._sandbox.network_access else "denied"
        if contract.permissions.get("network") != required_network:
            return _invalid_configuration(
                "agent network permission does not match the Codex sandbox",
                "codex.network_permission_mismatch",
            )
        if (
            CODEX_WEB_SEARCH_TOOL in contract.allowed_tools
            and not self._sandbox.network_access
        ):
            return _invalid_configuration(
                "Codex web search requires network authority",
                "codex.web_search_requires_network",
            )
        return None

    async def _stream_result(
        self,
        request: AgentExecutionRequest,
        context: RunContext,
        turn: CodexTurn,
    ) -> AgentExecutionResult:
        constraint_task = asyncio.create_task(_wait_for_constraint(context))
        event_task: asyncio.Task[CodexTurnEvent] | None = None
        approval_requested = False
        tool_calls = 0
        usage = AgentRuntimeUsage()
        terminal: CodexTurnCompleted | None = None
        events = turn.events().__aiter__()
        try:
            while terminal is None:
                event_task = asyncio.create_task(_next_event(events))
                done, _ = await asyncio.wait(
                    {event_task, constraint_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if constraint_task in done:
                    failure = constraint_task.result()
                    await _interrupt_safely(turn)
                    event_task.cancel()
                    await asyncio.gather(event_task, return_exceptions=True)
                    return AgentExecutionResult.from_failure(
                        failure,
                        usage=usage,
                        provider_metadata=_metadata(turn),
                    )

                try:
                    event = event_task.result()
                except StopAsyncIteration:
                    break
                event_task = None

                if isinstance(event, CodexTurnStarted):
                    continue
                if isinstance(event, CodexProgressReported):
                    _emit(
                        context,
                        EventKind.PROGRESS_REPORTED,
                        {
                            **_base_attributes(request),
                            "provider": _PROVIDER,
                            "progress_kind": event.kind.value,
                            "turn_id": event.turn_id,
                        },
                    )
                elif isinstance(event, CodexToolCallStarted):
                    tool_calls += 1
                    if (
                        event.tool_name not in request.contract.allowed_tools
                        or tool_calls > request.contract.limits.max_tool_calls
                    ):
                        await _interrupt_safely(turn)
                        return AgentExecutionResult.from_failure(
                            _invalid_configuration(
                                "Codex attempted a tool outside agent authority",
                                "codex.tool_authority_exceeded",
                            ),
                            usage=usage,
                            provider_metadata=_metadata(turn),
                        )
                    _emit_tool(context, request, event, started=True)
                elif isinstance(event, CodexToolCallCompleted):
                    _emit_tool(context, request, event, started=False)
                elif isinstance(event, CodexApprovalRequested):
                    approval_requested = True
                    _emit(
                        context,
                        EventKind.APPROVAL_REQUESTED,
                        {
                            **_base_attributes(request),
                            "provider": _PROVIDER,
                            "approval_id": event.approval_id,
                            "tool_name": event.tool_name,
                            "turn_id": event.turn_id,
                        },
                    )
                elif isinstance(event, CodexUsageReported):
                    usage = AgentRuntimeUsage(
                        input_tokens=event.input_tokens,
                        cached_input_tokens=event.cached_input_tokens,
                        output_tokens=event.output_tokens,
                    )
                    _emit_usage(context, request, event.turn_id, usage)
                elif isinstance(event, CodexTurnCompleted):
                    terminal = event

            if approval_requested:
                kind = (
                    FailureKind.APPROVAL_REQUIRED
                    if self._approval_mode is CodexApprovalMode.MANUAL
                    else FailureKind.APPROVAL_DENIED
                )
                return AgentExecutionResult.from_failure(
                    Failure(
                        kind=kind,
                        message="Codex tool approval was not granted",
                        code=(
                            "codex.approval_required"
                            if kind is FailureKind.APPROVAL_REQUIRED
                            else "codex.approval_denied"
                        ),
                    ),
                    usage=usage,
                    provider_metadata=_metadata(turn),
                )
            if terminal is None:
                return AgentExecutionResult.from_failure(
                    Failure(
                        kind=FailureKind.PERMANENT_PROVIDER,
                        message="Codex stream ended without a terminal event",
                        code="codex.missing_terminal_event",
                    ),
                    usage=usage,
                    provider_metadata=_metadata(turn),
                )
            return _terminal_result(terminal, turn, usage)
        finally:
            constraint_task.cancel()
            if event_task is not None:
                event_task.cancel()
            await asyncio.gather(
                constraint_task,
                *(tuple([event_task]) if event_task is not None else ()),
                return_exceptions=True,
            )
            close_events = getattr(events, "aclose", None)
            if close_events is not None:
                try:
                    await close_events()
                except Exception:
                    pass


async def _wait_for_constraint(context: RunContext) -> Failure:
    cancellation_task = asyncio.create_task(context.cancellation.wait_cancelled())
    deadline_task: asyncio.Task[None] | None = None
    if context.deadline is not None:
        deadline_task = asyncio.create_task(
            asyncio.sleep(context.deadline.remaining_seconds(context.clock))
        )
    try:
        if deadline_task is None:
            cancellation = await cancellation_task
            return Failure(kind=FailureKind.CANCELLED, message=cancellation.reason)
        done, _ = await asyncio.wait(
            {cancellation_task, deadline_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if cancellation_task in done:
            cancellation = cancellation_task.result()
            return Failure(kind=FailureKind.CANCELLED, message=cancellation.reason)
        if context.deadline is None:
            raise AssertionError("deadline task requires a deadline")
        return normalize_exception(DeadlineExceeded(context.deadline))
    finally:
        cancellation_task.cancel()
        if deadline_task is not None:
            deadline_task.cancel()
        await asyncio.gather(
            cancellation_task,
            *(tuple([deadline_task]) if deadline_task is not None else ()),
            return_exceptions=True,
        )


async def _next_event(
    events: AsyncIterator[CodexTurnEvent],
) -> CodexTurnEvent:
    return await anext(events)


def _terminal_result(
    terminal: CodexTurnCompleted,
    turn: CodexTurn,
    usage: AgentRuntimeUsage,
) -> AgentExecutionResult:
    metadata = _metadata(turn)
    if terminal.status is CodexTurnStatus.SUCCEEDED:
        return AgentExecutionResult.succeeded(
            terminal.output,
            usage=usage,
            provider_metadata=metadata,
        )
    failures = {
        CodexTurnStatus.INTERRUPTED: Failure(
            kind=FailureKind.CANCELLED,
            message="Codex turn was interrupted",
            code="codex.interrupted",
        ),
        CodexTurnStatus.REFUSED: Failure(
            kind=FailureKind.POLICY_REFUSAL,
            message="Codex refused the request",
            code=terminal.error_code or "codex.refused",
        ),
        CodexTurnStatus.APPROVAL_REQUIRED: Failure(
            kind=FailureKind.APPROVAL_REQUIRED,
            message="Codex requires tool approval",
            code=terminal.error_code or "codex.approval_required",
        ),
        CodexTurnStatus.INVALID_OUTPUT: Failure(
            kind=FailureKind.PERMANENT_PROVIDER,
            message="Codex returned invalid structured output",
            code=terminal.error_code or "codex.invalid_output",
        ),
        CodexTurnStatus.FAILED: Failure(
            kind=_failure_kind_for_code(terminal.error_code),
            message="Codex could not complete the request",
            code=terminal.error_code or "codex.failed",
        ),
    }
    return AgentExecutionResult.from_failure(
        failures[terminal.status],
        usage=usage,
        provider_metadata=metadata,
    )


def _emit_usage(
    context: RunContext,
    request: AgentExecutionRequest,
    turn_id: str,
    usage: AgentRuntimeUsage,
) -> None:
    attributes: dict[str, JsonValue] = {
        **_base_attributes(request),
        "provider": _PROVIDER,
        "turn_id": turn_id,
    }
    if usage.input_tokens is not None:
        attributes["input_tokens"] = usage.input_tokens
    if usage.cached_input_tokens is not None:
        attributes["cached_input_tokens"] = usage.cached_input_tokens
    if usage.output_tokens is not None:
        attributes["output_tokens"] = usage.output_tokens
    _emit(context, EventKind.USAGE_REPORTED, attributes)


def _failure_kind_for_code(code: str | None) -> FailureKind:
    if code in {
        "codex.server_overloaded",
        "codex.http_connection_failed",
        "codex.response_stream_connection_failed",
        "codex.response_stream_disconnected",
        "codex.response_too_many_failed_attempts",
        "codex.internal_server_error",
    }:
        return FailureKind.TRANSIENT_PROVIDER
    if code in {"codex.usage_limit_exceeded", "codex.session_budget_exceeded"}:
        return FailureKind.BUDGET_EXHAUSTED
    if code == "codex.cyber_policy":
        return FailureKind.POLICY_REFUSAL
    return FailureKind.PERMANENT_PROVIDER


def _provider_exception_failure(error: Exception) -> Failure:
    return Failure(
        kind=FailureKind.TRANSIENT_PROVIDER,
        message="Codex runtime transport failed",
        code="codex.transport_failed",
        metadata={"exception_type": type(error).__qualname__},
    )


def _invalid_configuration(message: str, code: str) -> Failure:
    return Failure(kind=FailureKind.INVALID_INPUT, message=message, code=code)


def _encode_prompt(value: JsonValue) -> str:
    return json.dumps(
        thaw_json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _optional_string_option(
    request: AgentExecutionRequest,
    key: str,
) -> str | None:
    value = request.provider_options.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"Codex provider option {key} must be trimmed text")
    return value


def _check_constraints(context: RunContext) -> None:
    context.cancellation.raise_if_cancelled()
    if context.deadline is not None:
        context.deadline.raise_if_expired(context.clock)


async def _interrupt_safely(turn: CodexTurn) -> None:
    try:
        await turn.interrupt()
    except Exception:
        pass


def _metadata(turn: CodexTurn) -> Mapping[str, str]:
    return {
        "provider": _PROVIDER,
        "thread_id": turn.thread_id,
        "turn_id": turn.turn_id,
    }


def _base_attributes(request: AgentExecutionRequest) -> dict[str, JsonValue]:
    return {
        "agent_id": request.contract.agent_id,
        "agent_version": request.contract.version,
        "provider": _PROVIDER,
    }


def _emit_tool(
    context: RunContext,
    request: AgentExecutionRequest,
    event: CodexToolCallStarted | CodexToolCallCompleted,
    *,
    started: bool,
) -> None:
    attributes: dict[str, JsonValue] = {
        **_base_attributes(request),
        "turn_id": event.turn_id,
        "call_id": event.call_id,
        "tool_name": event.tool_name,
    }
    if isinstance(event, CodexToolCallCompleted):
        attributes["status"] = "succeeded" if event.succeeded else "failed"
    _emit(
        context,
        EventKind.TOOL_CALL_STARTED if started else EventKind.TOOL_CALL_COMPLETED,
        attributes,
    )


def _emit_completed(
    context: RunContext,
    request: AgentExecutionRequest,
    result: AgentExecutionResult,
) -> None:
    attributes: dict[str, JsonValue] = {
        **_base_attributes(request),
        "status": result.result.status.value,
    }
    failure = result.result.failure
    if failure is not None:
        attributes["failure_kind"] = failure.kind.value
        if failure.code is not None:
            attributes["failure_code"] = failure.code
    _emit(context, EventKind.PROVIDER_CALL_COMPLETED, attributes)


def _emit(
    context: RunContext,
    kind: EventKind,
    attributes: Mapping[str, JsonValue],
) -> None:
    context.event_sink.emit(
        Event.from_context(
            event_id=context.event_id_generator.generate(),
            kind=kind,
            context=context,
            attributes=attributes,
        )
    )
