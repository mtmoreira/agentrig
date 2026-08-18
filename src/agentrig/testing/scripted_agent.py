"""Scripted autonomous-agent runtime for deterministic test scenarios."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from threading import Lock
from typing import TypeAlias

from agentrig.agents import (
    AgentExecutionRequest,
    AgentExecutionResult,
    AgentRuntimeUsage,
)
from agentrig.core._validation import require_trimmed_string
from agentrig.core.context import RunContext
from agentrig.core.errors import Failure, FailureKind
from agentrig.core.events import Event, EventKind, JsonValue


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedAgentProgress:
    """One safe progress summary emitted by a scripted runtime."""

    message: str

    def __post_init__(self) -> None:
        require_trimmed_string("scripted progress message", self.message)


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedToolRequest:
    """One tool request made by a scripted autonomous run."""

    tool_name: str

    def __post_init__(self) -> None:
        require_trimmed_string("scripted tool name", self.tool_name)


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedApprovalRequest:
    """One terminal approval request made by a scripted autonomous run."""

    approval_id: str
    summary: str

    def __post_init__(self) -> None:
        require_trimmed_string("scripted approval ID", self.approval_id)
        require_trimmed_string("scripted approval summary", self.summary)


ScriptedAgentAction: TypeAlias = (
    ScriptedAgentProgress | ScriptedToolRequest | ScriptedApprovalRequest
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedAgentScenario:
    """Ordered runtime actions and the normalized result they produce."""

    result: AgentExecutionResult
    actions: tuple[ScriptedAgentAction, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.result, AgentExecutionResult):
            raise TypeError(
                "scripted agent scenario result must be an AgentExecutionResult"
            )
        copied_actions = tuple(self.actions)
        for action in copied_actions:
            if not isinstance(
                action,
                (
                    ScriptedAgentProgress,
                    ScriptedToolRequest,
                    ScriptedApprovalRequest,
                ),
            ):
                raise TypeError("scripted agent scenario contains an invalid action")
        object.__setattr__(self, "actions", copied_actions)

        approvals = tuple(
            action
            for action in copied_actions
            if isinstance(action, ScriptedApprovalRequest)
        )
        if len(approvals) > 1:
            raise ValueError(
                "scripted agent scenario permits at most one approval request"
            )
        if approvals and copied_actions[-1] is not approvals[0]:
            raise ValueError("scripted approval request must be the final action")

        failure = self.result.result.failure
        requires_approval = (
            failure is not None
            and failure.kind is FailureKind.APPROVAL_REQUIRED
        )
        if bool(approvals) != requires_approval:
            raise ValueError(
                "scripted approval action and approval_required result must match"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedAgentRuntimeCall:
    """One request and context presented to a scripted agent runtime."""

    index: int
    request: AgentExecutionRequest
    context: RunContext


class ScriptedAgentRuntime:
    """Return deterministic agent scenarios in call order."""

    def __init__(
        self,
        *,
        scenarios: Iterable[ScriptedAgentScenario],
        repeat_last: bool = False,
    ) -> None:
        if not isinstance(repeat_last, bool):
            raise TypeError("scripted agent repeat_last must be a bool")
        copied_scenarios = tuple(scenarios)
        if not copied_scenarios:
            raise ValueError("scripted agent runtime requires at least one scenario")
        for scenario in copied_scenarios:
            if not isinstance(scenario, ScriptedAgentScenario):
                raise TypeError(
                    "scripted agent runtime requires ScriptedAgentScenario values"
                )

        self._scenarios = copied_scenarios
        self._repeat_last = repeat_last
        self._next_scenario = 0
        self._calls: list[ScriptedAgentRuntimeCall] = []
        self._lock = Lock()

    @property
    def calls(self) -> tuple[ScriptedAgentRuntimeCall, ...]:
        """Return a stable snapshot of recorded runtime calls."""
        with self._lock:
            return tuple(self._calls)

    @property
    def is_exhausted(self) -> bool:
        """Whether another call would return the exhaustion failure."""
        with self._lock:
            return (
                not self._repeat_last
                and self._next_scenario >= len(self._scenarios)
            )

    async def execute(
        self,
        request: AgentExecutionRequest,
        context: RunContext,
    ) -> AgentExecutionResult:
        """Consume one scenario after checking caller-owned constraints."""
        if not isinstance(request, AgentExecutionRequest):
            raise TypeError(
                "scripted agent request must be an AgentExecutionRequest"
            )
        if not isinstance(context, RunContext):
            raise TypeError("scripted agent context must be a RunContext")
        _check_constraints(context)

        with self._lock:
            call = ScriptedAgentRuntimeCall(
                index=len(self._calls),
                request=request,
                context=context,
            )
            self._calls.append(call)
            scenario = self._take_next_scenario()

        _emit(
            context,
            EventKind.PROVIDER_CALL_STARTED,
            _base_attributes(request, call_index=call.index),
        )

        if scenario is None:
            result = AgentExecutionResult.from_failure(
                Failure(
                    kind=FailureKind.UNEXPECTED,
                    message="scripted agent runtime has no remaining scenarios",
                    code="scripted_agent_runtime.exhausted",
                    metadata={
                        "agent_id": request.contract.agent_id,
                        "agent_version": request.contract.version,
                    },
                )
            )
            _emit_completed(context, request, call.index, result)
            return result

        disallowed_tool = _find_disallowed_tool(request, scenario)
        if disallowed_tool is not None:
            result = AgentExecutionResult.from_failure(
                Failure(
                    kind=FailureKind.INVALID_INPUT,
                    message="agent requested a tool outside its allowlist",
                    code="agent.tool_not_allowed",
                    metadata={
                        "agent_id": request.contract.agent_id,
                        "tool_name": disallowed_tool,
                    },
                )
            )
            _emit_completed(context, request, call.index, result)
            return result

        tool_index = 0
        for action in scenario.actions:
            _check_constraints(context)
            if isinstance(action, ScriptedAgentProgress):
                _emit(
                    context,
                    EventKind.PROGRESS_REPORTED,
                    {
                        **_base_attributes(request, call_index=call.index),
                        "message": action.message,
                    },
                )
            elif isinstance(action, ScriptedToolRequest):
                _emit_tool_request(
                    context,
                    request,
                    action,
                    call_index=call.index,
                    tool_index=tool_index,
                )
                tool_index += 1
            else:
                _emit(
                    context,
                    EventKind.APPROVAL_REQUESTED,
                    {
                        **_base_attributes(request, call_index=call.index),
                        "approval_id": action.approval_id,
                        "summary": action.summary,
                    },
                )

        _check_constraints(context)
        _emit_usage(context, request, call.index, scenario.result.usage)
        _emit_completed(context, request, call.index, scenario.result)
        return scenario.result

    def _take_next_scenario(self) -> ScriptedAgentScenario | None:
        if self._next_scenario >= len(self._scenarios):
            return None
        scenario = self._scenarios[self._next_scenario]
        if not (
            self._repeat_last
            and self._next_scenario == len(self._scenarios) - 1
        ):
            self._next_scenario += 1
        return scenario


def _check_constraints(context: RunContext) -> None:
    context.cancellation.raise_if_cancelled()
    if context.deadline is not None:
        context.deadline.raise_if_expired(context.clock)


def _find_disallowed_tool(
    request: AgentExecutionRequest,
    scenario: ScriptedAgentScenario,
) -> str | None:
    allowed_tools = frozenset(request.contract.allowed_tools)
    for action in scenario.actions:
        if (
            isinstance(action, ScriptedToolRequest)
            and action.tool_name not in allowed_tools
        ):
            return action.tool_name
    return None


def _emit_tool_request(
    context: RunContext,
    request: AgentExecutionRequest,
    action: ScriptedToolRequest,
    *,
    call_index: int,
    tool_index: int,
) -> None:
    attributes: dict[str, JsonValue] = {
        **_base_attributes(request, call_index=call_index),
        "tool_name": action.tool_name,
        "tool_index": tool_index,
    }
    _emit(context, EventKind.TOOL_CALL_STARTED, attributes)
    _check_constraints(context)
    _emit(
        context,
        EventKind.TOOL_CALL_COMPLETED,
        {**attributes, "status": "succeeded"},
    )


def _emit_completed(
    context: RunContext,
    request: AgentExecutionRequest,
    call_index: int,
    result: AgentExecutionResult,
) -> None:
    attributes: dict[str, JsonValue] = {
        **_base_attributes(request, call_index=call_index),
        "status": result.result.status.value,
    }
    failure = result.result.failure
    if failure is not None:
        attributes["failure_kind"] = failure.kind.value
        if failure.code is not None:
            attributes["failure_code"] = failure.code
    _emit(context, EventKind.PROVIDER_CALL_COMPLETED, attributes)


def _emit_usage(
    context: RunContext,
    request: AgentExecutionRequest,
    call_index: int,
    usage: AgentRuntimeUsage,
) -> None:
    if not usage.is_reported:
        return
    attributes: dict[str, JsonValue] = {
        **_base_attributes(request, call_index=call_index),
    }
    if usage.input_tokens is not None:
        attributes["input_tokens"] = usage.input_tokens
    if usage.cached_input_tokens is not None:
        attributes["cached_input_tokens"] = usage.cached_input_tokens
    if usage.output_tokens is not None:
        attributes["output_tokens"] = usage.output_tokens
    _emit(context, EventKind.USAGE_REPORTED, attributes)


def _base_attributes(
    request: AgentExecutionRequest,
    *,
    call_index: int,
) -> dict[str, JsonValue]:
    return {
        "agent_id": request.contract.agent_id,
        "agent_version": request.contract.version,
        "call_index": call_index,
    }


def _emit(
    context: RunContext,
    kind: EventKind,
    attributes: dict[str, JsonValue],
) -> None:
    context.event_sink.emit(
        Event.from_context(
            event_id=context.event_id_generator.generate(),
            kind=kind,
            context=context,
            attributes=attributes,
        )
    )
