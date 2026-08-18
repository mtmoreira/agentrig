"""Stable injected boundary for the optional Codex autonomous runtime."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Protocol, TypeAlias, runtime_checkable

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    DataRetention,
)
from agentrig.core._json import JsonValue, freeze_json_object, freeze_json_value
from agentrig.core._validation import require_trimmed_string

CODEX_SDK_VERSION = "0.144.4"
CODEX_SHELL_TOOL = "codex.shell"
CODEX_WEB_SEARCH_TOOL = "codex.web_search"
CODEX_SUPPORTED_TOOLS = frozenset(
    {
        CODEX_SHELL_TOOL,
        CODEX_WEB_SEARCH_TOOL,
    }
)

CODEX_AGENT_RUNTIME_CAPABILITY = CapabilityDescriptor(
    capability_id="openai.codex.agent_runtime",
    version=CODEX_SDK_VERSION,
    kind=CapabilityKind.AGENT_RUNTIME,
    features=frozenset(
        {
            CapabilityFeature.STREAMING,
            CapabilityFeature.CANCELLATION,
            CapabilityFeature.STRUCTURED_OUTPUT,
            CapabilityFeature.SESSION_CONTINUATION,
            CapabilityFeature.APPROVAL_REQUESTS,
            CapabilityFeature.TOOL_USE,
            CapabilityFeature.USAGE_REPORTING,
        }
    ),
    data_retention=DataRetention.PROVIDER_MANAGED,
)


@runtime_checkable
class CodexAuthenticationSource(Protocol):
    """Resolve a private process-environment overlay at client creation."""

    def resolve_environment(self) -> Mapping[str, str]:
        """Return only application-authorized authentication variables."""
        ...


class CodexSandboxMode(StrEnum):
    """Bounded sandbox modes supported by the initial adapter."""

    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"


class CodexApprovalMode(StrEnum):
    """Approval behavior that never grants authority implicitly."""

    DENY_ALL = "deny_all"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True, kw_only=True)
class CodexSandboxPolicy:
    """Explicit filesystem and network authority for one Codex turn."""

    mode: CodexSandboxMode
    cwd: str
    writable_roots: tuple[str, ...] = ()
    network_access: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.mode, CodexSandboxMode):
            raise TypeError("Codex sandbox mode must be a CodexSandboxMode")
        _require_bounded_absolute_path("Codex sandbox cwd", self.cwd)
        roots = tuple(self.writable_roots)
        for root in roots:
            _require_bounded_absolute_path("Codex writable root", root)
        if len(roots) != len(set(roots)):
            raise ValueError("Codex writable roots must not contain duplicates")
        if not isinstance(self.network_access, bool):
            raise TypeError("Codex network_access must be a bool")

        if self.mode is CodexSandboxMode.READ_ONLY and roots:
            raise ValueError("read-only Codex sandboxes cannot have writable roots")
        if self.mode is CodexSandboxMode.WORKSPACE_WRITE:
            if not roots:
                raise ValueError(
                    "workspace-write Codex sandboxes require writable roots"
                )
            if self.cwd not in roots:
                raise ValueError(
                    "Codex sandbox cwd must be an explicit writable root"
                )
        object.__setattr__(self, "writable_roots", roots)


@dataclass(frozen=True, slots=True, kw_only=True)
class CodexThreadRequest:
    """Provider configuration required to start or resume one Codex thread."""

    model: str
    instructions: str
    sandbox: CodexSandboxPolicy
    approval_mode: CodexApprovalMode
    allowed_tools: tuple[str, ...] = ()
    ephemeral: bool = True
    service_name: str = "agentrig"

    def __post_init__(self) -> None:
        require_trimmed_string("Codex model", self.model)
        require_trimmed_string("Codex instructions", self.instructions)
        if not isinstance(self.sandbox, CodexSandboxPolicy):
            raise TypeError("Codex thread sandbox must be a CodexSandboxPolicy")
        if not isinstance(self.approval_mode, CodexApprovalMode):
            raise TypeError(
                "Codex thread approval_mode must be a CodexApprovalMode"
            )
        tools = tuple(self.allowed_tools)
        if len(tools) != len(set(tools)):
            raise ValueError("Codex allowed tools must not contain duplicates")
        unsupported = set(tools) - CODEX_SUPPORTED_TOOLS
        if unsupported:
            raise ValueError("Codex allowed tools contain unsupported values")
        if (
            CODEX_WEB_SEARCH_TOOL in tools
            and not self.sandbox.network_access
        ):
            raise ValueError("Codex web search requires network access")
        if not isinstance(self.ephemeral, bool):
            raise TypeError("Codex thread ephemeral must be a bool")
        require_trimmed_string("Codex service name", self.service_name)
        object.__setattr__(self, "allowed_tools", tools)


@dataclass(frozen=True, slots=True, kw_only=True)
class CodexTurnRequest:
    """Structured input and repeated authority for one Codex turn."""

    prompt: str
    output_schema: Mapping[str, JsonValue]
    sandbox: CodexSandboxPolicy
    approval_mode: CodexApprovalMode

    def __post_init__(self) -> None:
        require_trimmed_string("Codex turn prompt", self.prompt)
        frozen_schema = freeze_json_object(
            "Codex output schema",
            self.output_schema,
        )
        if not frozen_schema:
            raise ValueError("Codex output schema must not be empty")
        if not isinstance(self.sandbox, CodexSandboxPolicy):
            raise TypeError("Codex turn sandbox must be a CodexSandboxPolicy")
        if not isinstance(self.approval_mode, CodexApprovalMode):
            raise TypeError(
                "Codex turn approval_mode must be a CodexApprovalMode"
            )
        object.__setattr__(self, "output_schema", frozen_schema)


class CodexProgressKind(StrEnum):
    """Safe progress categories that omit provider text and payloads."""

    PLAN = "plan"
    REASONING = "reasoning"
    CONTEXT_COMPACTION = "context_compaction"


class CodexTurnStatus(StrEnum):
    """Terminal states normalized at the injected Codex boundary."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    REFUSED = "refused"
    APPROVAL_REQUIRED = "approval_required"
    INVALID_OUTPUT = "invalid_output"


@dataclass(frozen=True, slots=True, kw_only=True)
class CodexTurnStarted:
    """Signal that the provider accepted and started one turn."""

    turn_id: str

    def __post_init__(self) -> None:
        require_trimmed_string("Codex turn ID", self.turn_id)


@dataclass(frozen=True, slots=True, kw_only=True)
class CodexProgressReported:
    """Safe progress category without raw model reasoning or text."""

    turn_id: str
    kind: CodexProgressKind

    def __post_init__(self) -> None:
        require_trimmed_string("Codex turn ID", self.turn_id)
        if not isinstance(self.kind, CodexProgressKind):
            raise TypeError("Codex progress kind must be a CodexProgressKind")


@dataclass(frozen=True, slots=True, kw_only=True)
class CodexToolCallStarted:
    """Safe start record for an allowlisted provider tool call."""

    turn_id: str
    call_id: str
    tool_name: str

    def __post_init__(self) -> None:
        _validate_tool_event(self.turn_id, self.call_id, self.tool_name)


@dataclass(frozen=True, slots=True, kw_only=True)
class CodexToolCallCompleted:
    """Safe terminal record for an allowlisted provider tool call."""

    turn_id: str
    call_id: str
    tool_name: str
    succeeded: bool

    def __post_init__(self) -> None:
        _validate_tool_event(self.turn_id, self.call_id, self.tool_name)
        if not isinstance(self.succeeded, bool):
            raise TypeError("Codex tool success flag must be a bool")


@dataclass(frozen=True, slots=True, kw_only=True)
class CodexApprovalRequested:
    """Bounded approval record without command arguments or provider payloads."""

    turn_id: str
    approval_id: str
    tool_name: str

    def __post_init__(self) -> None:
        require_trimmed_string("Codex turn ID", self.turn_id)
        require_trimmed_string("Codex approval ID", self.approval_id)
        require_trimmed_string("Codex approval tool name", self.tool_name)


@dataclass(frozen=True, slots=True, kw_only=True)
class CodexUsageReported:
    """Non-negative token usage detached from provider payloads."""

    turn_id: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        require_trimmed_string("Codex turn ID", self.turn_id)
        _require_non_negative_integer("Codex input tokens", self.input_tokens)
        _require_non_negative_integer(
            "Codex cached input tokens",
            self.cached_input_tokens,
        )
        _require_non_negative_integer("Codex output tokens", self.output_tokens)
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError(
                "Codex cached input tokens cannot exceed input tokens"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class CodexTurnCompleted:
    """Structured terminal output or a safe provider error code."""

    turn_id: str
    status: CodexTurnStatus
    output: JsonValue = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        require_trimmed_string("Codex turn ID", self.turn_id)
        if not isinstance(self.status, CodexTurnStatus):
            raise TypeError("Codex turn status must be a CodexTurnStatus")
        frozen_output = freeze_json_value("Codex turn output", self.output)
        if self.status is CodexTurnStatus.SUCCEEDED:
            if self.error_code is not None:
                raise ValueError(
                    "successful Codex turns cannot have an error code"
                )
        else:
            if self.output is not None:
                raise ValueError(
                    "non-successful Codex turns cannot have output"
                )
            if self.error_code is not None:
                require_trimmed_string(
                    "Codex turn error code",
                    self.error_code,
                )
        object.__setattr__(self, "output", frozen_output)


CodexTurnEvent: TypeAlias = (
    CodexTurnStarted
    | CodexProgressReported
    | CodexToolCallStarted
    | CodexToolCallCompleted
    | CodexApprovalRequested
    | CodexUsageReported
    | CodexTurnCompleted
)


@runtime_checkable
class CodexTurn(Protocol):
    """Controllable streamed turn returned by an injected client."""

    @property
    def thread_id(self) -> str:
        """Return the provider thread identifier."""
        ...

    @property
    def turn_id(self) -> str:
        """Return the provider turn identifier."""
        ...

    def events(self) -> AsyncIterator[CodexTurnEvent]:
        """Stream safe translated events through terminal completion."""
        ...

    async def interrupt(self) -> None:
        """Request interruption while leaving the stream drainable."""
        ...


@runtime_checkable
class CodexThread(Protocol):
    """Started or resumed Codex thread detached from SDK-specific types."""

    @property
    def thread_id(self) -> str:
        """Return the provider thread identifier."""
        ...

    async def start_turn(self, request: CodexTurnRequest) -> CodexTurn:
        """Start one turn with explicit repeated authority."""
        ...


@runtime_checkable
class CodexClient(Protocol):
    """Small async client seam implemented by the optional SDK bridge."""

    async def start_thread(self, request: CodexThreadRequest) -> CodexThread:
        """Start a fresh provider thread."""
        ...

    async def resume_thread(
        self,
        thread_id: str,
        request: CodexThreadRequest,
    ) -> CodexThread:
        """Resume a provider thread using the caller's current authority."""
        ...

    async def close(self) -> None:
        """Release the client and bundled runtime resources."""
        ...


@runtime_checkable
class CodexClientFactory(Protocol):
    """Inject client construction without starting Codex in unit tests."""

    def create(self) -> CodexClient:
        """Create one unstarted async client instance."""
        ...


def _validate_tool_event(turn_id: str, call_id: str, tool_name: str) -> None:
    require_trimmed_string("Codex turn ID", turn_id)
    require_trimmed_string("Codex tool call ID", call_id)
    require_trimmed_string("Codex tool name", tool_name)


def _require_bounded_absolute_path(field_name: str, value: str) -> None:
    require_trimmed_string(field_name, value)
    if "\\" in value or "\x00" in value:
        raise ValueError(f"{field_name} must use safe POSIX syntax")
    parsed = PurePosixPath(value)
    if (
        not parsed.is_absolute()
        or value == "/"
        or value.startswith("//")
        or ".." in parsed.parts
        or parsed.as_posix() != value
    ):
        raise ValueError(
            f"{field_name} must be canonical, absolute, and bounded"
        )


def _require_non_negative_integer(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
