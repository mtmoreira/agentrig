"""Provider-neutral autonomous agent runtime boundary."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentrig.agents.contract import AgentContract, AgentResult
from agentrig.core._json import JsonValue, freeze_json_object, freeze_json_value
from agentrig.core._validation import freeze_string_map, require_trimmed_string
from agentrig.core.artifacts import ArtifactRef
from agentrig.core.context import RunContext
from agentrig.core.errors import Failure


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentRuntimeUsage:
    """Portable token counts reported by one autonomous runtime execution."""

    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        _require_optional_token_count(
            "agent runtime input_tokens",
            self.input_tokens,
        )
        _require_optional_token_count(
            "agent runtime cached_input_tokens",
            self.cached_input_tokens,
        )
        _require_optional_token_count(
            "agent runtime output_tokens",
            self.output_tokens,
        )
        if self.cached_input_tokens is not None:
            if self.input_tokens is None:
                raise ValueError(
                    "agent runtime cached_input_tokens require input_tokens"
                )
            if self.cached_input_tokens > self.input_tokens:
                raise ValueError(
                    "agent runtime cached_input_tokens cannot exceed input_tokens"
                )

    @property
    def total_tokens(self) -> int | None:
        """Return a total only when input and output counts are available."""
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens

    @property
    def is_reported(self) -> bool:
        """Whether the runtime reported any portable usage count."""
        return any(
            value is not None
            for value in (
                self.input_tokens,
                self.cached_input_tokens,
                self.output_tokens,
            )
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentExecutionRequest:
    """Immutable encoded input and configuration passed to an agent runtime."""

    contract: AgentContract[Any, Any]
    instructions: str
    input: JsonValue
    provider_options: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.contract, AgentContract):
            raise TypeError("runtime request contract must be an AgentContract")
        require_trimmed_string("agent instructions", self.instructions)
        object.__setattr__(
            self,
            "input",
            freeze_json_value("runtime request input", self.input),
        )
        object.__setattr__(
            self,
            "provider_options",
            freeze_json_object(
                "runtime provider options",
                self.provider_options,
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentExecutionResult:
    """Normalized runtime result with non-portable metadata kept separate."""

    result: AgentResult[JsonValue]
    usage: AgentRuntimeUsage = field(default_factory=AgentRuntimeUsage)
    provider_metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.result, AgentResult):
            raise TypeError("runtime result must contain an AgentResult")
        if not isinstance(self.usage, AgentRuntimeUsage):
            raise TypeError("runtime result usage must be AgentRuntimeUsage")
        if self.result.is_success:
            frozen_output = freeze_json_value(
                "runtime result output",
                self.result.output,
            )
            object.__setattr__(
                self,
                "result",
                AgentResult.succeeded(
                    frozen_output,
                    artifacts=self.result.artifacts,
                ),
            )
        object.__setattr__(
            self,
            "provider_metadata",
            freeze_string_map(
                "runtime provider metadata",
                self.provider_metadata,
            ),
        )

    @classmethod
    def succeeded(
        cls,
        output: JsonValue,
        *,
        artifacts: Iterable[ArtifactRef] = (),
        usage: AgentRuntimeUsage | None = None,
        provider_metadata: Mapping[str, str] | None = None,
    ) -> AgentExecutionResult:
        """Create a successful normalized runtime result."""
        return cls(
            result=AgentResult.succeeded(output, artifacts=artifacts),
            usage=usage if usage is not None else AgentRuntimeUsage(),
            provider_metadata=(
                provider_metadata if provider_metadata is not None else {}
            ),
        )

    @classmethod
    def from_failure(
        cls,
        failure: Failure,
        *,
        artifacts: Iterable[ArtifactRef] = (),
        usage: AgentRuntimeUsage | None = None,
        provider_metadata: Mapping[str, str] | None = None,
    ) -> AgentExecutionResult:
        """Create a failed normalized runtime result."""
        return cls(
            result=AgentResult.from_failure(failure, artifacts=artifacts),
            usage=usage if usage is not None else AgentRuntimeUsage(),
            provider_metadata=(
                provider_metadata if provider_metadata is not None else {}
            ),
        )


@runtime_checkable
class AgentRuntime(Protocol):
    """Execute provider-neutral agent requests through one runtime substrate."""

    async def execute(
        self,
        request: AgentExecutionRequest,
        context: RunContext,
    ) -> AgentExecutionResult:
        """Translate one autonomous provider run into a normalized result."""
        ...


def _require_optional_token_count(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")
