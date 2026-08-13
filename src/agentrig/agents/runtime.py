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
    provider_metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.result, AgentResult):
            raise TypeError("runtime result must contain an AgentResult")
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
        provider_metadata: Mapping[str, str] | None = None,
    ) -> AgentExecutionResult:
        """Create a successful normalized runtime result."""
        return cls(
            result=AgentResult.succeeded(output, artifacts=artifacts),
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
        provider_metadata: Mapping[str, str] | None = None,
    ) -> AgentExecutionResult:
        """Create a failed normalized runtime result."""
        return cls(
            result=AgentResult.from_failure(failure, artifacts=artifacts),
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
