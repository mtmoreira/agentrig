"""Scripted typed tools for deterministic tests."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Generic, TypeAlias, TypeVar

from agentrig.capabilities import (
    ToolContract,
    ToolInvocation,
    ToolResult,
)
from agentrig.core._json import JsonValue, freeze_json_value
from agentrig.core.artifacts import ArtifactRef
from agentrig.core.context import RunContext
from agentrig.core.errors import AgentRigError, Failure
from agentrig.testing._scripted_capabilities import (
    check_constraints,
    exhaustion_failure,
    require_context,
)
from agentrig.testing._scripted_outcomes import ScriptedOutcomes

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedToolSuccess:
    """Encoded success and durable artifacts for one tool invocation."""

    encoded_output: JsonValue = field(repr=False)
    artifacts: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "encoded_output",
            freeze_json_value(
                "scripted tool encoded output",
                self.encoded_output,
            ),
        )
        object.__setattr__(
            self,
            "artifacts",
            _copy_artifacts(self.artifacts),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedToolFailure:
    """Normalized expected failure and partial artifacts for one tool call."""

    failure: Failure
    artifacts: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.failure, Failure):
            raise TypeError("scripted tool failure must be a Failure")
        object.__setattr__(
            self,
            "artifacts",
            _copy_artifacts(self.artifacts),
        )


ScriptedToolOutcome: TypeAlias = ScriptedToolSuccess | ScriptedToolFailure


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedToolCall(Generic[InputT, OutputT]):
    """One invocation and context presented to a scripted tool."""

    index: int
    invocation: ToolInvocation[InputT, OutputT]
    context: RunContext


class ScriptedTool(Generic[InputT, OutputT]):
    """Return predefined schema-decoded tool outcomes in call order."""

    def __init__(
        self,
        *,
        contract: ToolContract[InputT, OutputT],
        outcomes: Iterable[ScriptedToolOutcome],
        repeat_last: bool = False,
    ) -> None:
        if not isinstance(contract, ToolContract):
            raise TypeError("scripted tool contract must be a ToolContract")
        copied_outcomes = tuple(outcomes)
        if not copied_outcomes:
            raise ValueError("scripted tool requires at least one outcome")
        if any(
            not isinstance(
                outcome,
                (ScriptedToolSuccess, ScriptedToolFailure),
            )
            for outcome in copied_outcomes
        ):
            raise TypeError(
                "scripted tool outcomes must contain ScriptedToolSuccess "
                "or ScriptedToolFailure values"
            )

        self._contract = contract
        self._script = ScriptedOutcomes[
            ScriptedToolOutcome,
            ScriptedToolCall[InputT, OutputT],
        ](outcomes=copied_outcomes, repeat_last=repeat_last)

    @property
    def contract(self) -> ToolContract[InputT, OutputT]:
        return self._contract

    @property
    def calls(self) -> tuple[ScriptedToolCall[InputT, OutputT], ...]:
        """Return a stable snapshot of recorded tool calls."""
        return self._script.calls

    @property
    def is_exhausted(self) -> bool:
        """Whether another call would raise the exhaustion failure."""
        return self._script.is_exhausted

    async def invoke(
        self,
        invocation: ToolInvocation[InputT, OutputT],
        context: RunContext,
    ) -> ToolResult[OutputT]:
        """Consume one outcome after exact-contract and constraint checks."""
        if not isinstance(invocation, ToolInvocation):
            raise TypeError(
                "scripted tool invocation must be a ToolInvocation"
            )
        require_context(context, "scripted tool")
        if invocation.contract is not self.contract:
            raise ValueError(
                "scripted tool invocation contract does not match the tool"
            )
        check_constraints(context)

        outcome = self._script.record_and_take(
            lambda index: ScriptedToolCall(
                index=index,
                invocation=invocation,
                context=context,
            )
        )
        if outcome is None:
            raise AgentRigError(
                exhaustion_failure(
                    self.contract.descriptor,
                    code="scripted_tool.exhausted",
                    message="scripted tool has no remaining outcomes",
                )
            )
        if isinstance(outcome, ScriptedToolFailure):
            return ToolResult.from_failure(
                invocation=invocation,
                failure=outcome.failure,
                artifacts=outcome.artifacts,
            )
        return ToolResult.succeeded(
            invocation=invocation,
            encoded_output=outcome.encoded_output,
            artifacts=outcome.artifacts,
        )


def _copy_artifacts(
    artifacts: Iterable[ArtifactRef],
) -> tuple[ArtifactRef, ...]:
    copied_artifacts = tuple(artifacts)
    if any(
        not isinstance(artifact, ArtifactRef)
        for artifact in copied_artifacts
    ):
        raise TypeError(
            "scripted tool artifacts must contain ArtifactRef values"
        )
    artifact_ids = tuple(
        artifact.artifact_id for artifact in copied_artifacts
    )
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError(
            "scripted tool artifact IDs must not contain duplicates"
        )
    return copied_artifacts
