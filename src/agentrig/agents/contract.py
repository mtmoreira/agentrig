"""Stable, provider-independent agent contracts and result records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, TypeVar, cast

from agentrig.core._validation import freeze_string_map, require_trimmed_string
from agentrig.core.artifacts import ArtifactRef
from agentrig.core.effects import EffectProfile
from agentrig.core.errors import AgentRigError, Failure, FailureKind

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

_BLOCKING_FAILURE_KINDS = frozenset(
    {
        FailureKind.APPROVAL_REQUIRED,
        FailureKind.WORKFLOW_BLOCKED,
    }
)


class AgentStatus(StrEnum):
    """Stable terminal states returned by an agent boundary."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentLimits:
    """Provider-neutral bounds on autonomous agent execution."""

    max_turns: int
    max_tool_calls: int

    def __post_init__(self) -> None:
        _require_positive_integer("agent max_turns", self.max_turns)
        _require_non_negative_integer(
            "agent max_tool_calls",
            self.max_tool_calls,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentContract(Generic[InputT, OutputT]):
    """Versioned identity, schemas, authority, and limits for one agent."""

    agent_id: str
    version: str
    purpose: str
    input_schema: str
    output_schema: str
    prompt_version: str
    effect_profile: EffectProfile
    limits: AgentLimits
    stopping_policy: str
    allowed_tools: tuple[str, ...] = ()
    allowed_capabilities: tuple[str, ...] = ()
    permissions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_trimmed_string("agent ID", self.agent_id)
        require_trimmed_string("agent version", self.version)
        require_trimmed_string("agent purpose", self.purpose)
        require_trimmed_string("agent input schema", self.input_schema)
        require_trimmed_string("agent output schema", self.output_schema)
        require_trimmed_string("agent prompt version", self.prompt_version)
        if not isinstance(self.effect_profile, EffectProfile):
            raise TypeError("agent effect_profile must be an EffectProfile")
        if not isinstance(self.limits, AgentLimits):
            raise TypeError("agent limits must be AgentLimits")
        require_trimmed_string("agent stopping policy", self.stopping_policy)

        object.__setattr__(
            self,
            "allowed_tools",
            _copy_identifiers("allowed tool", self.allowed_tools),
        )
        object.__setattr__(
            self,
            "allowed_capabilities",
            _copy_identifiers(
                "allowed capability",
                self.allowed_capabilities,
            ),
        )
        object.__setattr__(
            self,
            "permissions",
            freeze_string_map("agent permissions", self.permissions),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentResult(Generic[OutputT]):
    """A portable typed agent output or normalized terminal failure."""

    status: AgentStatus
    output: OutputT | None = None
    failure: Failure | None = None
    artifacts: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, AgentStatus):
            raise TypeError("agent result status must be an AgentStatus")

        copied_artifacts = tuple(self.artifacts)
        for artifact in copied_artifacts:
            if not isinstance(artifact, ArtifactRef):
                raise TypeError("agent result artifacts must be ArtifactRef values")
        artifact_ids = tuple(
            artifact.artifact_id for artifact in copied_artifacts
        )
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("agent result artifacts must have unique IDs")
        object.__setattr__(self, "artifacts", copied_artifacts)

        if self.status is AgentStatus.SUCCEEDED:
            if self.failure is not None:
                raise ValueError("successful agent result must not have a failure")
            return

        if self.output is not None:
            raise ValueError("non-successful agent result must not have output")
        if not isinstance(self.failure, Failure):
            raise ValueError("non-successful agent result must have a failure")

        expected_status = _status_for_failure(self.failure.kind)
        if self.status is not expected_status:
            raise ValueError(
                f"{self.failure.kind.value} failure requires "
                f"{expected_status.value} agent status"
            )

    @classmethod
    def succeeded(
        cls,
        output: OutputT,
        *,
        artifacts: Iterable[ArtifactRef] = (),
    ) -> AgentResult[OutputT]:
        """Create a successful result, including successful ``None`` output."""
        return cls(
            status=AgentStatus.SUCCEEDED,
            output=output,
            artifacts=tuple(artifacts),
        )

    @classmethod
    def from_failure(
        cls,
        failure: Failure,
        *,
        artifacts: Iterable[ArtifactRef] = (),
    ) -> AgentResult[OutputT]:
        """Create the deterministic terminal status for a failure category."""
        if not isinstance(failure, Failure):
            raise TypeError("failure must be a Failure")
        return cls(
            status=_status_for_failure(failure.kind),
            failure=failure,
            artifacts=tuple(artifacts),
        )

    @property
    def is_success(self) -> bool:
        return self.status is AgentStatus.SUCCEEDED

    def unwrap(self) -> OutputT:
        """Return successful output or raise the normalized failure."""
        if self.status is AgentStatus.SUCCEEDED:
            return cast(OutputT, self.output)
        if self.failure is None:
            raise AssertionError("validated non-successful result has no failure")
        raise AgentRigError(self.failure)


def _copy_identifiers(field_name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    copied = tuple(values)
    for value in copied:
        require_trimmed_string(field_name, value)
    if len(copied) != len(set(copied)):
        raise ValueError(f"{field_name}s must not contain duplicates")
    return copied


def _require_positive_integer(field_name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_non_negative_integer(field_name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _status_for_failure(kind: FailureKind) -> AgentStatus:
    if kind is FailureKind.CANCELLED:
        return AgentStatus.CANCELLED
    if kind in _BLOCKING_FAILURE_KINDS:
        return AgentStatus.BLOCKED
    return AgentStatus.FAILED
