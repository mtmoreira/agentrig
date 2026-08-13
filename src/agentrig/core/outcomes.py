"""Typed execution outcomes with normalized terminal states."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar, cast

from agentrig.core.artifacts import ArtifactRef
from agentrig.core.errors import AgentRigError, Failure, FailureKind

OutputT = TypeVar("OutputT")

_BLOCKING_FAILURE_KINDS = frozenset(
    {
        FailureKind.APPROVAL_REQUIRED,
        FailureKind.WORKFLOW_BLOCKED,
    }
)


class ExecutionStatus(StrEnum):
    """Stable top-level states returned by execution boundaries."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionOutcome(Generic[OutputT]):
    """A typed result or normalized terminal failure with artifacts."""

    status: ExecutionStatus
    output: OutputT | None = None
    failure: Failure | None = None
    artifacts: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, ExecutionStatus):
            raise TypeError("execution status must be an ExecutionStatus")

        copied_artifacts = tuple(self.artifacts)
        for artifact in copied_artifacts:
            if not isinstance(artifact, ArtifactRef):
                raise TypeError("outcome artifacts must be ArtifactRef values")
        artifact_ids = tuple(
            artifact.artifact_id for artifact in copied_artifacts
        )
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("outcome artifacts must have unique IDs")
        object.__setattr__(self, "artifacts", copied_artifacts)

        if self.status is ExecutionStatus.SUCCEEDED:
            if self.failure is not None:
                raise ValueError("successful outcome must not have a failure")
            return

        if self.output is not None:
            raise ValueError("non-successful outcome must not have output")
        if not isinstance(self.failure, Failure):
            raise ValueError("non-successful outcome must have a failure")

        expected_status = _status_for_failure(self.failure.kind)
        if self.status is not expected_status:
            raise ValueError(
                f"{self.failure.kind.value} failure requires "
                f"{expected_status.value} status"
            )

    @classmethod
    def succeeded(
        cls,
        output: OutputT,
        *,
        artifacts: Iterable[ArtifactRef] = (),
    ) -> ExecutionOutcome[OutputT]:
        """Create a successful outcome, including successful ``None`` output."""
        return cls(
            status=ExecutionStatus.SUCCEEDED,
            output=output,
            artifacts=tuple(artifacts),
        )

    @classmethod
    def from_failure(
        cls,
        failure: Failure,
        *,
        artifacts: Iterable[ArtifactRef] = (),
    ) -> ExecutionOutcome[OutputT]:
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
        return self.status is ExecutionStatus.SUCCEEDED

    def unwrap(self) -> OutputT:
        """Return successful output or raise the normalized failure."""
        if self.status is ExecutionStatus.SUCCEEDED:
            return cast(OutputT, self.output)
        if self.failure is None:
            raise AssertionError("validated non-successful outcome has no failure")
        raise AgentRigError(self.failure)


def _status_for_failure(kind: FailureKind) -> ExecutionStatus:
    if kind is FailureKind.CANCELLED:
        return ExecutionStatus.CANCELLED
    if kind in _BLOCKING_FAILURE_KINDS:
        return ExecutionStatus.BLOCKED
    return ExecutionStatus.FAILED
