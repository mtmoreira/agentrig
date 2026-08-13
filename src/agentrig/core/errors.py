"""Normalized failure categories and safe exception conversion."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from agentrig.core._validation import freeze_string_map, require_trimmed_string
from agentrig.core.cancellation import RunCancelled
from agentrig.core.deadline import DeadlineExceeded


class FailureKind(StrEnum):
    """Stable, provider-independent failure classification."""

    INVALID_INPUT = "invalid_input"
    TRANSIENT_PROVIDER = "transient_provider"
    PERMANENT_PROVIDER = "permanent_provider"
    POLICY_REFUSAL = "policy_refusal"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_DENIED = "approval_denied"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CANCELLED = "cancelled"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    GRADER_FAILED = "grader_failed"
    WORKFLOW_BLOCKED = "workflow_blocked"
    UNEXPECTED = "unexpected"


@dataclass(frozen=True, slots=True, kw_only=True)
class Failure:
    """Sanitized failure details safe to retain in execution outcomes."""

    kind: FailureKind
    message: str
    code: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, FailureKind):
            raise TypeError("failure kind must be a FailureKind")
        require_trimmed_string("failure message", self.message)
        if self.code is not None:
            require_trimmed_string("failure code", self.code)
        object.__setattr__(
            self,
            "metadata",
            freeze_string_map("failure metadata", self.metadata),
        )

    @property
    def is_retryable(self) -> bool:
        """Whether the category is safe to consider for bounded retry."""
        return self.kind is FailureKind.TRANSIENT_PROVIDER


class AgentRigError(RuntimeError):
    """Exception wrapper for an already normalized failure."""

    def __init__(self, failure: Failure) -> None:
        if not isinstance(failure, Failure):
            raise TypeError("failure must be a Failure")
        self.failure = failure
        super().__init__(failure.message)


def normalize_exception(error: BaseException) -> Failure:
    """Convert an exception without retaining arbitrary exception text."""
    if isinstance(error, AgentRigError):
        return error.failure
    if isinstance(error, RunCancelled):
        return Failure(
            kind=FailureKind.CANCELLED,
            message=error.cancellation.reason,
        )
    if isinstance(error, asyncio.CancelledError):
        return Failure(
            kind=FailureKind.CANCELLED,
            message="execution cancelled",
        )
    if isinstance(error, DeadlineExceeded):
        return Failure(
            kind=FailureKind.DEADLINE_EXCEEDED,
            message="execution deadline exceeded",
            metadata={"expires_at": error.deadline.expires_at.isoformat()},
        )

    exception_type = type(error)
    return Failure(
        kind=FailureKind.UNEXPECTED,
        message="unexpected implementation failure",
        metadata={
            "exception_type": (
                f"{exception_type.__module__}.{exception_type.__qualname__}"
            )
        },
    )
