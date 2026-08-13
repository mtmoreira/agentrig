"""Scripted test doubles for deterministic grading scenarios."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from threading import Lock
from typing import Generic, TypeVar

from agentrig.core.errors import AgentRigError, Failure, FailureKind
from agentrig.core.grading import (
    Grade,
    GraderDescriptor,
    GradingContext,
)

SubjectT = TypeVar("SubjectT")

ScriptedGraderOutcome = Grade | Failure


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedGraderCall(Generic[SubjectT]):
    """One subject and context presented to a scripted grader."""

    index: int
    subject: SubjectT
    context: GradingContext


class ScriptedGrader(Generic[SubjectT]):
    """Return predefined grades or normalized failures in call order."""

    def __init__(
        self,
        *,
        descriptor: GraderDescriptor,
        outcomes: Iterable[ScriptedGraderOutcome],
        repeat_last: bool = False,
    ) -> None:
        if not isinstance(descriptor, GraderDescriptor):
            raise TypeError("scripted grader descriptor must be a GraderDescriptor")
        if not isinstance(repeat_last, bool):
            raise TypeError("scripted grader repeat_last must be a bool")

        copied_outcomes = tuple(outcomes)
        if not copied_outcomes:
            raise ValueError("scripted grader requires at least one outcome")
        for outcome in copied_outcomes:
            if isinstance(outcome, Grade):
                if outcome.grader != descriptor:
                    raise ValueError(
                        "scripted grade descriptor must match the scripted grader"
                    )
            elif isinstance(outcome, Failure):
                if outcome.kind is not FailureKind.GRADER_FAILED:
                    raise ValueError(
                        "scripted failures must use the grader_failed category"
                    )
            else:
                raise TypeError(
                    "scripted outcomes must contain Grade or Failure values"
                )

        self._descriptor = descriptor
        self._outcomes = copied_outcomes
        self._repeat_last = repeat_last
        self._next_outcome = 0
        self._calls: list[ScriptedGraderCall[SubjectT]] = []
        self._lock = Lock()

    @property
    def descriptor(self) -> GraderDescriptor:
        return self._descriptor

    @property
    def calls(self) -> tuple[ScriptedGraderCall[SubjectT], ...]:
        """Return a stable snapshot of recorded calls."""
        with self._lock:
            return tuple(self._calls)

    @property
    def is_exhausted(self) -> bool:
        """Whether another call would fail because the script is exhausted."""
        with self._lock:
            return (
                not self._repeat_last
                and self._next_outcome >= len(self._outcomes)
            )

    async def grade(
        self,
        subject: SubjectT,
        context: GradingContext,
    ) -> Grade:
        """Consume and return the next outcome after checking run constraints."""
        if not isinstance(context, GradingContext):
            raise TypeError("scripted grader context must be a GradingContext")
        run_context = context.run_context
        run_context.cancellation.raise_if_cancelled()
        if run_context.deadline is not None:
            run_context.deadline.raise_if_expired(run_context.clock)

        with self._lock:
            call = ScriptedGraderCall(
                index=len(self._calls),
                subject=subject,
                context=context,
            )
            self._calls.append(call)
            outcome = self._take_next_outcome()

        if outcome is None:
            raise AgentRigError(
                Failure(
                    kind=FailureKind.GRADER_FAILED,
                    message="scripted grader has no remaining outcomes",
                    code="scripted_grader.exhausted",
                    metadata={
                        "grader_id": self.descriptor.grader_id,
                        "grader_version": self.descriptor.version,
                    },
                )
            )
        if isinstance(outcome, Failure):
            raise AgentRigError(outcome)
        return outcome

    def _take_next_outcome(self) -> ScriptedGraderOutcome | None:
        if self._next_outcome >= len(self._outcomes):
            return None

        outcome = self._outcomes[self._next_outcome]
        if not (
            self._repeat_last
            and self._next_outcome == len(self._outcomes) - 1
        ):
            self._next_outcome += 1
        return outcome
