"""Bounded retries for classified failures and repeatable step effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from agentrig.core.context import RunContext
from agentrig.core.events import EventKind
from agentrig.core.outcomes import ExecutionOutcome
from agentrig.workflow.execution import (
    _emit,
    _failure_code_attribute,
    _step_attributes,
    execute_step,
)
from agentrig.workflow.step import Step

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True, slots=True, kw_only=True)
class RetryPolicy:
    """Limit total attempts, including the initial execution."""

    max_attempts: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("retry max_attempts must be a positive integer")


NO_RETRY_POLICY = RetryPolicy()


async def execute_step_with_retry(
    step: Step[InputT, OutputT],
    input: InputT,
    context: RunContext,
    *,
    retry_policy: RetryPolicy,
) -> ExecutionOutcome[OutputT]:
    """Execute one operation with bounded, effect-aware transient retries."""
    if not isinstance(retry_policy, RetryPolicy):
        raise TypeError("retry_policy must be a RetryPolicy")

    for attempt in range(1, retry_policy.max_attempts + 1):
        outcome = await execute_step(step, input, context, attempt=attempt)
        if outcome.is_success:
            return outcome

        failure = outcome.failure
        if failure is None:
            raise AssertionError("non-successful outcome has no failure")
        if (
            attempt == retry_policy.max_attempts
            or not failure.is_retryable
            or not step.descriptor.effect_profile.allows_automatic_retry
        ):
            return outcome

        _emit(
            context,
            EventKind.RETRY_SCHEDULED,
            {
                **_step_attributes(step, attempt=attempt),
                "next_attempt": attempt + 1,
                "max_attempts": retry_policy.max_attempts,
                "failure_kind": failure.kind.value,
                **_failure_code_attribute(failure),
            },
        )

    raise AssertionError("validated retry policy produced no attempts")
