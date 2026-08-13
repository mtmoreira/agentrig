"""Outcome-capturing workflow execution with safe lifecycle events."""

from __future__ import annotations

import asyncio
from typing import Any, TypeVar

from agentrig.core.context import RunContext
from agentrig.core.errors import Failure, normalize_exception
from agentrig.core.events import Event, EventKind, JsonValue
from agentrig.core.outcomes import ExecutionOutcome
from agentrig.workflow.step import Step

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


async def execute_step(
    step: Step[InputT, OutputT],
    input: InputT,
    context: RunContext,
    *,
    attempt: int = 1,
) -> ExecutionOutcome[OutputT]:
    """Execute one step, returning a normalized outcome and lifecycle events."""
    if not isinstance(step, Step):
        raise TypeError("executed value must satisfy the Step protocol")
    if not isinstance(context, RunContext):
        raise TypeError("step execution context must be a RunContext")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("step attempt must be a positive integer")

    attributes = _step_attributes(step, attempt=attempt)
    _emit(context, EventKind.STEP_STARTED, attributes)

    try:
        context.cancellation.raise_if_cancelled()
        if context.deadline is not None:
            context.deadline.raise_if_expired(context.clock)
        output = await step.run(input, context)
    except asyncio.CancelledError as error:
        return _failed_outcome(context, attributes, error)
    except Exception as error:
        return _failed_outcome(context, attributes, error)

    outcome = ExecutionOutcome.succeeded(output)
    _emit(
        context,
        EventKind.STEP_COMPLETED,
        {**attributes, "status": outcome.status.value},
    )
    return outcome


def _failed_outcome(
    context: RunContext,
    attributes: dict[str, JsonValue],
    error: BaseException,
) -> ExecutionOutcome[OutputT]:
    failure = normalize_exception(error)
    outcome: ExecutionOutcome[OutputT] = ExecutionOutcome.from_failure(failure)
    _emit(
        context,
        EventKind.STEP_COMPLETED,
        {
            **attributes,
            "status": outcome.status.value,
            "failure_kind": failure.kind.value,
            **_failure_code_attribute(failure),
        },
    )
    return outcome


def _step_attributes(
    step: Step[Any, Any],
    *,
    attempt: int,
) -> dict[str, JsonValue]:
    descriptor = step.descriptor
    return {
        "step_id": descriptor.step_id,
        "step_version": descriptor.version,
        "effect_profile": descriptor.effect_profile.value,
        "attempt": attempt,
    }


def _failure_code_attribute(failure: Failure) -> dict[str, JsonValue]:
    if failure.code is None:
        return {}
    return {"failure_code": failure.code}


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
