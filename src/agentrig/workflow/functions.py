"""Typed adapters from ordinary callables to workflow steps."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from inspect import iscoroutinefunction
from typing import Generic, Literal, TypeVar

from agentrig.core.context import RunContext
from agentrig.workflow.step import StepDescriptor

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

AsyncStepFunction = Callable[[InputT, RunContext], Awaitable[OutputT]]
SyncStepFunction = Callable[[InputT, RunContext], OutputT]


@dataclass(frozen=True, slots=True, init=False)
class FunctionStep(Generic[InputT, OutputT]):
    """Adapt a typed callable without hiding synchronous execution semantics."""

    descriptor: StepDescriptor
    _run: AsyncStepFunction[InputT, OutputT] = field(
        compare=False,
        repr=False,
    )

    def __init__(
        self,
        *,
        descriptor: StepDescriptor,
        function: AsyncStepFunction[InputT, OutputT],
    ) -> None:
        """Adapt an async callable with the standard step signature."""
        _require_descriptor(descriptor)
        _require_callable("async step function", function)
        if not _is_async_callable(function):
            raise TypeError("async step function must be an async callable")
        object.__setattr__(self, "descriptor", descriptor)
        object.__setattr__(self, "_run", function)

    @classmethod
    def from_sync(
        cls,
        *,
        descriptor: StepDescriptor,
        function: SyncStepFunction[InputT, OutputT],
        approved: Literal[True],
    ) -> FunctionStep[InputT, OutputT]:
        """Adapt a bounded sync callable after an explicit inline-run approval."""
        _require_descriptor(descriptor)
        _require_callable("sync step function", function)
        if approved is not True:
            raise ValueError("sync step function requires approved=True")
        if _is_async_callable(function):
            raise TypeError("sync step function must not be an async callable")

        async def run_sync(input: InputT, context: RunContext) -> OutputT:
            return function(input, context)

        instance = object.__new__(cls)
        object.__setattr__(instance, "descriptor", descriptor)
        object.__setattr__(instance, "_run", run_sync)
        return instance

    async def run(self, input: InputT, context: RunContext) -> OutputT:
        """Check inherited run constraints, then invoke the adapted callable."""
        if not isinstance(context, RunContext):
            raise TypeError("function step context must be a RunContext")
        context.cancellation.raise_if_cancelled()
        if context.deadline is not None:
            context.deadline.raise_if_expired(context.clock)
        return await self._run(input, context)


def _require_descriptor(descriptor: StepDescriptor) -> None:
    if not isinstance(descriptor, StepDescriptor):
        raise TypeError("function step descriptor must be a StepDescriptor")


def _require_callable(name: str, function: object) -> None:
    if not callable(function):
        raise TypeError(f"{name} must be callable")


def _is_async_callable(function: object) -> bool:
    return iscoroutinefunction(function) or iscoroutinefunction(
        getattr(function, "__call__", None)
    )
