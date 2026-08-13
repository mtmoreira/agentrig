"""Deterministic sequential composition with typed adjacent handoffs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast, overload

from agentrig.core.context import RunContext
from agentrig.workflow.step import Step, StepDescriptor

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
NextOutputT = TypeVar("NextOutputT")
Intermediate1T = TypeVar("Intermediate1T")
Intermediate2T = TypeVar("Intermediate2T")
Intermediate3T = TypeVar("Intermediate3T")
Intermediate4T = TypeVar("Intermediate4T")
Intermediate5T = TypeVar("Intermediate5T")


@dataclass(frozen=True, slots=True, init=False)
class Sequence(Generic[InputT, OutputT]):
    """Pass each step's typed output to the next step in declaration order."""

    _steps: tuple[Step[Any, Any], ...]

    @overload
    def __init__(self, first: Step[InputT, OutputT], /) -> None: ...

    @overload
    def __init__(
        self,
        first: Step[InputT, Intermediate1T],
        second: Step[Intermediate1T, OutputT],
        /,
    ) -> None: ...

    @overload
    def __init__(
        self,
        first: Step[InputT, Intermediate1T],
        second: Step[Intermediate1T, Intermediate2T],
        third: Step[Intermediate2T, OutputT],
        /,
    ) -> None: ...

    @overload
    def __init__(
        self,
        first: Step[InputT, Intermediate1T],
        second: Step[Intermediate1T, Intermediate2T],
        third: Step[Intermediate2T, Intermediate3T],
        fourth: Step[Intermediate3T, OutputT],
        /,
    ) -> None: ...

    @overload
    def __init__(
        self,
        first: Step[InputT, Intermediate1T],
        second: Step[Intermediate1T, Intermediate2T],
        third: Step[Intermediate2T, Intermediate3T],
        fourth: Step[Intermediate3T, Intermediate4T],
        fifth: Step[Intermediate4T, OutputT],
        /,
    ) -> None: ...

    @overload
    def __init__(
        self,
        first: Step[InputT, Intermediate1T],
        second: Step[Intermediate1T, Intermediate2T],
        third: Step[Intermediate2T, Intermediate3T],
        fourth: Step[Intermediate3T, Intermediate4T],
        fifth: Step[Intermediate4T, Intermediate5T],
        sixth: Step[Intermediate5T, OutputT],
        /,
    ) -> None: ...

    def __init__(self, *steps: Step[Any, Any]) -> None:
        copied_steps = tuple(steps)
        if not copied_steps:
            raise ValueError("sequence requires at least one step")
        for step in copied_steps:
            _require_step(step)
        object.__setattr__(self, "_steps", copied_steps)

    @property
    def steps(self) -> tuple[Step[Any, Any], ...]:
        """Return the immutable steps in execution order."""
        return self._steps

    def then(
        self,
        step: Step[OutputT, NextOutputT],
    ) -> Sequence[InputT, NextOutputT]:
        """Return a new sequence with one type-compatible trailing step."""
        _require_step(step)
        sequence = object.__new__(Sequence)
        object.__setattr__(sequence, "_steps", (*self._steps, step))
        return cast(Sequence[InputT, NextOutputT], sequence)

    async def run(self, input: InputT, context: RunContext) -> OutputT:
        """Execute in order, propagating constraints and stopping on failure."""
        if not isinstance(context, RunContext):
            raise TypeError("sequence context must be a RunContext")

        current: Any = input
        for step in self._steps:
            context.cancellation.raise_if_cancelled()
            if context.deadline is not None:
                context.deadline.raise_if_expired(context.clock)
            step_context = context.derive_child()
            current = await step.run(current, step_context)
        return cast(OutputT, current)


def _require_step(step: object) -> None:
    if not isinstance(step, Step):
        raise TypeError("sequence values must satisfy the Step protocol")
    if not isinstance(step.descriptor, StepDescriptor):
        raise TypeError("sequence step descriptor must be a StepDescriptor")
