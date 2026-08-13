"""Deterministic sequential composition with typed adjacent handoffs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast, overload

from agentrig.core.context import RunContext
from agentrig.core.outcomes import ExecutionOutcome
from agentrig.workflow.execution import execute_step
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
        """Execute and unwrap the final outcome for step-like composition."""
        return (await self.execute(input, context)).unwrap()

    async def execute(
        self,
        input: InputT,
        context: RunContext,
    ) -> ExecutionOutcome[OutputT]:
        """Execute in order and return the first failure or final output."""
        if not isinstance(context, RunContext):
            raise TypeError("sequence context must be a RunContext")

        current: Any = input
        for step in self._steps:
            step_context = context.derive_child()
            outcome = await execute_step(step, current, step_context)
            if not outcome.is_success:
                return cast(ExecutionOutcome[OutputT], outcome)
            current = outcome.unwrap()
        return ExecutionOutcome.succeeded(cast(OutputT, current))


def _require_step(step: object) -> None:
    if not isinstance(step, Step):
        raise TypeError("sequence values must satisfy the Step protocol")
    if not isinstance(step.descriptor, StepDescriptor):
        raise TypeError("sequence step descriptor must be a StepDescriptor")
