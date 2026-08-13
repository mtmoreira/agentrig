"""Typed executable workflow contract."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from agentrig.core.context import RunContext
from agentrig.core.outcomes import ExecutionOutcome

InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT")


@runtime_checkable
class Workflow(Protocol[InputT, OutputT]):
    """Execute typed control flow and return one normalized outcome."""

    async def execute(
        self,
        input: InputT,
        context: RunContext,
    ) -> ExecutionOutcome[OutputT]:
        """Return the workflow's typed output or normalized terminal failure."""
        ...
