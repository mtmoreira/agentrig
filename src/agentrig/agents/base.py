"""Typed executable agent protocol."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from agentrig.agents.contract import AgentContract, AgentResult
from agentrig.core.context import RunContext

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@runtime_checkable
class Agent(Protocol[InputT, OutputT]):
    """Run one typed, goal-directed agent inside an explicit context."""

    @property
    def contract(self) -> AgentContract[InputT, OutputT]:
        """Return the stable contract for this configured agent."""
        ...

    async def run(
        self,
        input: InputT,
        context: RunContext,
    ) -> AgentResult[OutputT]:
        """Return a portable typed result without provider session details."""
        ...
