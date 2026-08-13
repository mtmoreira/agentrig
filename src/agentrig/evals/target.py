"""Typed execution boundary for systems under evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, Protocol, TypeVar, runtime_checkable

from agentrig.agents import Agent, AgentContract, AgentResult
from agentrig.core._validation import require_trimmed_string
from agentrig.core.context import RunContext
from agentrig.core.outcomes import ExecutionOutcome

InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT")


class EvalTargetKind(StrEnum):
    """Stable categories of systems that can be evaluated."""

    AGENT = "agent"
    WORKFLOW = "workflow"
    CAPABILITY = "capability"
    INTEGRATION = "integration"


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalTargetDescriptor:
    """Stable identity and category for one configured evaluation target."""

    target_id: str
    version: str
    kind: EvalTargetKind

    def __post_init__(self) -> None:
        require_trimmed_string("eval target ID", self.target_id)
        require_trimmed_string("eval target version", self.version)
        if not isinstance(self.kind, EvalTargetKind):
            raise TypeError("eval target kind must be an EvalTargetKind")


@runtime_checkable
class EvalTarget(Protocol[InputT, OutputT]):
    """Execute one typed case through a normalized evaluation boundary."""

    @property
    def descriptor(self) -> EvalTargetDescriptor:
        """Return the stable identity of this configured target."""
        ...

    async def run(
        self,
        input: InputT,
        context: RunContext,
    ) -> ExecutionOutcome[OutputT]:
        """Run one case input inside the eval runner's isolated context."""
        ...


@dataclass(frozen=True, slots=True, init=False)
class AgentEvalTarget(Generic[InputT, OutputT]):
    """Expose one typed agent through the normalized eval-target boundary."""

    agent: Agent[InputT, OutputT] = field(compare=False, repr=False)
    descriptor: EvalTargetDescriptor

    def __init__(self, agent: Agent[InputT, OutputT], /) -> None:
        if not isinstance(agent, Agent):
            raise TypeError("agent eval target value must satisfy Agent")
        contract = agent.contract
        if not isinstance(contract, AgentContract):
            raise TypeError("agent eval target contract must be an AgentContract")
        object.__setattr__(self, "agent", agent)
        object.__setattr__(
            self,
            "descriptor",
            EvalTargetDescriptor(
                target_id=contract.agent_id,
                version=contract.version,
                kind=EvalTargetKind.AGENT,
            ),
        )

    async def run(
        self,
        input: InputT,
        context: RunContext,
    ) -> ExecutionOutcome[OutputT]:
        """Run the agent and preserve its normalized outcome and artifacts."""
        if not isinstance(context, RunContext):
            raise TypeError("agent eval target context must be a RunContext")
        result = await self.agent.run(input, context)
        if not isinstance(result, AgentResult):
            raise TypeError("agent eval target must return an AgentResult")
        if result.is_success:
            return ExecutionOutcome.succeeded(
                result.unwrap(),
                artifacts=result.artifacts,
            )
        if result.failure is None:
            raise AssertionError("validated agent failure has no details")
        return ExecutionOutcome.from_failure(
            result.failure,
            artifacts=result.artifacts,
        )
