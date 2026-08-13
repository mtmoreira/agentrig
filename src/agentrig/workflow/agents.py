"""Typed workflow adapter for provider-independent agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from agentrig.agents import Agent, AgentContract, AgentResult
from agentrig.core.context import RunContext
from agentrig.workflow.step import StepDescriptor

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True, slots=True, init=False)
class AgentStep(Generic[InputT, OutputT]):
    """Expose one typed agent as an effect-aware workflow step."""

    agent: Agent[InputT, OutputT] = field(compare=False, repr=False)
    descriptor: StepDescriptor

    def __init__(self, agent: Agent[InputT, OutputT], /) -> None:
        if not isinstance(agent, Agent):
            raise TypeError("agent step value must satisfy the Agent protocol")
        contract = agent.contract
        if not isinstance(contract, AgentContract):
            raise TypeError("agent step contract must be an AgentContract")

        object.__setattr__(self, "agent", agent)
        object.__setattr__(
            self,
            "descriptor",
            StepDescriptor(
                step_id=contract.agent_id,
                version=contract.version,
                effect_profile=contract.effect_profile,
            ),
        )

    async def run(self, input: InputT, context: RunContext) -> OutputT:
        """Run the agent and raise its normalized failure at the step boundary."""
        if not isinstance(context, RunContext):
            raise TypeError("agent step context must be a RunContext")
        context.cancellation.raise_if_cancelled()
        if context.deadline is not None:
            context.deadline.raise_if_expired(context.clock)

        result = await self.agent.run(input, context)
        if not isinstance(result, AgentResult):
            raise TypeError("agent step agent must return an AgentResult")
        return result.unwrap()
