"""Typed workflow adapter for provider-independent agents."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from agentrig.agents import Agent, AgentContract, AgentResult
from agentrig.core.context import RunContext
from agentrig.core.errors import Failure, normalize_exception
from agentrig.core.outcomes import ExecutionOutcome
from agentrig.workflow.base import Workflow
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


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowAgent(Generic[InputT, OutputT]):
    """Expose one configured workflow through the portable agent contract."""

    workflow: Workflow[InputT, OutputT]
    contract: AgentContract[InputT, OutputT]

    def __post_init__(self) -> None:
        if not isinstance(self.workflow, Workflow):
            raise TypeError("workflow agent workflow must satisfy Workflow")
        if not isinstance(self.contract, AgentContract):
            raise TypeError("workflow agent contract must be an AgentContract")

    async def run(
        self,
        input: InputT,
        context: RunContext,
    ) -> AgentResult[OutputT]:
        """Execute the workflow and translate its portable terminal outcome."""
        if not isinstance(context, RunContext):
            raise TypeError("workflow agent context must be a RunContext")

        constraint_failure = _check_constraints(context)
        if constraint_failure is not None:
            return AgentResult.from_failure(constraint_failure)

        try:
            outcome = await self.workflow.execute(input, context)
            if not isinstance(outcome, ExecutionOutcome):
                raise TypeError("workflow returned an invalid outcome")
            constraint_failure = _check_constraints(context)
            if constraint_failure is not None:
                return AgentResult.from_failure(
                    constraint_failure,
                    artifacts=outcome.artifacts,
                )
        except asyncio.CancelledError as error:
            return AgentResult.from_failure(normalize_exception(error))
        except Exception as error:
            return AgentResult.from_failure(normalize_exception(error))

        if not outcome.is_success:
            if outcome.failure is None:
                raise AssertionError("validated workflow failure has no details")
            return AgentResult.from_failure(
                outcome.failure,
                artifacts=outcome.artifacts,
            )
        return AgentResult.succeeded(
            outcome.unwrap(),
            artifacts=outcome.artifacts,
        )


def _check_constraints(context: RunContext) -> Failure | None:
    try:
        context.cancellation.raise_if_cancelled()
        if context.deadline is not None:
            context.deadline.raise_if_expired(context.clock)
    except asyncio.CancelledError as error:
        return normalize_exception(error)
    except Exception as error:
        return normalize_exception(error)
    return None
