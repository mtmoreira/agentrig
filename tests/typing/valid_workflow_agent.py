"""Positive fixture for typed workflow-to-agent substitution."""

from agentrig.agents import Agent, AgentContract, AgentLimits
from agentrig.core import EffectProfile, RunContext
from agentrig.workflow import (
    FunctionStep,
    Sequence,
    StepDescriptor,
    Workflow,
    WorkflowAgent,
)


async def length(input: str, context: RunContext) -> int:
    del context
    return len(input)


workflow: Workflow[str, int] = Sequence(
    FunctionStep(
        descriptor=StepDescriptor(
            step_id="length",
            version="1",
            effect_profile=EffectProfile.READ_ONLY,
        ),
        function=length,
    )
)
contract: AgentContract[str, int] = AgentContract(
    agent_id="length-workflow",
    version="1",
    purpose="Measure one string",
    input_schema="example.text.v1",
    output_schema="example.length.v1",
    prompt_version="workflow-1",
    effect_profile=EffectProfile.READ_ONLY,
    limits=AgentLimits(max_turns=1, max_tool_calls=0),
    stopping_policy="workflow_completed",
)

agent: Agent[str, int] = WorkflowAgent(
    workflow=workflow,
    contract=contract,
)
