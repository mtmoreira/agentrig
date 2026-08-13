"""Positive fixture for typed agent-to-workflow composition."""

from agentrig.agents import AgentContract, AgentLimits, AgentResult
from agentrig.core import EffectProfile, RunContext
from agentrig.workflow import (
    AgentStep,
    FunctionStep,
    Sequence,
    Step,
    StepDescriptor,
)


class EchoAgent:
    @property
    def contract(self) -> AgentContract[str, str]:
        return AgentContract(
            agent_id="echo",
            version="1",
            purpose="Echo one string",
            input_schema="example.text.v1",
            output_schema="example.text.v1",
            prompt_version="prompt-1",
            effect_profile=EffectProfile.READ_ONLY,
            limits=AgentLimits(max_turns=1, max_tool_calls=0),
            stopping_policy="output_schema_satisfied",
        )

    async def run(
        self,
        input: str,
        context: RunContext,
    ) -> AgentResult[str]:
        del context
        return AgentResult.succeeded(input)


async def length(input: str, context: RunContext) -> int:
    del context
    return len(input)


agent_step: Step[str, str] = AgentStep(EchoAgent())
length_step = FunctionStep(
    descriptor=StepDescriptor(
        step_id="length",
        version="1",
        effect_profile=EffectProfile.READ_ONLY,
    ),
    function=length,
)

sequence: Sequence[str, int] = Sequence(agent_step, length_step)
