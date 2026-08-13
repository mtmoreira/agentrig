"""Positive fixture for typed adjacent sequence handoffs."""

from agentrig.core import RunContext
from agentrig.workflow import (
    EffectProfile,
    FunctionStep,
    Sequence,
    StepDescriptor,
)


async def length(input: str, context: RunContext) -> int:
    del context
    return len(input)


async def render(input: int, context: RunContext) -> str:
    del context
    return str(input)


length_step = FunctionStep(
    descriptor=StepDescriptor(
        step_id="length",
        version="1",
        effect_profile=EffectProfile.READ_ONLY,
    ),
    function=length,
)
render_step = FunctionStep(
    descriptor=StepDescriptor(
        step_id="render",
        version="1",
        effect_profile=EffectProfile.READ_ONLY,
    ),
    function=render,
)

sequence: Sequence[str, str] = Sequence(length_step, render_step)
