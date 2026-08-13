"""Negative fixture: an int output cannot feed a bytes input."""

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


async def decode(input: bytes, context: RunContext) -> str:
    del context
    return input.decode()


length_step = FunctionStep(
    descriptor=StepDescriptor(
        step_id="length",
        version="1",
        effect_profile=EffectProfile.READ_ONLY,
    ),
    function=length,
)
decode_step = FunctionStep(
    descriptor=StepDescriptor(
        step_id="decode",
        version="1",
        effect_profile=EffectProfile.READ_ONLY,
    ),
    function=decode,
)

sequence: Sequence[str, str] = Sequence(length_step, decode_step)
