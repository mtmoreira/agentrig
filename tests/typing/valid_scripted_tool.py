"""Positive fixture for scripted typed tools and their contract suite."""

from collections.abc import Mapping
from dataclasses import dataclass

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityKind,
    Tool,
    ToolContract,
    ToolInvocation,
    ToolSchema,
)
from agentrig.core import EffectProfile, JsonValue, RunContext
from agentrig.testing import (
    ScriptedTool,
    ScriptedToolSuccess,
    ToolContractSuite,
)


@dataclass(frozen=True, slots=True)
class AddInput:
    left: int
    right: int


@dataclass(frozen=True, slots=True)
class AddOutput:
    total: int


def encode_input(value: AddInput) -> JsonValue:
    return {"left": value.left, "right": value.right}


def decode_input(value: JsonValue) -> AddInput:
    if not isinstance(value, Mapping):
        raise ValueError("input must be an object")
    left = value.get("left")
    right = value.get("right")
    if not isinstance(left, int) or not isinstance(right, int):
        raise ValueError("input fields must be integers")
    return AddInput(left=left, right=right)


def encode_output(value: AddOutput) -> JsonValue:
    return {"total": value.total}


def decode_output(value: JsonValue) -> AddOutput:
    if not isinstance(value, Mapping):
        raise ValueError("output must be an object")
    total = value.get("total")
    if not isinstance(total, int):
        raise ValueError("output total must be an integer")
    return AddOutput(total=total)


contract = ToolContract(
    descriptor=CapabilityDescriptor(
        capability_id="math.add",
        version="1",
        kind=CapabilityKind.TOOL,
    ),
    purpose="Add two integers.",
    effect_profile=EffectProfile.READ_ONLY,
    input_schema=ToolSchema(
        schema_id="math.add.input.v1",
        json_schema={"type": "object"},
        encoder=encode_input,
        decoder=decode_input,
    ),
    output_schema=ToolSchema(
        schema_id="math.add.output.v1",
        json_schema={"type": "object"},
        encoder=encode_output,
        decoder=decode_output,
    ),
)
tool_fake = ScriptedTool[AddInput, AddOutput](
    contract=contract,
    outcomes=(ScriptedToolSuccess(encoded_output={"total": 5}),),
)
tool: Tool[AddInput, AddOutput] = tool_fake


def tool_suite(
    supported_invocation: ToolInvocation[AddInput, AddOutput],
    unsupported_invocation: ToolInvocation[AddInput, AddOutput],
    context: RunContext,
    cancelled_context: RunContext,
) -> ToolContractSuite[AddInput, AddOutput]:
    return ToolContractSuite(
        tool=tool,
        supported_invocation=supported_invocation,
        unsupported_invocation=unsupported_invocation,
        context=context,
        cancelled_context=cancelled_context,
        invocation_count=lambda: len(tool_fake.calls),
    )
