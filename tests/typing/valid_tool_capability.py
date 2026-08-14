"""Positive fixture for typed tool invocation and results."""

from collections.abc import Mapping
from dataclasses import dataclass

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityKind,
    Tool,
    ToolContract,
    ToolInvocation,
    ToolResult,
    ToolSchema,
)
from agentrig.core import EffectProfile, JsonValue, RunContext


@dataclass(frozen=True)
class AddInput:
    left: int
    right: int


@dataclass(frozen=True)
class AddOutput:
    total: int


def encode_input(value: AddInput) -> JsonValue:
    return {"left": value.left, "right": value.right}


def decode_input(value: JsonValue) -> AddInput:
    if not isinstance(value, Mapping):
        raise ValueError("input must be an object")
    left = value.get("left")
    right = value.get("right")
    if (
        isinstance(left, bool)
        or not isinstance(left, int)
        or isinstance(right, bool)
        or not isinstance(right, int)
    ):
        raise ValueError("input fields must be integers")
    return AddInput(left=left, right=right)


def encode_output(value: AddOutput) -> JsonValue:
    return {"total": value.total}


def decode_output(value: JsonValue) -> AddOutput:
    if not isinstance(value, Mapping):
        raise ValueError("output must be an object")
    total = value.get("total")
    if isinstance(total, bool) or not isinstance(total, int):
        raise ValueError("output total must be an integer")
    return AddOutput(total=total)


contract = ToolContract(
    descriptor=CapabilityDescriptor(
        capability_id="math.add",
        version="1",
        kind=CapabilityKind.TOOL,
    ),
    purpose="Add two bounded integers.",
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


@dataclass(frozen=True)
class AddTool:
    contract: ToolContract[AddInput, AddOutput]

    async def invoke(
        self,
        invocation: ToolInvocation[AddInput, AddOutput],
        context: RunContext,
    ) -> ToolResult[AddOutput]:
        del context
        return ToolResult.succeeded(
            invocation=invocation,
            encoded_output={
                "total": invocation.input.left + invocation.input.right
            },
        )


tool: Tool[AddInput, AddOutput] = AddTool(contract=contract)


async def invoke(
    invocation: ToolInvocation[AddInput, AddOutput],
    context: RunContext,
) -> AddOutput:
    return (await tool.invoke(invocation, context)).unwrap()
