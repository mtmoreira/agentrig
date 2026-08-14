"""Reusable contract probes for typed tool implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from agentrig.capabilities import (
    CapabilityKind,
    Tool,
    ToolContract,
    ToolInvocation,
    ToolResult,
)
from agentrig.core.context import RunContext
from agentrig.testing._capability_contracts import (
    InvocationCount,
    validate_contract_context,
    verify_cancellation_does_not_invoke,
    verify_preflight_does_not_invoke,
)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolContractSuite(Generic[InputT, OutputT]):
    """Portable checks shared by typed tool implementations."""

    tool: Tool[InputT, OutputT] = field(repr=False, compare=False)
    supported_invocation: ToolInvocation[InputT, OutputT] = field(
        repr=False
    )
    unsupported_invocation: ToolInvocation[InputT, OutputT] = field(
        repr=False
    )
    context: RunContext = field(repr=False)
    cancelled_context: RunContext = field(repr=False)
    invocation_count: InvocationCount = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.tool, Tool):
            raise TypeError("tool contract implementation must satisfy Tool")
        if not isinstance(self.tool.contract, ToolContract):
            raise TypeError(
                "tool contract implementation must expose a ToolContract"
            )
        if not isinstance(self.supported_invocation, ToolInvocation):
            raise TypeError(
                "tool contract supported_invocation must be a ToolInvocation"
            )
        if not isinstance(self.unsupported_invocation, ToolInvocation):
            raise TypeError(
                "tool contract unsupported_invocation must be a ToolInvocation"
            )
        if self.supported_invocation.contract is not self.tool.contract:
            raise ValueError(
                "tool contract supported invocation must use the exact contract"
            )
        if self.unsupported_invocation.contract is self.tool.contract:
            raise ValueError(
                "tool contract unsupported invocation must use another contract"
            )
        validate_contract_context(
            label="tool",
            descriptor=self.tool.contract.descriptor,
            expected_kind=CapabilityKind.TOOL,
            context=self.context,
            cancelled_context=self.cancelled_context,
            invocation_count=self.invocation_count,
        )

    async def verify(self) -> ToolResult[OutputT]:
        """Run shared binding, preflight, and cancellation checks."""
        result = await self.tool.invoke(
            self.supported_invocation,
            self.context,
        )
        if not isinstance(result, ToolResult):
            raise AssertionError("tool returned a non-ToolResult value")
        if result.invocation_id != self.supported_invocation.invocation_id:
            raise AssertionError(
                "tool result is not bound to the requested invocation"
            )
        contract = self.supported_invocation.contract
        if (
            result.tool_id != contract.tool_id
            or result.tool_version != contract.version
            or result.output_schema_id != contract.output_schema.schema_id
        ):
            raise AssertionError(
                "tool result is not bound to the invoked tool contract"
            )

        await verify_preflight_does_not_invoke(
            label="tool",
            operation=lambda: self.tool.invoke(
                self.unsupported_invocation,
                self.context,
            ),
            invocation_count=self.invocation_count,
        )
        await verify_cancellation_does_not_invoke(
            label="tool",
            operation=lambda: self.tool.invoke(
                self.supported_invocation,
                self.cancelled_context,
            ),
            invocation_count=self.invocation_count,
        )
        return result
