"""Positive fixture for application-scoped runtime catalog typing."""

from __future__ import annotations

from dataclasses import dataclass

from agentrig.agents import (
    AgentExecutionRequest,
    AgentExecutionResult,
    AgentRuntime,
    AgentRuntimeCatalog,
    AgentRuntimeRegistration,
)
from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityKind,
    CapabilityRequirements,
)
from agentrig.core import RunContext


@dataclass(frozen=True)
class ExampleRuntime:
    async def execute(
        self,
        request: AgentExecutionRequest,
        context: RunContext,
    ) -> AgentExecutionResult:
        del request, context
        return AgentExecutionResult.succeeded({"status": "complete"})


descriptor = CapabilityDescriptor(
    capability_id="example.agent_runtime",
    version="1",
    kind=CapabilityKind.AGENT_RUNTIME,
)
catalog = AgentRuntimeCatalog(
    (
        AgentRuntimeRegistration(
            binding_id="example-runtime",
            descriptor=descriptor,
            runtime=ExampleRuntime(),
        ),
    )
)
runtime: AgentRuntime = catalog.resolve(
    "example-runtime",
    CapabilityRequirements(kind=CapabilityKind.AGENT_RUNTIME),
)
