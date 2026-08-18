"""Positive fixture for application-scoped runtime catalog typing."""

from __future__ import annotations

from dataclasses import dataclass

from agentrig.agents import (
    AgentExecutionRequest,
    AgentExecutionResult,
    AgentRuntime,
    AgentRuntimeCatalog,
    AgentRuntimeRegistration,
    AgentRuntimeUsage,
)
from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
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
        return AgentExecutionResult.succeeded(
            {"status": "complete"},
            usage=AgentRuntimeUsage(input_tokens=3, output_tokens=2),
        )


descriptor = CapabilityDescriptor(
    capability_id="example.agent_runtime",
    version="1",
    kind=CapabilityKind.AGENT_RUNTIME,
    features=frozenset({CapabilityFeature.USAGE_REPORTING}),
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
