"""Provider-independent autonomous agent contracts."""

from agentrig.agents.base import Agent
from agentrig.agents.catalog import (
    AgentRuntimeCatalog,
    AgentRuntimeRegistration,
)
from agentrig.agents.configured import (
    AgentInputCodec,
    AgentOutputCodec,
    ConfiguredAgent,
)
from agentrig.agents.contract import (
    AgentContract,
    AgentLimits,
    AgentResult,
    AgentStatus,
)
from agentrig.agents.runtime import (
    AgentExecutionRequest,
    AgentExecutionResult,
    AgentRuntime,
    AgentRuntimeUsage,
)

__all__ = (
    "Agent",
    "AgentContract",
    "AgentExecutionRequest",
    "AgentExecutionResult",
    "AgentInputCodec",
    "AgentLimits",
    "AgentOutputCodec",
    "AgentResult",
    "AgentRuntime",
    "AgentRuntimeCatalog",
    "AgentRuntimeRegistration",
    "AgentRuntimeUsage",
    "AgentStatus",
    "ConfiguredAgent",
)
