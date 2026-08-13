"""Provider-independent autonomous agent contracts."""

from agentrig.agents.base import Agent
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
)

__all__ = (
    "Agent",
    "AgentContract",
    "AgentExecutionRequest",
    "AgentExecutionResult",
    "AgentLimits",
    "AgentResult",
    "AgentRuntime",
    "AgentStatus",
)
