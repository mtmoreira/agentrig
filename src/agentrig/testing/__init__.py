"""Deterministic test doubles for AgentRig contracts."""

from agentrig.testing.scripted import ScriptedGrader, ScriptedGraderCall
from agentrig.testing.scripted_agent import (
    ScriptedAgentProgress,
    ScriptedAgentRuntime,
    ScriptedAgentRuntimeCall,
    ScriptedAgentScenario,
    ScriptedApprovalRequest,
    ScriptedToolRequest,
)

__all__ = (
    "ScriptedAgentProgress",
    "ScriptedAgentRuntime",
    "ScriptedAgentRuntimeCall",
    "ScriptedAgentScenario",
    "ScriptedApprovalRequest",
    "ScriptedGrader",
    "ScriptedGraderCall",
    "ScriptedToolRequest",
)
