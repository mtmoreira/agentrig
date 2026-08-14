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
from agentrig.testing.generation_contracts import (
    StructuredGeneratorContractSuite,
    TextGeneratorContractSuite,
)
from agentrig.testing.scripted_generation import (
    ScriptedStructuredGeneration,
    ScriptedStructuredGenerator,
    ScriptedStructuredGeneratorCall,
    ScriptedTextGenerator,
    ScriptedTextGeneratorCall,
)

__all__ = (
    "ScriptedAgentProgress",
    "ScriptedAgentRuntime",
    "ScriptedAgentRuntimeCall",
    "ScriptedAgentScenario",
    "ScriptedApprovalRequest",
    "ScriptedGrader",
    "ScriptedGraderCall",
    "ScriptedStructuredGeneration",
    "ScriptedStructuredGenerator",
    "ScriptedStructuredGeneratorCall",
    "ScriptedTextGenerator",
    "ScriptedTextGeneratorCall",
    "ScriptedToolRequest",
    "StructuredGeneratorContractSuite",
    "TextGeneratorContractSuite",
)
