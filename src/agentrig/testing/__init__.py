"""Deterministic test doubles for AgentRig contracts."""

from agentrig.testing.action_contracts import (
    CodingAgentContractSuite,
    ImageGeneratorContractSuite,
)
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
from agentrig.testing.scripted_actions import (
    ScriptedCodingAgent,
    ScriptedCodingAgentCall,
    ScriptedCodingScenario,
    ScriptedImageGeneration,
    ScriptedImageGenerator,
    ScriptedImageGeneratorCall,
)

__all__ = (
    "CodingAgentContractSuite",
    "ImageGeneratorContractSuite",
    "ScriptedAgentProgress",
    "ScriptedAgentRuntime",
    "ScriptedAgentRuntimeCall",
    "ScriptedAgentScenario",
    "ScriptedApprovalRequest",
    "ScriptedCodingAgent",
    "ScriptedCodingAgentCall",
    "ScriptedCodingScenario",
    "ScriptedGrader",
    "ScriptedGraderCall",
    "ScriptedImageGeneration",
    "ScriptedImageGenerator",
    "ScriptedImageGeneratorCall",
    "ScriptedStructuredGeneration",
    "ScriptedStructuredGenerator",
    "ScriptedStructuredGeneratorCall",
    "ScriptedTextGenerator",
    "ScriptedTextGeneratorCall",
    "ScriptedToolRequest",
    "StructuredGeneratorContractSuite",
    "TextGeneratorContractSuite",
)
