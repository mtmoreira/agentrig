"""Deterministic test doubles for AgentRig contracts."""

from agentrig.testing.action_contracts import (
    CodingAgentContractSuite,
    ImageGeneratorContractSuite,
)
from agentrig.testing.data_contracts import (
    RetrieverContractSuite,
    SearchProviderContractSuite,
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
from agentrig.testing.scripted_data import (
    ScriptedRetrievalScenario,
    ScriptedRetriever,
    ScriptedRetrieverCall,
    ScriptedSearchProvider,
    ScriptedSearchProviderCall,
    ScriptedSearchScenario,
)
from agentrig.testing.scripted_tool import (
    ScriptedTool,
    ScriptedToolCall,
    ScriptedToolFailure,
    ScriptedToolSuccess,
)
from agentrig.testing.tool_contracts import ToolContractSuite

__all__ = (
    "CodingAgentContractSuite",
    "ImageGeneratorContractSuite",
    "RetrieverContractSuite",
    "SearchProviderContractSuite",
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
    "ScriptedRetrievalScenario",
    "ScriptedRetriever",
    "ScriptedRetrieverCall",
    "ScriptedSearchProvider",
    "ScriptedSearchProviderCall",
    "ScriptedSearchScenario",
    "ScriptedStructuredGeneration",
    "ScriptedStructuredGenerator",
    "ScriptedStructuredGeneratorCall",
    "ScriptedTextGenerator",
    "ScriptedTextGeneratorCall",
    "ScriptedTool",
    "ScriptedToolCall",
    "ScriptedToolFailure",
    "ScriptedToolRequest",
    "ScriptedToolSuccess",
    "StructuredGeneratorContractSuite",
    "TextGeneratorContractSuite",
    "ToolContractSuite",
)
