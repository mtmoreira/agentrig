"""Provider-independent capability protocols and data contracts."""

from agentrig.capabilities.base import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    CapabilityRequirements,
    DataRetention,
)
from agentrig.capabilities.text_generation import (
    GenerationUsage,
    ModelMetadata,
    TextGenerationFinishReason,
    TextGenerationRequest,
    TextGenerationResult,
    TextGenerator,
    TextMessage,
    TextMessageRole,
)

__all__ = (
    "CapabilityDescriptor",
    "CapabilityFeature",
    "CapabilityKind",
    "CapabilityLimit",
    "CapabilityRequirements",
    "DataRetention",
    "GenerationUsage",
    "ModelMetadata",
    "TextGenerationFinishReason",
    "TextGenerationRequest",
    "TextGenerationResult",
    "TextGenerator",
    "TextMessage",
    "TextMessageRole",
)
