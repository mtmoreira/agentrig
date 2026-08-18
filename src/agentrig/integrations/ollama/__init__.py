"""Ollama integration contracts and runtime adapter."""

from agentrig.integrations.ollama.ollama import (
    OLLAMA_AGENT_RUNTIME_CAPABILITY,
    OLLAMA_CLIENT_VERSION,
    OllamaAuthenticationSource,
    OllamaChatMessage,
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaClient,
    OllamaClientFactory,
    OllamaFinishReason,
    OllamaRuntimeOptions,
)
from agentrig.integrations.ollama.runtime import OllamaAgentRuntime

__all__ = (
    "OLLAMA_AGENT_RUNTIME_CAPABILITY",
    "OLLAMA_CLIENT_VERSION",
    "OllamaAgentRuntime",
    "OllamaAuthenticationSource",
    "OllamaChatMessage",
    "OllamaChatRequest",
    "OllamaChatResponse",
    "OllamaClient",
    "OllamaClientFactory",
    "OllamaFinishReason",
    "OllamaRuntimeOptions",
)
