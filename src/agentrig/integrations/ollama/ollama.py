"""Stable injected boundary for the optional Ollama agent runtime."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    DataRetention,
)
from agentrig.core._json import JsonValue, freeze_json_object
from agentrig.core._validation import require_trimmed_string

OLLAMA_CLIENT_VERSION = "0.6.2"

OLLAMA_AGENT_RUNTIME_CAPABILITY = CapabilityDescriptor(
    capability_id="ollama.agent_runtime",
    version=OLLAMA_CLIENT_VERSION,
    kind=CapabilityKind.AGENT_RUNTIME,
    features=frozenset(
        {
            CapabilityFeature.CANCELLATION,
            CapabilityFeature.STRUCTURED_OUTPUT,
        }
    ),
    data_retention=DataRetention.UNKNOWN,
)


@runtime_checkable
class OllamaAuthenticationSource(Protocol):
    """Resolve private authorization headers at client creation."""

    def resolve_headers(self) -> Mapping[str, str]:
        """Return only application-authorized transport headers."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class OllamaChatMessage:
    """One private chat message passed through the injected client seam."""

    role: str
    content: str = field(repr=False)

    def __post_init__(self) -> None:
        if self.role not in {"system", "user"}:
            raise ValueError("Ollama message role must be system or user")
        require_trimmed_string("Ollama message content", self.content)


@dataclass(frozen=True, slots=True, kw_only=True)
class OllamaRuntimeOptions:
    """Safe model settings fixed by one application runtime binding."""

    temperature: float | None = None
    seed: int | None = None
    max_output_tokens: int | None = None
    keep_alive: float | str | None = None

    def __post_init__(self) -> None:
        if self.temperature is not None:
            if (
                isinstance(self.temperature, bool)
                or not isinstance(self.temperature, (int, float))
                or not math.isfinite(self.temperature)
                or self.temperature < 0
            ):
                raise ValueError(
                    "Ollama temperature must be a non-negative finite number"
                )
            object.__setattr__(self, "temperature", float(self.temperature))
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("Ollama seed must be a non-negative integer")
        if self.max_output_tokens is not None and (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens < 1
        ):
            raise ValueError(
                "Ollama max output tokens must be a positive integer"
            )
        if self.keep_alive is not None:
            if isinstance(self.keep_alive, bool):
                raise ValueError("Ollama keep_alive must be text or a number")
            if isinstance(self.keep_alive, str):
                require_trimmed_string("Ollama keep_alive", self.keep_alive)
            elif not isinstance(self.keep_alive, (int, float)) or not math.isfinite(
                self.keep_alive
            ):
                raise ValueError("Ollama keep_alive must be text or a number")
            else:
                object.__setattr__(self, "keep_alive", float(self.keep_alive))

    def to_provider_options(self) -> Mapping[str, JsonValue]:
        """Return a fresh safe mapping understood by the Ollama client."""
        options: dict[str, JsonValue] = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.seed is not None:
            options["seed"] = self.seed
        if self.max_output_tokens is not None:
            options["num_predict"] = self.max_output_tokens
        return freeze_json_object("Ollama runtime options", options)


class OllamaFinishReason(StrEnum):
    """Safe normalized reasons for one terminal Ollama response."""

    STOP = "stop"
    LENGTH = "length"
    OTHER = "other"


@dataclass(frozen=True, slots=True, kw_only=True)
class OllamaChatRequest:
    """One strict non-streaming structured chat request."""

    model: str
    messages: tuple[OllamaChatMessage, ...] = field(repr=False)
    output_schema: Mapping[str, JsonValue] = field(repr=False)
    options: Mapping[str, JsonValue] = field(default_factory=dict)
    keep_alive: float | str | None = None

    def __post_init__(self) -> None:
        require_trimmed_string("Ollama model", self.model)
        messages = tuple(self.messages)
        if not messages:
            raise ValueError("Ollama chat requires at least one message")
        if any(not isinstance(message, OllamaChatMessage) for message in messages):
            raise TypeError("Ollama chat messages must be OllamaChatMessage values")
        schema = freeze_json_object("Ollama output schema", self.output_schema)
        if not schema:
            raise ValueError("Ollama output schema must not be empty")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "output_schema", schema)
        object.__setattr__(
            self,
            "options",
            freeze_json_object("Ollama chat options", self.options),
        )
        if self.keep_alive is not None:
            OllamaRuntimeOptions(keep_alive=self.keep_alive)


@dataclass(frozen=True, slots=True, kw_only=True)
class OllamaChatResponse:
    """Safe structured fields detached from the raw Ollama response."""

    content: str = field(repr=False)
    model: str
    finish_reason: OllamaFinishReason = OllamaFinishReason.OTHER
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        require_trimmed_string("Ollama response content", self.content)
        require_trimmed_string("Ollama response model", self.model)
        if not isinstance(self.finish_reason, OllamaFinishReason):
            raise TypeError(
                "Ollama finish reason must be an OllamaFinishReason"
            )
        for name, value in (
            ("input tokens", self.input_tokens),
            ("output tokens", self.output_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Ollama {name} must be a non-negative integer")


@runtime_checkable
class OllamaClient(Protocol):
    """Minimal asynchronous Ollama client used by the runtime adapter."""

    async def chat(self, request: OllamaChatRequest) -> OllamaChatResponse:
        """Execute one non-streaming structured chat request."""
        ...

    async def close(self) -> None:
        """Release transport resources."""
        ...


@runtime_checkable
class OllamaClientFactory(Protocol):
    """Construct one short-lived Ollama client without global state."""

    def create(self) -> OllamaClient:
        """Create a client after runtime preflight succeeds."""
        ...
