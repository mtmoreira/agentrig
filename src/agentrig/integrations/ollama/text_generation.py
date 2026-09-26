"""Raw schema-constrained Ollama text through the portable TextGenerator contract."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    DataRetention,
    GenerationUsage,
    ModelMetadata,
    TextGenerationFinishReason,
    TextGenerationRequest,
    TextGenerationResult,
)
from agentrig.core import AgentRigError, Failure, FailureKind, RunContext, normalize_exception
from agentrig.integrations.ollama.ollama import (
    OLLAMA_CLIENT_VERSION,
    OllamaChatMessage,
    OllamaChatRequest,
    OllamaClient,
    OllamaClientFactory,
    OllamaFinishReason,
    OllamaRuntimeOptions,
)
from agentrig.integrations.ollama.runtime import _chat_with_constraints

_CLIENT_CLOSE_GRACE_SECONDS = 1.0

OLLAMA_TEXT_GENERATION_CAPABILITY = CapabilityDescriptor(
    capability_id="ollama.text_generation",
    version=OLLAMA_CLIENT_VERSION,
    kind=CapabilityKind.TEXT_GENERATION,
    features=frozenset({
        CapabilityFeature.MESSAGE_INPUT,
        CapabilityFeature.STRUCTURED_OUTPUT,
        CapabilityFeature.CANCELLATION,
        CapabilityFeature.USAGE_REPORTING,
    }),
    limits={CapabilityLimit.MAX_OUTPUT_TOKENS: 8192},
    data_retention=DataRetention.UNKNOWN,
)


class OllamaTextGenerator:
    """Return provider content verbatim; callers own decoding and repair.

    Binding-level options are fixed by the application. An output schema is
    required for each request, but this adapter deliberately does not parse the
    returned text or retry. The caller retains semantic authority.
    """

    def __init__(
        self,
        *,
        client_factory: OllamaClientFactory,
        model: str,
        options: OllamaRuntimeOptions | None = None,
    ) -> None:
        if not isinstance(client_factory, OllamaClientFactory):
            raise TypeError("Ollama text client factory is invalid")
        if not isinstance(model, str) or not model or model != model.strip():
            raise ValueError("Ollama text model must be nonempty and trimmed")
        if options is not None and not isinstance(options, OllamaRuntimeOptions):
            raise TypeError("Ollama text options are invalid")
        self._client_factory = client_factory
        self._model = model
        self._options = options if options is not None else OllamaRuntimeOptions()

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return OLLAMA_TEXT_GENERATION_CAPABILITY

    async def generate(
        self,
        request: TextGenerationRequest,
        context: RunContext,
    ) -> TextGenerationResult:
        if not isinstance(request, TextGenerationRequest):
            raise TypeError("Ollama text request is invalid")
        if not isinstance(context, RunContext):
            raise TypeError("Ollama text context is invalid")
        try:
            context.cancellation.raise_if_cancelled()
            if context.deadline is not None:
                context.deadline.raise_if_expired(context.clock)
        except Exception as error:
            raise AgentRigError(normalize_exception(error)) from None
        request.require_supported_by(self.descriptor)
        if request.output_schema is None:
            raise AgentRigError(Failure(
                kind=FailureKind.INVALID_INPUT,
                code="ollama.output_schema_required",
                message="Ollama text generation requires an output schema",
            ))

        options = self._options
        if request.max_output_tokens is not None:
            options = replace(options, max_output_tokens=request.max_output_tokens)
        messages = (
            (OllamaChatMessage(role="user", content=request.prompt),)
            if request.prompt is not None else tuple(
                OllamaChatMessage(role=item.role.value, content=item.text)
                for item in request.messages
                if item.text is not None
            )
        )
        chat_request = OllamaChatRequest(
            model=self._model,
            messages=messages,
            output_schema=request.output_schema,
            options=options.to_provider_options(),
            think=options.think,
            keep_alive=options.keep_alive,
            capture_reasoning=(context.private_trace_capture is not None
                               and context.private_trace_capture.include_reasoning),
        )
        capture = context.private_trace_capture
        if capture is not None:
            capture.record(context, kind="provider.request", metadata={
                "provider": "ollama", "model": self._model,
            }, content={
                "messages": [{"role": item.role, "content": item.content}
                             for item in chat_request.messages],
                "output_schema": dict(chat_request.output_schema),
                "options": dict(chat_request.options), "think": chat_request.think,
                "stream": False,
            })
        client: OllamaClient | None = None
        try:
            client = self._client_factory.create()
            response = await _chat_with_constraints(client, chat_request, context)
            if capture is not None:
                capture.record(context, kind="provider.response", metadata={
                    "provider": "ollama", "model": self._model,
                }, content={
                    "content": response.content,
                    "reported_model": response.model,
                    "thinking": response.thinking if capture.include_reasoning else None,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                })
            finish_reason = {
                OllamaFinishReason.STOP: TextGenerationFinishReason.COMPLETED,
                OllamaFinishReason.LENGTH: TextGenerationFinishReason.LENGTH,
            }.get(response.finish_reason, TextGenerationFinishReason.OTHER)
            return TextGenerationResult(
                text=response.content,
                usage=GenerationUsage(
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                ),
                model=ModelMetadata(provider="ollama", model_id=response.model),
                finish_reason=finish_reason,
            )
        except AgentRigError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise AgentRigError(Failure(
                kind=FailureKind.TRANSIENT_PROVIDER,
                code="ollama.text_generation_failed",
                message="Ollama text generation failed",
                metadata={"exception_type": type(error).__qualname__},
            )) from None
        finally:
            if client is not None:
                try:
                    await asyncio.wait_for(client.close(), timeout=_CLIENT_CLOSE_GRACE_SECONDS)
                except Exception:
                    raise AgentRigError(Failure(
                        kind=FailureKind.TRANSIENT_PROVIDER,
                        code="ollama.client_close_failed",
                        message="Ollama client cleanup did not complete",
                    )) from None
