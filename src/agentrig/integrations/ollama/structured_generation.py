"""Portable schema-decoded multimodal generation through Ollama."""

from __future__ import annotations

import asyncio
from hashlib import sha256
import json
from typing import Generic, TypeVar

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    DataRetention,
    GenerationUsage,
    ModelMetadata,
    StructuredGenerationRequest,
    StructuredGenerationResult,
    TextGenerationFinishReason,
)
from agentrig.core import ArtifactRef, ArtifactResolver, ResolvedArtifact, RunContext
from agentrig.core.errors import AgentRigError, Failure, FailureKind, normalize_exception
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

OutputT = TypeVar("OutputT")
_CLIENT_CLOSE_GRACE_SECONDS = 1.0

OLLAMA_STRUCTURED_GENERATION_CAPABILITY = CapabilityDescriptor(
    capability_id="ollama.structured_generation",
    version=OLLAMA_CLIENT_VERSION,
    kind=CapabilityKind.STRUCTURED_GENERATION,
    features=frozenset({
        CapabilityFeature.MESSAGE_INPUT,
        CapabilityFeature.MULTIMODAL_INPUT,
        CapabilityFeature.CANCELLATION,
        CapabilityFeature.STRUCTURED_OUTPUT,
        CapabilityFeature.USAGE_REPORTING,
    }),
    limits={
        CapabilityLimit.MAX_INPUT_ARTIFACTS: 10,
        CapabilityLimit.MAX_OUTPUT_TOKENS: 8192,
    },
    data_retention=DataRetention.UNKNOWN,
)


class OllamaStructuredGenerator(Generic[OutputT]):
    """Resolve image artifacts, request strict JSON, and apply the caller decoder."""

    descriptor = OLLAMA_STRUCTURED_GENERATION_CAPABILITY

    def __init__(
        self,
        *,
        client_factory: OllamaClientFactory,
        artifact_resolver: ArtifactResolver,
        model: str,
        options: OllamaRuntimeOptions | None = None,
    ) -> None:
        if not isinstance(client_factory, OllamaClientFactory):
            raise TypeError("Ollama structured client factory is invalid")
        if not isinstance(artifact_resolver, ArtifactResolver):
            raise TypeError("Ollama structured artifact resolver is invalid")
        if not isinstance(model, str) or not model or model != model.strip():
            raise ValueError("Ollama structured model must be nonempty and trimmed")
        if options is not None and not isinstance(options, OllamaRuntimeOptions):
            raise TypeError("Ollama structured options are invalid")
        self._client_factory = client_factory
        self._artifact_resolver = artifact_resolver
        self._model = model
        self._options = options if options is not None else OllamaRuntimeOptions()

    async def generate(
        self,
        request: StructuredGenerationRequest[OutputT],
        context: RunContext,
    ) -> StructuredGenerationResult[OutputT]:
        if not isinstance(request, StructuredGenerationRequest):
            raise TypeError("Ollama structured request is invalid")
        if not isinstance(context, RunContext):
            raise TypeError("Ollama structured context is invalid")
        request.require_supported_by(self.descriptor)
        _check_constraints(context)
        options = self._options
        if request.input.max_output_tokens is not None:
            from dataclasses import replace
            options = replace(options, max_output_tokens=request.input.max_output_tokens)
        messages = await self._messages(request, context)
        chat_request = OllamaChatRequest(
            model=self._model,
            messages=messages,
            output_schema=request.output_schema.json_schema,
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
                "messages": [{"role": message.role, "content": message.content,
                    "image_sha256": [sha256(image).hexdigest()
                                     for image in message.images]}
                    for message in chat_request.messages],
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
        except AgentRigError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise AgentRigError(Failure(
                kind=FailureKind.TRANSIENT_PROVIDER,
                code="ollama.structured_generation_failed",
                message="Ollama structured generation failed",
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
        try:
            encoded = json.loads(response.content)
            finish_reason = {
                OllamaFinishReason.STOP: TextGenerationFinishReason.COMPLETED,
                OllamaFinishReason.LENGTH: TextGenerationFinishReason.LENGTH,
            }.get(response.finish_reason, TextGenerationFinishReason.OTHER)
            return StructuredGenerationResult(
                encoded_output=encoded,
                output_schema=request.output_schema,
                usage=GenerationUsage(
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                ),
                model=ModelMetadata(provider="ollama", model_id=response.model),
                finish_reason=finish_reason,
                provider_metadata=(
                    {"total_duration_ns": str(response.total_duration_ns)}
                    if response.total_duration_ns is not None else {}
                ),
            )
        except AgentRigError:
            raise
        except (TypeError, ValueError) as error:
            diagnostic = request.output_schema.resolve_decode_diagnostic(error)
            metadata = {"stage": "schema_decode" if diagnostic is not None else "json_parse"}
            if diagnostic is not None:
                metadata.update(code=diagnostic.code, location=diagnostic.location)
            raise AgentRigError(Failure(
                kind=FailureKind.PERMANENT_PROVIDER,
                code="ollama.structured.invalid_output",
                message="Ollama returned invalid structured output",
                metadata=metadata,
            )) from None

    async def _messages(
        self, request: StructuredGenerationRequest[OutputT], context: RunContext
    ) -> tuple[OllamaChatMessage, ...]:
        source = request.input
        if source.prompt is not None:
            return (OllamaChatMessage(
                role="user",
                content=source.prompt,
                images=await self._images(source.input_artifacts, context),
            ),)
        result: list[OllamaChatMessage] = []
        for message in source.messages:
            result.append(OllamaChatMessage(
                role=message.role.value,
                content=message.text or "",
                images=await self._images(message.artifacts, context),
            ))
        if source.input_artifacts:
            result.append(OllamaChatMessage(
                role="user", content="Inspect the attached images.",
                images=await self._images(source.input_artifacts, context),
            ))
        return tuple(result)

    async def _images(
        self, artifacts: tuple[ArtifactRef, ...], context: RunContext
    ) -> tuple[bytes, ...]:
        images: list[bytes] = []
        for artifact in artifacts:
            _check_constraints(context)
            resolved = await self._artifact_resolver.resolve(artifact)
            if not isinstance(resolved, ResolvedArtifact) or resolved.artifact != artifact:
                raise AgentRigError(Failure(
                    kind=FailureKind.INVALID_INPUT,
                    code="artifact.resolution_mismatch",
                    message="artifact resolver returned mismatched content",
                ))
            if resolved.artifact.media_type.partition(";")[0] not in {
                "image/jpeg", "image/png", "image/webp"
            }:
                raise AgentRigError(Failure(
                    kind=FailureKind.INVALID_INPUT,
                    code="ollama.image_media_type_unsupported",
                    message="Ollama image media type is unsupported",
                ))
            images.append(resolved.content)
        return tuple(images)


def _check_constraints(context: RunContext) -> None:
    try:
        context.cancellation.raise_if_cancelled()
        if context.deadline is not None:
            context.deadline.raise_if_expired(context.clock)
    except AgentRigError:
        raise
    except Exception as error:
        raise AgentRigError(normalize_exception(error)) from None
