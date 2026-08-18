"""Provider-neutral AgentRuntime adapter for injected Ollama clients."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping

from agentrig.agents import AgentExecutionRequest, AgentExecutionResult
from agentrig.core._json import JsonValue, thaw_json_value
from agentrig.core.context import RunContext
from agentrig.core.deadline import DeadlineExceeded
from agentrig.core.errors import (
    AgentRigError,
    Failure,
    FailureKind,
    normalize_exception,
)
from agentrig.core.events import Event, EventKind
from agentrig.integrations.ollama.ollama import (
    OLLAMA_AGENT_RUNTIME_CAPABILITY,
    OllamaChatMessage,
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaClient,
    OllamaClientFactory,
    OllamaFinishReason,
    OllamaRuntimeOptions,
)

_PROVIDER = "ollama"


class OllamaAgentRuntime:
    """Execute one structured portable agent turn through Ollama."""

    def __init__(
        self,
        *,
        client_factory: OllamaClientFactory,
        model: str,
        output_schemas: Mapping[str, Mapping[str, JsonValue]],
        options: OllamaRuntimeOptions | None = None,
    ) -> None:
        if not isinstance(client_factory, OllamaClientFactory):
            raise TypeError("Ollama runtime factory must satisfy OllamaClientFactory")
        if not isinstance(model, str) or not model or model != model.strip():
            raise ValueError("Ollama runtime model must be nonempty and trimmed")
        schemas: dict[str, Mapping[str, JsonValue]] = {}
        for schema_id, schema in output_schemas.items():
            if not isinstance(schema_id, str) or not schema_id or schema_id != schema_id.strip():
                raise ValueError("Ollama schema IDs must be nonempty and trimmed")
            request = OllamaChatRequest(
                model=model,
                messages=(OllamaChatMessage(role="user", content="schema check"),),
                output_schema=schema,
            )
            schemas[schema_id] = request.output_schema
        if not schemas:
            raise ValueError("Ollama runtime requires at least one output schema")
        if options is not None and not isinstance(options, OllamaRuntimeOptions):
            raise TypeError("Ollama runtime options must be OllamaRuntimeOptions")
        self._client_factory = client_factory
        self._model = model
        self._output_schemas = schemas
        self._options = options if options is not None else OllamaRuntimeOptions()

    async def execute(
        self,
        request: AgentExecutionRequest,
        context: RunContext,
    ) -> AgentExecutionResult:
        """Preflight, execute one chat, and decode structured JSON."""
        if not isinstance(request, AgentExecutionRequest):
            raise TypeError("Ollama runtime request must be an AgentExecutionRequest")
        if not isinstance(context, RunContext):
            raise TypeError("Ollama runtime context must be a RunContext")
        try:
            _check_constraints(context)
        except BaseException as error:
            return AgentExecutionResult.from_failure(normalize_exception(error))
        failure = self._validate_request(request)
        if failure is not None:
            return AgentExecutionResult.from_failure(failure)

        _emit(context, EventKind.PROVIDER_CALL_STARTED, _base_attributes(request))
        client: OllamaClient | None = None
        try:
            client = self._client_factory.create()
            response = await _chat_with_constraints(
                client,
                self._chat_request(request),
                context,
            )
            result = _decode_response(response, expected_model=self._model)
            _emit_usage(context, request, response)
        except AgentRigError as error:
            result = AgentExecutionResult.from_failure(error.failure)
        except asyncio.CancelledError as error:
            result = AgentExecutionResult.from_failure(normalize_exception(error))
        except Exception as error:
            result = AgentExecutionResult.from_failure(
                Failure(
                    kind=FailureKind.TRANSIENT_PROVIDER,
                    message="Ollama runtime transport failed",
                    code="ollama.transport_failed",
                    metadata={"exception_type": type(error).__qualname__},
                )
            )
        finally:
            if client is not None:
                try:
                    await client.close()
                except Exception:
                    pass
        _emit_completed(context, request, result)
        return result

    def _validate_request(self, request: AgentExecutionRequest) -> Failure | None:
        contract = request.contract
        if OLLAMA_AGENT_RUNTIME_CAPABILITY.capability_id not in contract.allowed_capabilities:
            return _invalid("Ollama runtime is not authorized", "ollama.capability_not_allowed")
        if contract.output_schema not in self._output_schemas:
            return _invalid("Ollama output schema is not configured", "ollama.output_schema_not_configured")
        if contract.allowed_tools:
            return _invalid("Ollama runtime does not support tools", "ollama.tools_not_supported")
        if request.provider_options:
            return _invalid("Ollama provider options are fixed by the runtime binding", "ollama.provider_options_invalid")
        if contract.permissions.get("workspace") != "denied":
            return _invalid("Ollama runtime requires denied workspace authority", "ollama.workspace_permission_mismatch")
        if contract.permissions.get("network") != "allowed":
            return _invalid("Ollama runtime requires network authority", "ollama.network_permission_mismatch")
        return None

    def _chat_request(self, request: AgentExecutionRequest) -> OllamaChatRequest:
        return OllamaChatRequest(
            model=self._model,
            messages=(
                OllamaChatMessage(role="system", content=request.instructions),
                OllamaChatMessage(role="user", content=_encode_input(request.input)),
            ),
            output_schema=self._output_schemas[request.contract.output_schema],
            options=self._options.to_provider_options(),
            keep_alive=self._options.keep_alive,
        )


async def _chat_with_constraints(
    client: OllamaClient,
    request: OllamaChatRequest,
    context: RunContext,
) -> OllamaChatResponse:
    chat_task = asyncio.create_task(client.chat(request))
    constraint_task = asyncio.create_task(_wait_for_constraint(context))
    try:
        done, _ = await asyncio.wait(
            {chat_task, constraint_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if constraint_task in done:
            chat_task.cancel()
            await asyncio.gather(chat_task, return_exceptions=True)
            raise AgentRigError(constraint_task.result())
        response = chat_task.result()
        if not isinstance(response, OllamaChatResponse):
            raise TypeError("Ollama client returned an invalid response")
        return response
    finally:
        constraint_task.cancel()
        if not chat_task.done():
            chat_task.cancel()
        await asyncio.gather(chat_task, constraint_task, return_exceptions=True)


async def _wait_for_constraint(context: RunContext) -> Failure:
    cancellation_task = asyncio.create_task(context.cancellation.wait_cancelled())
    deadline_task: asyncio.Task[None] | None = None
    if context.deadline is not None:
        deadline_task = asyncio.create_task(
            asyncio.sleep(context.deadline.remaining_seconds(context.clock))
        )
    try:
        if deadline_task is None:
            cancellation = await cancellation_task
            return Failure(kind=FailureKind.CANCELLED, message=cancellation.reason)
        done, _ = await asyncio.wait(
            {cancellation_task, deadline_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if cancellation_task in done:
            cancellation = cancellation_task.result()
            return Failure(kind=FailureKind.CANCELLED, message=cancellation.reason)
        if context.deadline is None:
            raise AssertionError("deadline task requires a deadline")
        return normalize_exception(DeadlineExceeded(context.deadline))
    finally:
        cancellation_task.cancel()
        if deadline_task is not None:
            deadline_task.cancel()
        await asyncio.gather(
            cancellation_task,
            *(tuple([deadline_task]) if deadline_task is not None else ()),
            return_exceptions=True,
        )


def _decode_response(
    response: OllamaChatResponse,
    *,
    expected_model: str,
) -> AgentExecutionResult:
    metadata = {
        "provider": _PROVIDER,
        "model": expected_model,
        "finish_reason": response.finish_reason.value,
    }
    try:
        if response.model != expected_model:
            raise ValueError("Ollama response model does not match request")
        output = json.loads(response.content)
        return AgentExecutionResult.succeeded(
            output,
            provider_metadata=metadata,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return AgentExecutionResult.from_failure(
            Failure(
                kind=FailureKind.PERMANENT_PROVIDER,
                message="Ollama returned invalid structured output",
                code="ollama.invalid_output",
            ),
            provider_metadata=metadata,
        )


def _encode_input(value: JsonValue) -> str:
    return json.dumps(
        thaw_json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _invalid(message: str, code: str) -> Failure:
    return Failure(kind=FailureKind.INVALID_INPUT, message=message, code=code)


def _check_constraints(context: RunContext) -> None:
    context.cancellation.raise_if_cancelled()
    if context.deadline is not None:
        context.deadline.raise_if_expired(context.clock)


def _base_attributes(request: AgentExecutionRequest) -> dict[str, JsonValue]:
    return {
        "agent_id": request.contract.agent_id,
        "agent_version": request.contract.version,
        "provider": _PROVIDER,
    }


def _emit_usage(
    context: RunContext,
    request: AgentExecutionRequest,
    response: OllamaChatResponse,
) -> None:
    _emit(
        context,
        EventKind.USAGE_REPORTED,
        {
            **_base_attributes(request),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        },
    )


def _emit_completed(
    context: RunContext,
    request: AgentExecutionRequest,
    result: AgentExecutionResult,
) -> None:
    attributes: dict[str, JsonValue] = {
        **_base_attributes(request),
        "status": result.result.status.value,
    }
    failure = result.result.failure
    if failure is not None:
        attributes["failure_kind"] = failure.kind.value
        if failure.code is not None:
            attributes["failure_code"] = failure.code
    _emit(context, EventKind.PROVIDER_CALL_COMPLETED, attributes)


def _emit(
    context: RunContext,
    kind: EventKind,
    attributes: Mapping[str, JsonValue],
) -> None:
    context.event_sink.emit(
        Event.from_context(
            event_id=context.event_id_generator.generate(),
            kind=kind,
            context=context,
            attributes=attributes,
        )
    )
