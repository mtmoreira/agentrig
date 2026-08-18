"""Optional official Ollama SDK bridge for AgentRig's injected contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast
from urllib.parse import urlsplit

from agentrig.core._json import JsonValue, thaw_json_value
from agentrig.core.errors import AgentRigError, Failure, FailureKind
from agentrig.integrations.ollama.ollama import (
    OllamaAuthenticationSource,
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaClient,
    OllamaFinishReason,
)


class _RawAsyncClient(Protocol):
    async def chat(self, **kwargs: Any) -> Any: ...

    async def close(self) -> None: ...


RawClientBuilder = Callable[[str, Mapping[str, str]], _RawAsyncClient]


class OllamaSdkClientFactory:
    """Create short-lived official SDK clients from application-owned config."""

    def __init__(
        self,
        *,
        host: str,
        authentication_source: OllamaAuthenticationSource | None = None,
        raw_client_builder: RawClientBuilder | None = None,
    ) -> None:
        self._host = _validate_host(host)
        if (
            authentication_source is not None
            and not isinstance(authentication_source, OllamaAuthenticationSource)
        ):
            raise TypeError(
                "Ollama authentication source must satisfy "
                "OllamaAuthenticationSource"
            )
        self._authentication_source = authentication_source
        self._builder = raw_client_builder or _default_raw_client_builder

    def create(self) -> OllamaClient:
        """Resolve private headers only after runtime preflight succeeds."""
        headers = (
            _resolve_authentication_headers(self._authentication_source)
            if self._authentication_source is not None
            else {}
        )
        try:
            raw_client = self._builder(self._host, headers)
        except AgentRigError:
            raise
        except Exception:
            raise AgentRigError(
                Failure(
                    kind=FailureKind.TRANSIENT_PROVIDER,
                    message="Ollama client could not be created",
                    code="ollama.client_creation_failed",
                )
            ) from None
        return _SdkClient(raw_client)


def _validate_host(host: str) -> str:
    if not isinstance(host, str) or not host or host != host.strip():
        raise ValueError("Ollama host must be nonempty and trimmed")
    if any(character in host for character in ("\x00", "\r", "\n")):
        raise ValueError("Ollama host contains invalid characters")
    try:
        parsed = urlsplit(host)
        hostname = parsed.hostname
        username = parsed.username
        password = parsed.password
        parsed.port
    except ValueError:
        raise ValueError("Ollama host is invalid") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or hostname is None
    ):
        raise ValueError("Ollama host must be an absolute HTTP(S) URL")
    if username is not None or password is not None:
        raise ValueError("Ollama host must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("Ollama host must not contain a query or fragment")
    return host


def _resolve_authentication_headers(
    source: OllamaAuthenticationSource,
) -> dict[str, str]:
    try:
        resolved = source.resolve_headers()
        if not isinstance(resolved, Mapping):
            raise TypeError("authentication headers must be a mapping")
        copied = dict(resolved)
        if not copied:
            raise ValueError("authentication headers must not be empty")
        for name, value in copied.items():
            if (
                not isinstance(name, str)
                or not name
                or name != name.strip()
                or not all(_is_header_name_character(character) for character in name)
            ):
                raise ValueError("authentication header name is invalid")
            if (
                not isinstance(value, str)
                or not value
                or "\x00" in value
                or "\r" in value
                or "\n" in value
            ):
                raise ValueError("authentication header value is invalid")
        return copied
    except Exception:
        raise AgentRigError(
            Failure(
                kind=FailureKind.PERMANENT_PROVIDER,
                message="Ollama authentication could not be resolved",
                code="ollama.authentication_resolution_failed",
            )
        ) from None


def _is_header_name_character(character: str) -> bool:
    return character.isascii() and (
        character.isalnum() or character in "!#$%&'*+-.^_`|~"
    )


def _default_raw_client_builder(
    host: str,
    headers: Mapping[str, str],
) -> _RawAsyncClient:
    from ollama import AsyncClient

    return cast(_RawAsyncClient, AsyncClient(host=host, headers=dict(headers)))


class _SdkClient:
    def __init__(self, raw_client: _RawAsyncClient) -> None:
        self._raw_client = raw_client

    async def chat(self, request: OllamaChatRequest) -> OllamaChatResponse:
        try:
            raw_response = await self._raw_client.chat(
                model=request.model,
                messages=[
                    {"role": message.role, "content": message.content}
                    for message in request.messages
                ],
                stream=False,
                think=request.think,
                format=_thaw_object(request.output_schema),
                options=_thaw_object(request.options),
                keep_alive=request.keep_alive,
            )
            return _normalize_response(raw_response)
        except AgentRigError:
            raise
        except Exception as error:
            raise AgentRigError(_request_failure(error)) from None

    async def close(self) -> None:
        try:
            await self._raw_client.close()
        except AgentRigError:
            raise
        except Exception:
            raise AgentRigError(
                Failure(
                    kind=FailureKind.TRANSIENT_PROVIDER,
                    message="Ollama client could not be closed",
                    code="ollama.client_close_failed",
                )
            ) from None


def _thaw_object(value: Mapping[str, JsonValue]) -> dict[str, Any]:
    return {
        key: thaw_json_value(item)
        for key, item in value.items()
    }


def _normalize_response(raw_response: Any) -> OllamaChatResponse:
    try:
        message = raw_response.message
        content = message.content
        model = raw_response.model
        done_reason = raw_response.done_reason
        input_tokens = raw_response.prompt_eval_count
        output_tokens = raw_response.eval_count
        if input_tokens is None:
            input_tokens = 0
        if output_tokens is None:
            output_tokens = 0
        return OllamaChatResponse(
            content=content,
            model=model,
            finish_reason=_finish_reason(done_reason),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except Exception:
        raise AgentRigError(
            Failure(
                kind=FailureKind.PERMANENT_PROVIDER,
                message="Ollama returned an invalid response",
                code="ollama.invalid_response",
            )
        ) from None


def _finish_reason(value: object) -> OllamaFinishReason:
    if value == "stop":
        return OllamaFinishReason.STOP
    if value == "length":
        return OllamaFinishReason.LENGTH
    return OllamaFinishReason.OTHER


def _request_failure(error: Exception) -> Failure:
    try:
        status_code = getattr(error, "status_code", None)
    except Exception:
        status_code = None
    if isinstance(status_code, bool) or not isinstance(status_code, int):
        return Failure(
            kind=FailureKind.TRANSIENT_PROVIDER,
            message="Ollama request failed",
            code="ollama.request_failed",
        )
    kind = (
        FailureKind.TRANSIENT_PROVIDER
        if status_code == 429 or status_code >= 500
        else FailureKind.PERMANENT_PROVIDER
    )
    return Failure(
        kind=kind,
        message="Ollama request failed",
        code="ollama.request_failed",
        metadata={"status_code": str(status_code)},
    )
