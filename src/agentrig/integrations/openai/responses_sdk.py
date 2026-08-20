"""Optional official OpenAI SDK bridge for the Responses adapter."""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast

from agentrig.core._json import JsonValue, thaw_json_value
from agentrig.core.errors import AgentRigError, Failure, FailureKind
from agentrig.integrations.openai.responses import (
    OpenAIResponsesAuthenticationSource,
    OpenAIResponsesClient,
    OpenAIResponsesRequest,
    OpenAIResponsesResult,
    OpenAIResponsesStatus,
)


class _RawResponses(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


class _RawAsyncClient(Protocol):
    responses: _RawResponses

    async def close(self) -> None: ...


RawClientBuilder = Callable[[str], _RawAsyncClient]


class OpenAIResponsesSdkClientFactory:
    """Create short-lived official SDK clients from late-bound credentials."""

    def __init__(
        self,
        *,
        authentication_source: OpenAIResponsesAuthenticationSource,
        raw_client_builder: RawClientBuilder | None = None,
    ) -> None:
        if not isinstance(authentication_source, OpenAIResponsesAuthenticationSource):
            raise TypeError("OpenAI Responses authentication source is invalid")
        self._authentication_source = authentication_source
        self._builder = raw_client_builder or _default_raw_client_builder

    def create(self) -> OpenAIResponsesClient:
        api_key = _resolve_api_key(self._authentication_source)
        try:
            return _SdkClient(self._builder(api_key))
        except AgentRigError:
            raise
        except Exception:
            raise AgentRigError(
                Failure(
                    kind=FailureKind.TRANSIENT_PROVIDER,
                    message="OpenAI Responses client could not be created",
                    code="openai.responses.client_creation_failed",
                )
            ) from None


def _resolve_api_key(source: OpenAIResponsesAuthenticationSource) -> str:
    try:
        value = source.resolve_api_key()
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or any(character in value for character in ("\x00", "\r", "\n"))
        ):
            raise ValueError("API key is invalid")
        return value
    except Exception:
        raise AgentRigError(
            Failure(
                kind=FailureKind.PERMANENT_PROVIDER,
                message="OpenAI Responses authentication could not be resolved",
                code="openai.responses.authentication_resolution_failed",
            )
        ) from None


def _default_raw_client_builder(api_key: str) -> _RawAsyncClient:
    from openai import AsyncOpenAI

    return cast(_RawAsyncClient, AsyncOpenAI(api_key=api_key))


class _SdkClient:
    def __init__(self, raw_client: _RawAsyncClient) -> None:
        self._raw_client = raw_client

    async def create(self, request: OpenAIResponsesRequest) -> OpenAIResponsesResult:
        try:
            response = await self._raw_client.responses.create(
                model=request.model,
                input=_input(request),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": request.schema_name,
                        "schema": _thaw_object(request.output_schema),
                        "strict": True,
                    }
                },
                max_output_tokens=request.max_output_tokens,
                store=False,
                stream=False,
                tools=[],
                truncation="disabled",
            )
            return _normalize_response(response)
        except AgentRigError:
            raise
        except Exception as error:
            raise AgentRigError(_request_failure(error)) from None

    async def close(self) -> None:
        try:
            await self._raw_client.close()
        except Exception:
            raise AgentRigError(
                Failure(
                    kind=FailureKind.TRANSIENT_PROVIDER,
                    message="OpenAI Responses client could not be closed",
                    code="openai.responses.client_close_failed",
                )
            ) from None


def _input(request: OpenAIResponsesRequest) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in request.messages:
        content: list[dict[str, Any]] = []
        if message.text is not None:
            content.append({"type": "input_text", "text": message.text})
        content.extend(
            {
                "type": "input_image",
                "image_url": (
                    f"data:{image.media_type};base64,"
                    + base64.b64encode(image.content).decode("ascii")
                ),
                "detail": "auto",
            }
            for image in message.images
        )
        result.append({"role": message.role.value, "content": content})
    return result


def _normalize_response(response: Any) -> OpenAIResponsesResult:
    try:
        status_value = response.status
        if status_value == "completed":
            status = OpenAIResponsesStatus.COMPLETED
        elif status_value == "incomplete":
            status = OpenAIResponsesStatus.INCOMPLETE
        else:
            raise ValueError("response status is not successful")
        usage = response.usage
        if usage is None:
            raise ValueError("response usage is missing")
        return OpenAIResponsesResult(
            output_text=response.output_text,
            model=response.model,
            status=status,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
    except Exception:
        raise AgentRigError(
            Failure(
                kind=FailureKind.PERMANENT_PROVIDER,
                message="OpenAI Responses returned an invalid response",
                code="openai.responses.invalid_response",
            )
        ) from None


def _thaw_object(value: Mapping[str, JsonValue]) -> dict[str, Any]:
    return {key: thaw_json_value(item) for key, item in value.items()}


def _request_failure(error: Exception) -> Failure:
    try:
        status_code = getattr(error, "status_code", None)
    except Exception:
        status_code = None
    if isinstance(status_code, bool) or not isinstance(status_code, int):
        return Failure(
            kind=FailureKind.TRANSIENT_PROVIDER,
            message="OpenAI Responses request failed",
            code="openai.responses.request_failed",
        )
    kind = (
        FailureKind.TRANSIENT_PROVIDER
        if status_code == 429 or status_code >= 500
        else FailureKind.PERMANENT_PROVIDER
    )
    return Failure(
        kind=kind,
        message="OpenAI Responses request failed",
        code="openai.responses.request_failed",
        metadata={"status_code": str(status_code)},
    )
