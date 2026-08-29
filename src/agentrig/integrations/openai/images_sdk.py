"""Optional official OpenAI SDK bridge for the image client seam."""

from __future__ import annotations

import base64
from collections.abc import Callable
from io import BytesIO
from typing import Any, Protocol, cast

from agentrig.capabilities import ImageInputRole, ImageUsage
from agentrig.core.errors import AgentRigError, Failure, FailureKind
from agentrig.integrations.openai.images import (
    OpenAIImageClient,
    OpenAIImageOperation,
    OpenAIImageRequest,
    OpenAIImageResult,
)
from agentrig.integrations.openai.responses import (
    OpenAIResponsesAuthenticationSource,
)


class _RawImages(Protocol):
    async def generate(self, **kwargs: Any) -> Any: ...
    async def edit(self, **kwargs: Any) -> Any: ...


class _RawAsyncClient(Protocol):
    images: _RawImages
    async def close(self) -> None: ...


RawImageClientBuilder = Callable[[str], _RawAsyncClient]


class _NamedBytesIO(BytesIO):
    name: str


class OpenAIImageSdkClientFactory:
    """Create short-lived image clients from a late-bound credential source."""

    def __init__(
        self,
        *,
        authentication_source: OpenAIResponsesAuthenticationSource,
        raw_client_builder: RawImageClientBuilder | None = None,
    ) -> None:
        if not isinstance(
            authentication_source, OpenAIResponsesAuthenticationSource
        ):
            raise TypeError("OpenAI image authentication source is invalid")
        self._source = authentication_source
        self._builder = raw_client_builder or _default_raw_client_builder

    def create(self) -> OpenAIImageClient:
        api_key = _resolve_api_key(self._source)
        try:
            return _SdkImageClient(self._builder(api_key))
        except AgentRigError:
            raise
        except Exception:
            raise AgentRigError(
                Failure(
                    kind=FailureKind.TRANSIENT_PROVIDER,
                    message="OpenAI image client could not be created",
                    code="openai.image.client_creation_failed",
                )
            ) from None


def _resolve_api_key(source: OpenAIResponsesAuthenticationSource) -> str:
    try:
        value = source.resolve_api_key()
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or any(item in value for item in ("\x00", "\r", "\n"))
        ):
            raise ValueError("API key is invalid")
        return value
    except Exception:
        raise AgentRigError(
            Failure(
                kind=FailureKind.PERMANENT_PROVIDER,
                message="OpenAI image authentication could not be resolved",
                code="openai.image.authentication_resolution_failed",
            )
        ) from None


def _default_raw_client_builder(api_key: str) -> _RawAsyncClient:
    from openai import AsyncOpenAI

    return cast(_RawAsyncClient, AsyncOpenAI(api_key=api_key))


class _SdkImageClient:
    def __init__(self, raw: _RawAsyncClient) -> None:
        self._raw = raw

    async def create(self, request: OpenAIImageRequest) -> OpenAIImageResult:
        try:
            common: dict[str, Any] = {
                "model": request.model,
                "prompt": request.prompt,
                "n": 1,
                "size": f"{request.width}x{request.height}",
                "output_format": _output_format(request.output_media_type),
            }
            if request.idempotency_key is not None:
                common["extra_headers"] = {
                    "Idempotency-Key": request.idempotency_key
                }
            if request.operation is OpenAIImageOperation.GENERATE:
                response = await self._raw.images.generate(**common)
            else:
                images = [
                    _upload(source.content, source.media_type, index)
                    for index, source in enumerate(request.sources)
                    if source.role is not ImageInputRole.EDIT_MASK
                ]
                mask = next(
                    (
                        _upload(source.content, source.media_type, "mask")
                        for source in request.sources
                        if source.role is ImageInputRole.EDIT_MASK
                    ),
                    None,
                )
                response = await self._raw.images.edit(
                    **common,
                    image=images[0] if len(images) == 1 else images,
                    **({"mask": mask} if mask is not None else {}),
                )
            return _normalize_response(response, request)
        except AgentRigError:
            raise
        except Exception as error:
            raise AgentRigError(_request_failure(error)) from None

    async def close(self) -> None:
        try:
            await self._raw.close()
        except Exception:
            raise AgentRigError(
                Failure(
                    kind=FailureKind.TRANSIENT_PROVIDER,
                    message="OpenAI image client could not be closed",
                    code="openai.image.client_close_failed",
                )
            ) from None


def _upload(content: bytes, media_type: str, suffix: object) -> _NamedBytesIO:
    stream = _NamedBytesIO(content)
    subtype = media_type.partition(";")[0].partition("/")[2]
    stream.name = f"image-{suffix}.{subtype}"
    return stream


def _output_format(media_type: str) -> str:
    subtype = media_type.partition(";")[0].partition("/")[2]
    if subtype not in {"png", "jpeg", "webp"}:
        raise AgentRigError(
            Failure(
                kind=FailureKind.INVALID_INPUT,
                message="OpenAI image output format is unsupported",
                code="openai.image.output_format_unsupported",
            )
        )
    return subtype


def _normalize_response(
    response: Any, request: OpenAIImageRequest
) -> OpenAIImageResult:
    try:
        data = response.data
        if not isinstance(data, list) or len(data) != 1:
            raise ValueError("response must contain exactly one image")
        encoded = data[0].b64_json
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("response image is missing")
        content = base64.b64decode(encoded, validate=True)
        if not content:
            raise ValueError("response image is empty")
        return OpenAIImageResult(
            content=content,
            media_type=request.output_media_type,
            model=request.model,
            usage=_usage(response),
        )
    except Exception:
        raise AgentRigError(
            Failure(
                kind=FailureKind.PERMANENT_PROVIDER,
                message="OpenAI image service returned an invalid response",
                code="openai.image.invalid_response",
            )
    ) from None


def _usage(response: Any) -> ImageUsage:
    value = getattr(response, "usage", None)
    if value is None:
        return ImageUsage()
    return ImageUsage(
        input_tokens=getattr(value, "input_tokens"),
        output_tokens=getattr(value, "output_tokens"),
    )


def _request_failure(error: Exception) -> Failure:
    try:
        status_code = getattr(error, "status_code", None)
    except Exception:
        status_code = None
    if isinstance(status_code, bool) or not isinstance(status_code, int):
        kind = FailureKind.TRANSIENT_PROVIDER
        metadata: dict[str, str] = {}
    else:
        kind = (
            FailureKind.TRANSIENT_PROVIDER
            if status_code == 429 or status_code >= 500
            else FailureKind.PERMANENT_PROVIDER
        )
        metadata = {"status_code": str(status_code)}
    return Failure(
        kind=kind,
        message="OpenAI image request failed",
        code="openai.image.request_failed",
        metadata=metadata,
    )
