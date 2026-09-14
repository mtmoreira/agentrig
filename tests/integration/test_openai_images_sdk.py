from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
import unittest
from unittest.mock import Mock, patch

from agentrig.capabilities import ImageInputFidelity, ImageInputRole, ImageQuality
from agentrig.integrations.openai import (
    OpenAIImageOperation,
    OpenAIImageRequest,
    OpenAIImageSource,
)
from agentrig.integrations.openai.images_sdk import (
    OpenAIImageSdkClientFactory,
    _default_raw_client_builder,
)


@dataclass
class Authentication:
    value: str = "test-only-key"

    def resolve_api_key(self) -> str:
        return self.value


@dataclass
class RawImageData:
    b64_json: str


@dataclass
class RawResponse:
    data: list[RawImageData]
    usage: RawUsage | None = None


@dataclass
class RawUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class RawImages:
    generate_calls: list[dict[str, object]] = field(default_factory=list)
    edit_calls: list[dict[str, object]] = field(default_factory=list)

    async def generate(self, **kwargs: object) -> RawResponse:
        self.generate_calls.append(kwargs)
        return _raw_response()

    async def edit(self, **kwargs: object) -> RawResponse:
        self.edit_calls.append(kwargs)
        return _raw_response()


@dataclass
class RawClient:
    images: RawImages = field(default_factory=RawImages)
    closed: int = 0

    async def close(self) -> None:
        self.closed += 1


def _raw_response() -> RawResponse:
    return RawResponse(
        data=[RawImageData(base64.b64encode(b"sdk-image").decode("ascii"))],
        usage=RawUsage(input_tokens=18, output_tokens=12),
    )


class OpenAIImageSdkTest(unittest.TestCase):
    def test_default_sdk_client_disables_retries_and_bounds_request_time(self) -> None:
        sdk = Mock()
        with patch.dict("sys.modules", {"openai": sdk}):
            client = _default_raw_client_builder("test-only-key")

        sdk.AsyncOpenAI.assert_called_once_with(
            api_key="test-only-key", max_retries=0, timeout=300.0
        )
        self.assertIs(client, sdk.AsyncOpenAI.return_value)

    def test_injected_sdk_bridge_separates_generate_and_edit(self) -> None:
        raw = RawClient()
        client = OpenAIImageSdkClientFactory(
            authentication_source=Authentication(),
            raw_client_builder=lambda api_key: raw,
        ).create()
        generated = asyncio.run(client.create(_generate_request()))
        edited = asyncio.run(client.create(_edit_request()))
        referenced = asyncio.run(client.create(_reference_request()))
        asyncio.run(client.close())

        self.assertEqual(generated.content, b"sdk-image")
        self.assertEqual(edited.content, b"sdk-image")
        self.assertEqual(len(raw.images.generate_calls), 1)
        self.assertEqual(len(raw.images.edit_calls), 2)
        self.assertIn("image", raw.images.edit_calls[0])
        self.assertIn("mask", raw.images.edit_calls[0])
        self.assertIsNone(edited.usage.cost)
        self.assertEqual(edited.usage.total_tokens, 30)
        self.assertNotIn("response_format", raw.images.generate_calls[0])
        self.assertEqual(raw.closed, 1)
        call = raw.images.edit_calls[1]
        self.assertEqual(call["model"], "gpt-image-1.5")
        self.assertEqual(call["quality"], "high")
        self.assertEqual(call["input_fidelity"], "high")
        self.assertEqual(call["n"], 1)
        self.assertEqual(call["size"], "1024x1536")
        self.assertEqual(call["output_format"], "png")
        self.assertEqual(
            [stream.getvalue() for stream in call["image"]],
            [b"tomas-bytes", b"nia-bytes"],
        )
        self.assertNotIn("mask", call)
        self.assertEqual(referenced.content, b"sdk-image")


def _generate_request() -> OpenAIImageRequest:
    return OpenAIImageRequest(
        operation=OpenAIImageOperation.GENERATE,
        model="gpt-image-test",
        prompt="Synthetic sky.",
        width=1024,
        height=1536,
        output_media_type="image/png",
    )


def _edit_request() -> OpenAIImageRequest:
    return OpenAIImageRequest(
        operation=OpenAIImageOperation.EDIT,
        model="gpt-image-test",
        prompt="Edit synthetic sky only.",
        width=1024,
        height=1536,
        output_media_type="image/png",
        sources=(
            OpenAIImageSource(
                role=ImageInputRole.EDIT_BASE,
                media_type="image/png",
                content=b"base",
            ),
            OpenAIImageSource(
                role=ImageInputRole.EDIT_MASK,
                media_type="image/png",
                content=b"mask",
            ),
        ),
    )


def _reference_request() -> OpenAIImageRequest:
    return OpenAIImageRequest(
        operation=OpenAIImageOperation.REFERENCE_GENERATE,
        model="gpt-image-1.5",
        prompt=(
            "Ordered source-image roles:\n1. identity_reference\n"
            "2. identity_reference\n\nSynthetic scene."
        ),
        width=1024,
        height=1536,
        output_media_type="image/png",
        quality=ImageQuality.HIGH,
        input_fidelity=ImageInputFidelity.HIGH,
        sources=(
            OpenAIImageSource(
                role=ImageInputRole.IDENTITY_REFERENCE,
                media_type="image/png",
                content=b"tomas-bytes",
            ),
            OpenAIImageSource(
                role=ImageInputRole.IDENTITY_REFERENCE,
                media_type="image/png",
                content=b"nia-bytes",
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
