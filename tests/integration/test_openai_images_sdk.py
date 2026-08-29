from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
import unittest

from agentrig.capabilities import ImageInputRole
from agentrig.integrations.openai import (
    OpenAIImageOperation,
    OpenAIImageRequest,
    OpenAIImageSource,
)
from agentrig.integrations.openai.images_sdk import (
    OpenAIImageSdkClientFactory,
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
    def test_injected_sdk_bridge_separates_generate_and_edit(self) -> None:
        raw = RawClient()
        client = OpenAIImageSdkClientFactory(
            authentication_source=Authentication(),
            raw_client_builder=lambda api_key: raw,
        ).create()
        generated = asyncio.run(client.create(_generate_request()))
        edited = asyncio.run(client.create(_edit_request()))
        asyncio.run(client.close())

        self.assertEqual(generated.content, b"sdk-image")
        self.assertEqual(edited.content, b"sdk-image")
        self.assertEqual(len(raw.images.generate_calls), 1)
        self.assertEqual(len(raw.images.edit_calls), 1)
        self.assertIn("image", raw.images.edit_calls[0])
        self.assertIn("mask", raw.images.edit_calls[0])
        self.assertIsNone(edited.usage.cost)
        self.assertEqual(edited.usage.total_tokens, 30)
        self.assertNotIn("response_format", raw.images.generate_calls[0])
        self.assertEqual(raw.closed, 1)


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


if __name__ == "__main__":
    unittest.main()
