from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
import json
import tempfile
import unittest

from agentrig.capabilities import (
    ImageGenerationRequest, ImageGenerationSettings, ImageInput,
    ImageInputFidelity, ImageInputRole, ImageOperation, ImageQuality,
    ImageSize, ImageSpecification,
)
from agentrig.core import (
    ArtifactId, ArtifactRef, CancellationSource, ResolvedArtifact,
    RunContext, RunId,
)
from agentrig.integrations.comfyui import (
    ComfyUIImageClient, ComfyUIImageGenerator, ComfyUIImageOperation,
    ComfyUIImageRequest, ComfyUIImageResult, ComfyUIHttpClientFactory,
    ComfyUIWorkflowCatalog,
)


@dataclass(frozen=True)
class Clock:
    def now(self) -> datetime:
        return datetime(2026, 10, 10, tzinfo=UTC)
    def monotonic(self) -> float:
        return 1.0


@dataclass
class Ids:
    def generate(self) -> RunId:
        return RunId("local-image-run")


def context() -> RunContext:
    return RunContext.create_root(
        clock=Clock(), id_generator=Ids(), cancellation=CancellationSource().token)


def artifact(name: str) -> ArtifactRef:
    return ArtifactRef(artifact_id=ArtifactId(name), kind="image",
        media_type="image/png", producer_run_id=RunId("source"),
        workspace_path=f"inputs/{name}.png")


@dataclass
class Resolver:
    async def resolve(self, value: ArtifactRef) -> ResolvedArtifact:
        return ResolvedArtifact(artifact=value, content=value.artifact_id.value.encode())


@dataclass
class Client:
    requests: list[ComfyUIImageRequest] = field(default_factory=list)
    closed: int = 0
    async def create(self, request: ComfyUIImageRequest) -> ComfyUIImageResult:
        self.requests.append(request)
        return ComfyUIImageResult(content=b"generated", media_type="image/png",
            model=request.model, workflow_id="prompt-1")
    async def close(self) -> None:
        self.closed += 1


@dataclass
class Factory:
    client: ComfyUIImageClient
    def create(self) -> ComfyUIImageClient:
        return self.client


@dataclass
class Publisher:
    async def publish(self, *, request: ImageGenerationRequest, content: bytes,
                      media_type: str, context: RunContext) -> ArtifactRef:
        return ArtifactRef(artifact_id=ArtifactId("result"), kind="image",
            media_type=media_type, producer_run_id=context.run_id,
            workspace_path="outputs/result.png",
            input_artifact_ids=request.source_artifact_ids)


class ComfyUIImageGeneratorTests(unittest.TestCase):
    def test_reference_generation_preserves_roles_settings_and_zero_provider_cost(self) -> None:
        first, second = artifact("first"), artifact("second")
        request = ImageGenerationRequest(
            specification=ImageSpecification(prompt="Draw a fictional couple.",
                size=ImageSize(width=1024, height=1536)),
            inputs=(ImageInput(role=ImageInputRole.IDENTITY_REFERENCE, artifact=second),
                    ImageInput(role=ImageInputRole.STYLE_REFERENCE, artifact=first)),
            settings=ImageGenerationSettings(operation=ImageOperation.REFERENCE_GENERATE,
                model="flux-2-klein-4b", quality=ImageQuality.MEDIUM,
                input_fidelity=ImageInputFidelity.HIGH),
            idempotency_key="book-cover-v1")
        client = Client()
        generator = ComfyUIImageGenerator(client_factory=Factory(client),
            artifact_resolver=Resolver(), artifact_publisher=Publisher(),
            model="unused")

        result = asyncio.run(generator.generate(request, context()))

        translated = client.requests[0]
        self.assertIs(translated.operation, ComfyUIImageOperation.REFERENCE_GENERATE)
        self.assertEqual(translated.model, "flux-2-klein-4b")
        self.assertEqual([item.content for item in translated.sources], [b"second", b"first"])
        self.assertIn("1. identity_reference", translated.prompt)
        self.assertIn("2. style_reference", translated.prompt)
        self.assertEqual((result.usage.cost, result.usage.currency), (0.0, "USD"))
        self.assertEqual(result.image.input_artifact_ids,
                         (second.artifact_id, first.artifact_id))
        self.assertEqual(client.closed, 1)

    def test_catalog_is_explicit_and_transport_is_loopback_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "generate.json").write_text(json.dumps({
                "1": {"class_type": "CLIPTextEncode",
                      "inputs": {"text": "$agentrig.prompt",
                          "model": "$agentrig.model"}},
                "2": {"class_type": "EmptyLatentImage", "inputs": {
                    "width": "$agentrig.width", "height": "$agentrig.height"}},
                "3": {"class_type": "Sampler", "inputs": {
                    "seed": "$agentrig.seed", "steps": "$agentrig.steps",
                    "strength": "$agentrig.reference_strength"}},
                "9": {"class_type": "SaveImage", "inputs": {
                    "filename_prefix": "local"}},
            }))
            (root / "manifest.json").write_text(json.dumps({
                "schema_id": "agentrig.comfyui-workflows.v1",
                "workflows": [{"operation": "generate", "source_count": 0,
                    "file": "generate.json", "output_node_id": "9"}],
            }))
            catalog = ComfyUIWorkflowCatalog.from_directory(root)
            request = ComfyUIImageRequest(operation=ComfyUIImageOperation.GENERATE,
                model="flux-2-klein-4b", prompt="A safe synthetic scene",
                width=1024, height=1024, output_media_type="image/png")
            self.assertEqual(catalog.select(request).output_node_id, "9")
            ComfyUIHttpClientFactory(catalog=catalog, host="http://localhost:8188")
            with self.assertRaisesRegex(ValueError, "loopback"):
                ComfyUIHttpClientFactory(catalog=catalog,
                    host="https://comfy.example.invalid")
            with self.assertRaisesRegex(ValueError, "loopback"):
                ComfyUIHttpClientFactory(catalog=catalog,
                    host="http://localhost:8188/unexpected-prefix")


if __name__ == "__main__":
    unittest.main()
