from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
import unittest

from agentrig.capabilities import (
    ImageGenerationRequest,
    ImageInput,
    ImageInputRole,
    ImageSize,
    ImageSpecification,
    ImageUsage,
)
from agentrig.core import (
    ArtifactId,
    ArtifactRef,
    CancellationSource,
    ResolvedArtifact,
    RunContext,
    RunId,
)
from agentrig.integrations.openai import (
    OpenAIImageClient,
    OpenAIImageGenerator,
    OpenAIImageOperation,
    OpenAIImageRequest,
    OpenAIImageResult,
)
from agentrig.workflow import (
    ImageExecutionPolicy,
    ImageGenerationExecutor,
    ImageRoute,
)


@dataclass(frozen=True)
class Clock:
    def now(self) -> datetime:
        return datetime(2026, 8, 28, tzinfo=UTC)
    def monotonic(self) -> float:
        return 100.0


@dataclass
class Ids:
    value: int = 0
    def generate(self) -> RunId:
        self.value += 1
        return RunId(f"run-{self.value}")


def context() -> RunContext:
    return RunContext.create_root(
        clock=Clock(), id_generator=Ids(), cancellation=CancellationSource().token
    )


def artifact(name: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(name),
        kind="image",
        media_type="image/png",
        producer_run_id=RunId("source"),
        workspace_path=f"inputs/{name}.png",
    )


@dataclass
class Resolver:
    calls: list[ArtifactRef] = field(default_factory=list)
    async def resolve(self, value: ArtifactRef) -> ResolvedArtifact:
        self.calls.append(value)
        return ResolvedArtifact(
            artifact=value,
            content=value.artifact_id.value.encode(),
        )


@dataclass
class Client:
    requests: list[OpenAIImageRequest] = field(default_factory=list)
    closed: int = 0
    async def create(self, request: OpenAIImageRequest) -> OpenAIImageResult:
        self.requests.append(request)
        return OpenAIImageResult(
            content=b"generated-image",
            media_type="image/png",
            model="gpt-image-test",
            usage=ImageUsage(),
        )
    async def close(self) -> None:
        self.closed += 1


@dataclass
class Factory:
    client: OpenAIImageClient
    calls: int = 0
    def create(self) -> OpenAIImageClient:
        self.calls += 1
        return self.client


@dataclass
class Publisher:
    contents: list[bytes] = field(default_factory=list)
    async def publish(
        self,
        *,
        request: ImageGenerationRequest,
        content: bytes,
        media_type: str,
        context: RunContext,
    ) -> ArtifactRef:
        self.contents.append(content)
        return ArtifactRef(
            artifact_id=ArtifactId("output"),
            kind="image",
            media_type=media_type,
            producer_run_id=context.run_id,
            workspace_path="outputs/image.png",
            input_artifact_ids=request.source_artifact_ids,
        )


@dataclass
class FailingPublisher:
    calls: int = 0

    async def publish(
        self,
        *,
        request: ImageGenerationRequest,
        content: bytes,
        media_type: str,
        context: RunContext,
    ) -> ArtifactRef:
        self.calls += 1
        raise RuntimeError("private storage detail")


class OpenAIImageGeneratorTest(unittest.TestCase):
    def test_translates_explicit_roles_without_leaking_storage(self) -> None:
        base, identity, mask = artifact("base"), artifact("identity"), artifact("mask")
        request = ImageGenerationRequest(
            specification=ImageSpecification(
                prompt="Preserve identity and replace only the sky.",
                size=ImageSize(width=1024, height=1536),
            ),
            inputs=(
                ImageInput(role=ImageInputRole.EDIT_BASE, artifact=base),
                ImageInput(role=ImageInputRole.IDENTITY_REFERENCE, artifact=identity),
                ImageInput(role=ImageInputRole.EDIT_MASK, artifact=mask),
            ),
        )
        resolver, client, publisher = Resolver(), Client(), Publisher()
        generator = OpenAIImageGenerator(
            client_factory=Factory(client),
            artifact_resolver=resolver,
            artifact_publisher=publisher,
            model="gpt-image-test",
        )

        result = asyncio.run(generator.generate(request, context()))

        self.assertEqual(client.requests[0].operation, OpenAIImageOperation.EDIT)
        self.assertEqual(
            tuple(item.role for item in client.requests[0].sources),
            (
                ImageInputRole.EDIT_BASE,
                ImageInputRole.IDENTITY_REFERENCE,
                ImageInputRole.EDIT_MASK,
            ),
        )
        self.assertEqual(result.image.input_artifact_ids, request.source_artifact_ids)
        self.assertIsNone(result.usage.cost)
        self.assertEqual(client.closed, 1)
        self.assertEqual(publisher.contents, [b"generated-image"])

    def test_selected_route_never_constructs_the_other_client(self) -> None:
        base = artifact("base")
        request = ImageGenerationRequest(
            specification=ImageSpecification(
                prompt="Edit only the synthetic sky.",
                size=ImageSize(width=1024, height=1536),
            ),
            inputs=(
                ImageInput(role=ImageInputRole.EDIT_BASE, artifact=base),
            ),
        )
        primary_factory = Factory(Client())
        other_factory = Factory(Client())
        resolver = Resolver()
        publisher = Publisher()
        executor = ImageGenerationExecutor(
            routes=(
                ImageRoute(
                    route_id="primary",
                    generator=OpenAIImageGenerator(
                        client_factory=primary_factory,
                        artifact_resolver=resolver,
                        artifact_publisher=publisher,
                        model="gpt-image-test",
                    ),
                ),
                ImageRoute(
                    route_id="other",
                    generator=OpenAIImageGenerator(
                        client_factory=other_factory,
                        artifact_resolver=resolver,
                        artifact_publisher=publisher,
                        model="gpt-image-test",
                    ),
                ),
            ),
            policy=ImageExecutionPolicy(),
        )

        result = asyncio.run(
            executor.execute(
                route_id="primary", request=request, context=context()
            )
        )

        self.assertTrue(result.outcome.is_success)
        self.assertEqual(primary_factory.calls, 1)
        self.assertEqual(other_factory.calls, 0)

    def test_publication_failure_is_not_retried_as_a_provider_failure(self) -> None:
        base = artifact("base")
        edit = ImageGenerationRequest(
            specification=ImageSpecification(
                prompt="Edit only the synthetic sky.",
                size=ImageSize(width=1024, height=1536),
            ),
            inputs=(
                ImageInput(role=ImageInputRole.EDIT_BASE, artifact=base),
            ),
            idempotency_key="bounded-edit-v1",
        )
        client = Client()
        publisher = FailingPublisher()
        executor = ImageGenerationExecutor(
            routes=(
                ImageRoute(
                    route_id="primary",
                    generator=OpenAIImageGenerator(
                        client_factory=Factory(client),
                        artifact_resolver=Resolver(),
                        artifact_publisher=publisher,
                        model="gpt-image-test",
                    ),
                ),
            ),
            policy=ImageExecutionPolicy(max_attempts=2),
        )

        result = asyncio.run(
            executor.execute(
                route_id="primary", request=edit, context=context()
            )
        )

        self.assertEqual(result.outcome.failure.kind.value, "unexpected")
        self.assertEqual(result.outcome.failure.code, "openai.image.publish_failed")
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(publisher.calls, 1)


if __name__ == "__main__":
    unittest.main()
