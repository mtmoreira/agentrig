from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
import unittest

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageInput,
    ImageInputRole,
    ImageSize,
    ImageSpecification,
    ImageUsage,
    ModelMetadata,
)
from agentrig.core import (
    AgentRigError,
    ArtifactId,
    ArtifactRef,
    CancellationSource,
    Deadline,
    Failure,
    FailureKind,
    RunContext,
    RunId,
)
from agentrig.testing import (
    ScriptedImageGeneration,
    ScriptedImageGenerator,
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


def context(source: CancellationSource | None = None) -> RunContext:
    owned = source if source is not None else CancellationSource()
    return RunContext.create_root(
        clock=Clock(), id_generator=Ids(), cancellation=owned.token
    )


def artifact(name: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(name),
        kind="image",
        media_type="image/png",
        producer_run_id=RunId("source"),
        workspace_path=f"inputs/{name}.png",
    )


def request(*inputs: ImageInput) -> ImageGenerationRequest:
    return ImageGenerationRequest(
        specification=ImageSpecification(
            prompt="Preserve the two characters and edit the background.",
            size=ImageSize(width=1024, height=1536),
        ),
        inputs=tuple(inputs),
        idempotency_key="image-request-1",
    )


def descriptor(*, usage: bool = False) -> CapabilityDescriptor:
    features = {
        CapabilityFeature.REFERENCE_IMAGES,
        CapabilityFeature.IMAGE_EDITING,
        CapabilityFeature.IDEMPOTENCY_KEYS,
    }
    if usage:
        features.add(CapabilityFeature.COST_REPORTING)
    return CapabilityDescriptor(
        capability_id="scripted.image.route",
        version="1",
        kind=CapabilityKind.IMAGE_GENERATION,
        features=frozenset(features),
        limits={CapabilityLimit.MAX_IMAGE_INPUTS: 4},
    )


def scenario(*, usage: ImageUsage | None = None) -> ScriptedImageGeneration:
    return ScriptedImageGeneration(
        artifact_id=ArtifactId("result"),
        workspace_path="outputs/result.png",
        model=ModelMetadata(provider="scripted", model_id="image-v1"),
        usage=usage if usage is not None else ImageUsage(),
    )


class ImageContractTest(unittest.TestCase):
    def test_explicit_roles_bind_ordered_lineage_and_edit_requirements(self) -> None:
        base = artifact("base")
        style = artifact("style")
        value = request(
            ImageInput(role=ImageInputRole.EDIT_BASE, artifact=base),
            ImageInput(role=ImageInputRole.STYLE_REFERENCE, artifact=style),
        )

        self.assertEqual(
            value.source_artifact_ids,
            (base.artifact_id, style.artifact_id),
        )
        self.assertIn(
            CapabilityFeature.IMAGE_EDITING, value.requirements.features
        )
        self.assertEqual(
            value.requirements.minimum_limits[CapabilityLimit.MAX_IMAGE_INPUTS],
            2,
        )

        with self.assertRaisesRegex(ValueError, "mask requires"):
            request(
                ImageInput(
                    role=ImageInputRole.EDIT_MASK,
                    artifact=artifact("mask"),
                )
            )
        with self.assertRaisesRegex(ValueError, "cannot be mixed"):
            ImageGenerationRequest(
                specification=value.specification,
                inputs=value.inputs,
                reference_images=(artifact("legacy"),),
            )

    def test_unknown_usage_is_not_rewritten_as_zero(self) -> None:
        usage = ImageUsage()
        self.assertIsNone(usage.input_images)
        self.assertIsNone(usage.output_images)
        self.assertIsNone(usage.input_tokens)
        self.assertIsNone(usage.total_tokens)
        self.assertIsNone(usage.cost)
        with self.assertRaisesRegex(ValueError, "reported together"):
            ImageUsage(cost=0.0)

    def test_explicit_result_lineage_is_exact_and_ordered(self) -> None:
        base = artifact("base")
        style = artifact("style")
        value = request(
            ImageInput(role=ImageInputRole.EDIT_BASE, artifact=base),
            ImageInput(role=ImageInputRole.STYLE_REFERENCE, artifact=style),
        )
        for lineage in (
            (style.artifact_id, base.artifact_id),
            (*value.source_artifact_ids, ArtifactId("unexpected")),
        ):
            with self.subTest(lineage=lineage):
                with self.assertRaisesRegex(ValueError, "exactly match"):
                    ImageGenerationResult(
                        request=value,
                        image=ArtifactRef(
                            artifact_id=ArtifactId("result"),
                            kind="image",
                            media_type="image/png",
                            producer_run_id=RunId("result-run"),
                            workspace_path="outputs/result.png",
                            input_artifact_ids=lineage,
                        ),
                        model=ModelMetadata(
                            provider="scripted", model_id="image-v1"
                        ),
                    )


class ImageExecutorTest(unittest.TestCase):
    def test_retries_only_selected_route_and_retains_safe_attempts(self) -> None:
        first = ScriptedImageGenerator(
            descriptor=descriptor(),
            outcomes=(
                Failure(
                    kind=FailureKind.TRANSIENT_PROVIDER,
                    message="temporarily busy",
                    code="provider.busy",
                ),
                scenario(),
            ),
        )
        other = ScriptedImageGenerator(
            descriptor=descriptor(), outcomes=(scenario(),)
        )
        executor = ImageGenerationExecutor(
            routes=(
                ImageRoute(route_id="primary", generator=first),
                ImageRoute(route_id="other", generator=other),
            ),
            policy=ImageExecutionPolicy(max_attempts=2),
        )

        result = asyncio.run(
            executor.execute(
                route_id="primary",
                request=request(),
                context=context(),
            )
        )

        self.assertTrue(result.outcome.is_success)
        self.assertEqual(len(first.calls), 2)
        self.assertEqual(other.calls, ())
        self.assertEqual(
            tuple(item.failure_kind for item in result.attempts),
            (FailureKind.TRANSIENT_PROVIDER, None),
        )

    def test_cancellation_and_cost_preflight_do_not_invoke(self) -> None:
        generator = ScriptedImageGenerator(
            descriptor=descriptor(), outcomes=(scenario(),)
        )
        executor = ImageGenerationExecutor(
            routes=(ImageRoute(route_id="only", generator=generator),),
            policy=ImageExecutionPolicy(max_total_cost=1.0, currency="USD"),
        )
        result = asyncio.run(
            executor.execute(
                route_id="only", request=request(), context=context()
            )
        )
        self.assertEqual(result.outcome.failure.code, "image.usage_required")
        self.assertEqual(generator.calls, ())

        expired = Deadline(
            expires_at=datetime(2026, 8, 28, tzinfo=UTC),
            monotonic_deadline=100.0,
        )
        deadline_context = context()
        deadline_context = deadline_context.derive_child(deadline=expired)
        deadline_result = asyncio.run(
            ImageGenerationExecutor(
                routes=(ImageRoute(route_id="only", generator=generator),),
                policy=ImageExecutionPolicy(),
            ).execute(
                route_id="only",
                request=request(),
                context=deadline_context,
            )
        )
        self.assertEqual(
            deadline_result.outcome.failure.kind,
            FailureKind.DEADLINE_EXCEEDED,
        )
        self.assertEqual(generator.calls, ())

        source = CancellationSource()
        source.cancel("caller stopped")
        cancelled = asyncio.run(
            ImageGenerationExecutor(
                routes=(ImageRoute(route_id="only", generator=generator),),
                policy=ImageExecutionPolicy(),
            ).execute(
                route_id="only", request=request(), context=context(source)
            )
        )
        self.assertEqual(cancelled.outcome.failure.kind, FailureKind.CANCELLED)
        self.assertEqual(cancelled.attempts, ())
        self.assertEqual(generator.calls, ())

    def test_unknown_or_excess_cost_fails_closed(self) -> None:
        unknown = ScriptedImageGenerator(
            descriptor=descriptor(usage=True), outcomes=(scenario(),)
        )
        policy = ImageExecutionPolicy(max_total_cost=1.0, currency="USD")
        unknown_result = asyncio.run(
            ImageGenerationExecutor(
                routes=(ImageRoute(route_id="only", generator=unknown),),
                policy=policy,
            ).execute(route_id="only", request=request(), context=context())
        )
        self.assertEqual(
            unknown_result.outcome.failure.code, "image.usage_unknown"
        )

        expensive = ScriptedImageGenerator(
            descriptor=descriptor(usage=True),
            outcomes=(scenario(usage=ImageUsage(cost=1.5, currency="USD")),),
        )
        expensive_result = asyncio.run(
            ImageGenerationExecutor(
                routes=(ImageRoute(route_id="only", generator=expensive),),
                policy=policy,
            ).execute(route_id="only", request=request(), context=context())
        )
        self.assertEqual(
            expensive_result.outcome.failure.code, "image.cost_exhausted"
        )

    def test_shared_semaphore_bounds_concurrent_invocations(self) -> None:
        async def exercise() -> tuple[int, int]:
            generator = BlockingGenerator(descriptor=descriptor())
            executor = ImageGenerationExecutor(
                routes=(ImageRoute(route_id="only", generator=generator),),
                policy=ImageExecutionPolicy(max_concurrency=1),
            )
            first = asyncio.create_task(
                executor.execute(
                    route_id="only", request=request(), context=context()
                )
            )
            await generator.entered.wait()
            second = asyncio.create_task(
                executor.execute(
                    route_id="only", request=request(), context=context()
                )
            )
            await asyncio.sleep(0)
            calls_before_release = generator.calls
            generator.release.set()
            await asyncio.gather(first, second)
            return calls_before_release, generator.max_active

        calls, max_active = asyncio.run(exercise())
        self.assertEqual(calls, 1)
        self.assertEqual(max_active, 1)


@dataclass
class BlockingGenerator:
    descriptor: CapabilityDescriptor
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    calls: int = 0
    active: int = 0
    max_active: int = 0

    async def generate(
        self, request: ImageGenerationRequest, run_context: RunContext
    ) -> ImageGenerationResult:
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.entered.set()
        await self.release.wait()
        self.active -= 1
        return ImageGenerationResult(
            request=request,
            image=ArtifactRef(
                artifact_id=ArtifactId(f"blocked-{self.calls}"),
                kind="image",
                media_type=request.specification.output_media_type,
                producer_run_id=run_context.run_id,
                workspace_path=f"outputs/blocked-{self.calls}.png",
                input_artifact_ids=request.source_artifact_ids,
            ),
            model=ModelMetadata(provider="scripted", model_id="blocking"),
        )


if __name__ == "__main__":
    unittest.main()
