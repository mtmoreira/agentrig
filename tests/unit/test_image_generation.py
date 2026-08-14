from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    DataRetention,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageGenerator,
    ImageRegion,
    ImageSize,
    ImageSpecification,
    ModelMetadata,
)
from agentrig.core import (
    ArtifactId,
    ArtifactRef,
    CancellationSource,
    RunCancelled,
    RunContext,
    RunId,
)


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 14, 5, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


@dataclass
class SequentialRunIdGenerator:
    next_value: int = 1

    def generate(self) -> RunId:
        run_id = RunId(f"run-{self.next_value}")
        self.next_value += 1
        return run_id


def create_context(
    source: CancellationSource | None = None,
) -> RunContext:
    owned_source = source if source is not None else CancellationSource()
    return RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialRunIdGenerator(),
        cancellation=owned_source.token,
    )


def create_artifact(
    artifact_id: str,
    *,
    kind: str = "image",
    media_type: str = "image/png",
    input_artifact_ids: tuple[ArtifactId, ...] = (),
) -> ArtifactRef:
    extension = "png" if media_type == "image/png" else "bin"
    return ArtifactRef(
        artifact_id=ArtifactId(artifact_id),
        kind=kind,
        media_type=media_type,
        producer_run_id=RunId("run-provider"),
        workspace_path=f"artifacts/{artifact_id}.{extension}",
        input_artifact_ids=input_artifact_ids,
    )


def create_specification(
    *,
    output_media_type: str = "image/png",
) -> ImageSpecification:
    return ImageSpecification(
        prompt="  Paint a quiet moonlit harbor.\n",
        size=ImageSize(width=1024, height=768),
        output_media_type=output_media_type,
    )


def create_request(
    *,
    reference_images: tuple[ArtifactRef, ...] = (),
    mask: ArtifactRef | None = None,
    regions: tuple[ImageRegion, ...] = (),
    idempotency_key: str | None = None,
    output_media_type: str = "image/png",
) -> ImageGenerationRequest:
    return ImageGenerationRequest(
        specification=create_specification(
            output_media_type=output_media_type
        ),
        reference_images=reference_images,
        mask=mask,
        regions=regions,
        idempotency_key=idempotency_key,
    )


def create_descriptor(
    *,
    features: frozenset[CapabilityFeature] | None = None,
    limits: dict[CapabilityLimit, int] | None = None,
) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id="example.image",
        version="1",
        kind=CapabilityKind.IMAGE_GENERATION,
        features=(
            features
            if features is not None
            else frozenset(
                {
                    CapabilityFeature.REFERENCE_IMAGES,
                    CapabilityFeature.MASKS,
                    CapabilityFeature.REGIONS,
                    CapabilityFeature.IDEMPOTENCY_KEYS,
                }
            )
        ),
        limits=(
            limits
            if limits is not None
            else {CapabilityLimit.MAX_REFERENCE_IMAGES: 4}
        ),
        data_retention=DataRetention.NOT_RETAINED,
    )


@dataclass
class ScriptedImageGenerator:
    descriptor: CapabilityDescriptor
    model: ModelMetadata = field(
        default_factory=lambda: ModelMetadata(
            provider="example",
            model_id="image-1",
        )
    )
    calls: list[tuple[ImageGenerationRequest, RunContext]] = field(
        default_factory=list
    )

    async def generate(
        self,
        request: ImageGenerationRequest,
        context: RunContext,
    ) -> ImageGenerationResult:
        request.require_supported_by(self.descriptor)
        context.cancellation.raise_if_cancelled()
        if context.deadline is not None:
            context.deadline.raise_if_expired(context.clock)
        self.calls.append((request, context))
        return ImageGenerationResult(
            request=request,
            image=ArtifactRef(
                artifact_id=ArtifactId(f"generated-{len(self.calls)}"),
                kind="image",
                media_type=request.specification.output_media_type,
                producer_run_id=context.run_id,
                workspace_path=f"outputs/generated-{len(self.calls)}.png",
                input_artifact_ids=request.source_artifact_ids,
            ),
            model=self.model,
        )


async def generate_typed(
    generator: ImageGenerator,
    request: ImageGenerationRequest,
    context: RunContext,
) -> ImageGenerationResult:
    return await generator.generate(request, context)


class ImageGeometryTest(unittest.TestCase):
    def test_preserves_private_specification_and_canvas_geometry(self) -> None:
        specification = create_specification()
        region = ImageRegion(x=20, y=30, width=300, height=200)

        self.assertEqual(
            specification.prompt,
            "  Paint a quiet moonlit harbor.\n",
        )
        self.assertEqual(specification.size, ImageSize(width=1024, height=768))
        self.assertEqual(specification.output_media_type, "image/png")
        self.assertTrue(region.fits_within(specification.size))
        self.assertNotIn("moonlit harbor", repr(specification))

    def test_rejects_invalid_dimensions_regions_prompt_or_media_type(self) -> None:
        for width, height in ((0, 1), (1, -1), (True, 1)):
            with self.subTest(width=width, height=height):
                with self.assertRaises(ValueError):
                    ImageSize(width=width, height=height)  # type: ignore[arg-type]

        for values in (
            {"x": -1, "y": 0, "width": 1, "height": 1},
            {"x": 0, "y": -1, "width": 1, "height": 1},
            {"x": 0, "y": 0, "width": 0, "height": 1},
            {"x": 0, "y": 0, "width": 1, "height": False},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    ImageRegion(**values)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            ImageSpecification(
                prompt=" \n",
                size=ImageSize(width=1, height=1),
            )
        with self.assertRaises(TypeError):
            ImageSpecification(
                prompt="draw",
                size="1x1",  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            ImageSpecification(
                prompt="draw",
                size=ImageSize(width=1, height=1),
                output_media_type="text/plain",
            )
        with self.assertRaises(TypeError):
            ImageRegion(x=0, y=0, width=1, height=1).fits_within(
                "1x1"  # type: ignore[arg-type]
            )


class ImageGenerationRequestTest(unittest.TestCase):
    def test_copies_inputs_and_derives_edit_requirements(self) -> None:
        references = [
            create_artifact("reference-1"),
            create_artifact("reference-2", media_type="image/jpeg"),
        ]
        mask = create_artifact("mask", kind="mask")
        regions = [ImageRegion(x=10, y=20, width=200, height=100)]
        request = ImageGenerationRequest(
            specification=create_specification(),
            reference_images=references,  # type: ignore[arg-type]
            mask=mask,
            regions=regions,  # type: ignore[arg-type]
            idempotency_key="render-1",
        )
        references.clear()
        regions.clear()

        self.assertEqual(
            request.source_artifact_ids,
            (
                ArtifactId("reference-1"),
                ArtifactId("reference-2"),
                ArtifactId("mask"),
            ),
        )
        self.assertEqual(
            request.requirements.features,
            frozenset(
                {
                    CapabilityFeature.REFERENCE_IMAGES,
                    CapabilityFeature.MASKS,
                    CapabilityFeature.REGIONS,
                    CapabilityFeature.IDEMPOTENCY_KEYS,
                }
            ),
        )
        self.assertEqual(
            request.requirements.minimum_limits,
            {CapabilityLimit.MAX_REFERENCE_IMAGES: 2},
        )
        self.assertNotIn("moonlit harbor", repr(request))
        self.assertNotIn("reference-1", repr(request))
        request.require_supported_by(create_descriptor())

    def test_unadorned_generation_has_no_optional_requirements(self) -> None:
        request = create_request()

        self.assertEqual(request.source_artifact_ids, ())
        self.assertEqual(request.requirements.features, frozenset())
        self.assertEqual(request.requirements.minimum_limits, {})
        request.require_supported_by(
            create_descriptor(features=frozenset(), limits={})
        )

    def test_rejects_unsupported_features_and_reference_capacity(self) -> None:
        request = create_request(
            reference_images=(create_artifact("reference"),),
            mask=create_artifact("mask", kind="mask"),
        )

        with self.assertRaisesRegex(ValueError, "feature:masks"):
            request.require_supported_by(
                create_descriptor(
                    features=frozenset(
                        {CapabilityFeature.REFERENCE_IMAGES}
                    ),
                    limits={CapabilityLimit.MAX_REFERENCE_IMAGES: 1},
                )
            )
        with self.assertRaisesRegex(ValueError, "max_reference_images>=1"):
            request.require_supported_by(
                create_descriptor(
                    features=frozenset(
                        {
                            CapabilityFeature.REFERENCE_IMAGES,
                            CapabilityFeature.MASKS,
                        }
                    ),
                    limits={},
                )
            )

    def test_rejects_invalid_source_artifacts_and_edit_controls(self) -> None:
        reference = create_artifact("reference")
        invalid_reference_sets = (
            ("invalid",),
            (create_artifact("text", media_type="text/plain"),),
            (reference, reference),
        )
        for reference_images in invalid_reference_sets:
            with self.subTest(reference_images=reference_images):
                with self.assertRaises((TypeError, ValueError)):
                    create_request(
                        reference_images=reference_images,  # type: ignore[arg-type]
                    )

        with self.assertRaises(TypeError):
            create_request(
                reference_images=(reference,),
                mask="invalid",  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            create_request(
                reference_images=(reference,),
                mask=create_artifact("mask", media_type="text/plain"),
            )
        with self.assertRaises(ValueError):
            create_request(reference_images=(reference,), mask=reference)
        with self.assertRaises(ValueError):
            create_request(mask=create_artifact("mask", kind="mask"))
        with self.assertRaises(ValueError):
            create_request(
                regions=(ImageRegion(x=0, y=0, width=10, height=10),)
            )
        with self.assertRaises(TypeError):
            create_request(
                reference_images=(reference,),
                regions=("invalid",),  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            create_request(
                reference_images=(reference,),
                regions=(
                    ImageRegion(x=0, y=0, width=10, height=10),
                    ImageRegion(x=0, y=0, width=10, height=10),
                ),
            )
        with self.assertRaises(ValueError):
            create_request(
                reference_images=(reference,),
                regions=(
                    ImageRegion(x=1000, y=0, width=25, height=10),
                ),
            )
        with self.assertRaises(ValueError):
            create_request(idempotency_key=" padded ")
        with self.assertRaises(TypeError):
            ImageGenerationRequest(
                specification="invalid",  # type: ignore[arg-type]
            )


class ImageGenerationResultTest(unittest.TestCase):
    def test_preserves_generated_artifact_model_and_complete_lineage(self) -> None:
        reference = create_artifact("reference")
        mask = create_artifact("mask", kind="mask")
        request = create_request(
            reference_images=(reference,),
            mask=mask,
            output_media_type="image/webp",
        )
        image = create_artifact(
            "generated",
            media_type="image/webp",
            input_artifact_ids=(
                reference.artifact_id,
                mask.artifact_id,
                ArtifactId("additional-source"),
            ),
        )
        model = ModelMetadata(provider="example", model_id="image-1")

        result = ImageGenerationResult(
            request=request,
            image=image,
            model=model,
        )

        self.assertIs(result.image, image)
        self.assertIs(result.model, model)
        self.assertNotIn("generated", repr(result))

    def test_rejects_invalid_artifact_metadata_or_missing_lineage(self) -> None:
        reference = create_artifact("reference")
        request = create_request(reference_images=(reference,))
        model = ModelMetadata(provider="example", model_id="image-1")

        invalid_images = (
            create_artifact("wrong-kind", kind="report"),
            create_artifact("wrong-media", media_type="text/plain"),
            create_artifact("wrong-format", media_type="image/jpeg"),
            create_artifact("missing-lineage"),
        )
        for image in invalid_images:
            with self.subTest(image=image.artifact_id):
                with self.assertRaises(ValueError):
                    ImageGenerationResult(
                        request=request,
                        image=image,
                        model=model,
                    )

        with self.assertRaises(TypeError):
            ImageGenerationResult(
                request="invalid",  # type: ignore[arg-type]
                image=create_artifact("generated"),
                model=model,
            )
        with self.assertRaises(TypeError):
            ImageGenerationResult(
                request=create_request(),
                image="invalid",  # type: ignore[arg-type]
                model=model,
            )
        with self.assertRaises(TypeError):
            ImageGenerationResult(
                request=create_request(),
                image=create_artifact("generated"),
                model="invalid",  # type: ignore[arg-type]
            )


class ImageGeneratorContractTest(unittest.TestCase):
    def test_protocol_supports_a_provider_neutral_generator(self) -> None:
        generator = ScriptedImageGenerator(descriptor=create_descriptor())
        reference = create_artifact("reference")
        request = create_request(reference_images=(reference,))
        context = create_context()

        result = asyncio.run(generate_typed(generator, request, context))

        self.assertIsInstance(generator, ImageGenerator)
        self.assertEqual(
            result.image.input_artifact_ids,
            (reference.artifact_id,),
        )
        self.assertEqual(result.image.producer_run_id, context.run_id)
        self.assertEqual(generator.calls, [(request, context)])

    def test_preflight_and_constraints_run_before_execution(self) -> None:
        unsupported = ScriptedImageGenerator(
            descriptor=create_descriptor(features=frozenset(), limits={})
        )
        request = create_request(
            reference_images=(create_artifact("reference"),)
        )

        with self.assertRaises(ValueError):
            asyncio.run(unsupported.generate(request, create_context()))
        self.assertEqual(unsupported.calls, [])

        source = CancellationSource()
        source.cancel("caller stopped")
        cancelled = ScriptedImageGenerator(descriptor=create_descriptor())
        with self.assertRaises(RunCancelled):
            asyncio.run(
                cancelled.generate(create_request(), create_context(source))
            )
        self.assertEqual(cancelled.calls, [])


if __name__ == "__main__":
    unittest.main()
