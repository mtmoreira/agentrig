"""Deterministic developer walkthrough for routed image editing."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import json

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    DataRetention,
    ImageGenerationRequest,
    ImageInput,
    ImageInputRole,
    ImageSize,
    ImageSpecification,
    ModelMetadata,
)
from agentrig.core import (
    ArtifactId,
    ArtifactRef,
    CancellationSource,
    Failure,
    FailureKind,
    RunContext,
    RunId,
)
from agentrig.testing import ScriptedImageGeneration, ScriptedImageGenerator
from agentrig.workflow import (
    ImageExecution,
    ImageExecutionPolicy,
    ImageGenerationExecutor,
    ImageRoute,
)

from examples.capabilities.bounded_image_edit.workflow import (
    execute_bounded_edit,
)


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 28, 12, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


@dataclass
class SequentialIds:
    value: int = 0

    def generate(self) -> RunId:
        self.value += 1
        return RunId(f"image-example-{self.value}")


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedImageEditRun:
    execution: ImageExecution
    selected_calls: int
    unselected_calls: int


def _artifact(name: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(name),
        kind="image",
        media_type="image/png",
        producer_run_id=RunId("synthetic-inputs"),
        workspace_path=f"synthetic/{name}.png",
    )


def example_request() -> ImageGenerationRequest:
    """Return a fictional, storage-independent edit request."""
    return ImageGenerationRequest(
        specification=ImageSpecification(
            prompt=(
                "Keep both fictional characters unchanged and replace only "
                "the flat studio background with a warm observatory sky."
            ),
            size=ImageSize(width=1024, height=1536),
        ),
        inputs=(
            ImageInput(
                role=ImageInputRole.EDIT_BASE,
                artifact=_artifact("nia-tomas-base"),
            ),
            ImageInput(
                role=ImageInputRole.IDENTITY_REFERENCE,
                artifact=_artifact("nia-identity-anchor"),
            ),
            ImageInput(
                role=ImageInputRole.IDENTITY_REFERENCE,
                artifact=_artifact("tomas-identity-anchor"),
            ),
            ImageInput(
                role=ImageInputRole.EDIT_MASK,
                artifact=_artifact("background-mask"),
            ),
        ),
        idempotency_key="synthetic-observatory-edit-v1",
    )


def _descriptor(capability_id: str) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id=capability_id,
        version="1",
        kind=CapabilityKind.IMAGE_GENERATION,
        features=frozenset(
            {
                CapabilityFeature.REFERENCE_IMAGES,
                CapabilityFeature.IMAGE_EDITING,
                CapabilityFeature.MASKS,
                CapabilityFeature.IDEMPOTENCY_KEYS,
            }
        ),
        limits={CapabilityLimit.MAX_IMAGE_INPUTS: 4},
        data_retention=DataRetention.NOT_RETAINED,
    )


def create_context() -> RunContext:
    return RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialIds(),
        cancellation=CancellationSource().token,
        correlation={"example": "bounded-image-edit"},
    )


async def run_scripted_example() -> ScriptedImageEditRun:
    selected = ScriptedImageGenerator(
        descriptor=_descriptor("scripted.image.primary"),
        outcomes=(
            Failure(
                kind=FailureKind.TRANSIENT_PROVIDER,
                message="scripted route is temporarily busy",
                code="scripted.image.busy",
            ),
            ScriptedImageGeneration(
                artifact_id=ArtifactId("observatory-edit"),
                workspace_path="outputs/observatory-edit.png",
                model=ModelMetadata(
                    provider="scripted", model_id="image-edit-v1"
                ),
            ),
        ),
    )
    unselected = ScriptedImageGenerator(
        descriptor=_descriptor("scripted.image.unselected"),
        outcomes=(
            ScriptedImageGeneration(
                artifact_id=ArtifactId("must-not-run"),
                workspace_path="outputs/must-not-run.png",
                model=ModelMetadata(provider="scripted", model_id="other"),
            ),
        ),
    )
    executor = ImageGenerationExecutor(
        routes=(
            ImageRoute(route_id="primary", generator=selected),
            ImageRoute(route_id="unselected", generator=unselected),
        ),
        policy=ImageExecutionPolicy(max_attempts=2, max_concurrency=1),
    )
    execution = await execute_bounded_edit(
        executor,
        route_id="primary",
        request=example_request(),
        context=create_context(),
    )
    return ScriptedImageEditRun(
        execution=execution,
        selected_calls=len(selected.calls),
        unselected_calls=len(unselected.calls),
    )


def report(run: ScriptedImageEditRun) -> dict[str, object]:
    result = run.execution.outcome.unwrap()
    return {
        "schema_id": "example.bounded-image-edit-report.v1",
        "route_id": run.execution.route_id,
        "attempts": [
            {
                "attempt": item.attempt,
                "capability_id": item.capability_id,
                "failure_kind": (
                    item.failure_kind.value
                    if item.failure_kind is not None
                    else None
                ),
            }
            for item in run.execution.attempts
        ],
        "input_roles": [item.role.value for item in example_request().inputs],
        "output_artifact_id": str(result.image.artifact_id),
        "output_lineage": [str(item) for item in result.image.input_artifact_ids],
        "selected_route_calls": run.selected_calls,
        "unselected_route_calls": run.unselected_calls,
        "usage_cost": result.usage.cost,
        "provider_calls_live": 0,
        "network_calls": 0,
    }


def main() -> None:
    print(
        json.dumps(
            report(asyncio.run(run_scripted_example())),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
