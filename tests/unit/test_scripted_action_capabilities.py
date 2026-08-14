from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import unittest

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    CapabilityRequirements,
    ChangedFileEvidence,
    CodingAgent,
    CodingChangeKind,
    CodingStatus,
    CodingTask,
    CodingValidationStatus,
    DataRetention,
    ImageGenerationRequest,
    ImageGenerator,
    ImageSize,
    ImageSpecification,
    ModelMetadata,
    ValidationEvidence,
    WorkspaceAuthorization,
)
from agentrig.core import (
    AgentRigError,
    ArtifactId,
    ArtifactRef,
    CancellationSource,
    Deadline,
    DeadlineExceeded,
    Failure,
    FailureKind,
    RunCancelled,
    RunContext,
    RunId,
)
from agentrig.testing import (
    CodingAgentContractSuite,
    ImageGeneratorContractSuite,
    ScriptedCodingAgent,
    ScriptedCodingScenario,
    ScriptedImageGeneration,
    ScriptedImageGenerator,
)


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 15, 7, 0, tzinfo=UTC)

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
    *,
    deadline: Deadline | None = None,
) -> RunContext:
    owned_source = source if source is not None else CancellationSource()
    return RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialRunIdGenerator(),
        cancellation=owned_source.token,
        deadline=deadline,
    )


def cancelled_context() -> RunContext:
    source = CancellationSource()
    source.cancel("contract cancellation")
    return create_context(source)


def coding_descriptor() -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id="scripted.coding",
        version="1",
        kind=CapabilityKind.CODING,
        features=frozenset({CapabilityFeature.TOOL_USE}),
        limits={CapabilityLimit.MAX_CHANGED_FILES: 2},
        data_retention=DataRetention.NOT_RETAINED,
    )


def workspace(
    writable_roots: tuple[str, ...] = ("src", "tests"),
) -> WorkspaceAuthorization:
    return WorkspaceAuthorization(
        workspace_id="storyworld",
        root_path="/work/storyworld",
        writable_roots=writable_roots,
    )


def coding_task(
    *,
    task_id: str = "task-1",
    max_changed_files: int = 1,
    writable_roots: tuple[str, ...] = ("src", "tests"),
) -> CodingTask:
    return CodingTask(
        task_id=task_id,
        workspace=workspace(writable_roots),
        objective="Implement the bounded change.",
        acceptance_criteria=("The relevant validation passes.",),
        max_changed_files=max_changed_files,
        requirements=coding_requirements(),
    )


def coding_requirements() -> CapabilityRequirements:
    return CapabilityRequirements(
        kind=CapabilityKind.CODING,
        features=frozenset({CapabilityFeature.TOOL_USE}),
        allowed_data_retention=frozenset({DataRetention.NOT_RETAINED}),
    )


def passing_validation(
    validation_id: str = "unit",
) -> ValidationEvidence:
    return ValidationEvidence(
        validation_id=validation_id,
        status=CodingValidationStatus.PASSED,
        summary="Relevant tests passed.",
        exit_code=0,
    )


def coding_scenario() -> ScriptedCodingScenario:
    return ScriptedCodingScenario(
        changed_files=(
            ChangedFileEvidence(
                path="src/story.py",
                change_kind=CodingChangeKind.MODIFIED,
            ),
        ),
        validations=(passing_validation(),),
    )


def blocked_failure() -> Failure:
    return Failure(
        kind=FailureKind.WORKFLOW_BLOCKED,
        message="external state is required",
        code="coding.waiting",
    )


def provider_failure() -> Failure:
    return Failure(
        kind=FailureKind.TRANSIENT_PROVIDER,
        message="scripted provider is temporarily unavailable",
        code="provider.busy",
    )


def image_descriptor() -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id="scripted.image",
        version="1",
        kind=CapabilityKind.IMAGE_GENERATION,
        features=frozenset({CapabilityFeature.REFERENCE_IMAGES}),
        limits={CapabilityLimit.MAX_REFERENCE_IMAGES: 1},
        data_retention=DataRetention.NOT_RETAINED,
    )


def image_artifact(artifact_id: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(artifact_id),
        kind="image",
        media_type="image/png",
        producer_run_id=RunId("run-source"),
        workspace_path=f"inputs/{artifact_id}.png",
    )


def image_request(
    *,
    references: tuple[ArtifactRef, ...] = (),
    media_type: str = "image/png",
) -> ImageGenerationRequest:
    return ImageGenerationRequest(
        specification=ImageSpecification(
            prompt="Paint a quiet harbor.",
            size=ImageSize(width=512, height=512),
            output_media_type=media_type,
        ),
        reference_images=references,
    )


def image_scenario(
    artifact_id: str = "generated-1",
) -> ScriptedImageGeneration:
    return ScriptedImageGeneration(
        artifact_id=ArtifactId(artifact_id),
        workspace_path=f"outputs/{artifact_id}.png",
        model=ModelMetadata(provider="scripted", model_id="image-1"),
    )


class ScriptedCodingAgentTest(unittest.TestCase):
    def test_returns_success_blocking_and_failures_in_order(self) -> None:
        blocker = blocked_failure()
        failure = provider_failure()
        agent = ScriptedCodingAgent(
            descriptor=coding_descriptor(),
            outcomes=(
                coding_scenario(),
                ScriptedCodingScenario(blocker=blocker),
                failure,
            ),
        )
        context = create_context()

        success = asyncio.run(agent.execute(coding_task(), context))
        snapshot = agent.calls
        blocked = asyncio.run(
            agent.execute(coding_task(task_id="task-2"), context)
        )
        with self.assertRaises(AgentRigError) as raised:
            asyncio.run(
                agent.execute(coding_task(task_id="task-3"), context)
            )

        self.assertIsInstance(agent, CodingAgent)
        self.assertEqual(success.status, CodingStatus.SUCCEEDED)
        self.assertEqual(success.task_id, "task-1")
        self.assertEqual(success.workspace_id, "storyworld")
        self.assertEqual(blocked.status, CodingStatus.BLOCKED)
        self.assertIs(blocked.blocker, blocker)
        self.assertIs(raised.exception.failure, failure)
        self.assertEqual(tuple(call.index for call in snapshot), (0,))
        self.assertEqual(tuple(call.index for call in agent.calls), (0, 1, 2))
        self.assertTrue(agent.is_exhausted)

    def test_enforces_each_task_authorization_and_changed_file_bound(self) -> None:
        agent = ScriptedCodingAgent(
            descriptor=coding_descriptor(),
            outcomes=(coding_scenario(),),
        )

        with self.assertRaisesRegex(ValueError, "outside its authorization"):
            asyncio.run(
                agent.execute(
                    coding_task(writable_roots=("tests",)),
                    create_context(),
                )
            )

        bounded = ScriptedCodingAgent(
            descriptor=coding_descriptor(),
            outcomes=(
                ScriptedCodingScenario(
                    changed_files=(
                        ChangedFileEvidence(
                            path="src/story.py",
                            change_kind=CodingChangeKind.MODIFIED,
                        ),
                        ChangedFileEvidence(
                            path="tests/test_story.py",
                            change_kind=CodingChangeKind.MODIFIED,
                        ),
                    ),
                    validations=(passing_validation(),),
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "changed-file bound"):
            asyncio.run(
                bounded.execute(
                    coding_task(max_changed_files=1),
                    create_context(),
                )
            )

    def test_preflight_and_constraints_do_not_consume_outcomes(self) -> None:
        scenario = coding_scenario()
        agent = ScriptedCodingAgent(
            descriptor=coding_descriptor(),
            outcomes=(scenario,),
        )

        with self.assertRaises(ValueError):
            asyncio.run(
                agent.execute(
                    coding_task(max_changed_files=3),
                    create_context(),
                )
            )
        with self.assertRaises(RunCancelled):
            asyncio.run(agent.execute(coding_task(), cancelled_context()))
        expired = Deadline(
            expires_at=datetime(2026, 8, 15, 7, 0, tzinfo=UTC),
            monotonic_deadline=100.0,
        )
        with self.assertRaises(DeadlineExceeded):
            asyncio.run(
                agent.execute(
                    coding_task(),
                    create_context(deadline=expired),
                )
            )

        self.assertEqual(agent.calls, ())
        self.assertFalse(agent.is_exhausted)
        result = asyncio.run(agent.execute(coding_task(), create_context()))
        self.assertTrue(result.is_success)

    def test_exhaustion_is_sanitized_and_repeat_last_is_unbounded(self) -> None:
        exhausted = ScriptedCodingAgent(
            descriptor=coding_descriptor(),
            outcomes=(coding_scenario(),),
        )
        asyncio.run(exhausted.execute(coding_task(), create_context()))

        with self.assertRaises(AgentRigError) as raised:
            asyncio.run(exhausted.execute(coding_task(), create_context()))

        self.assertEqual(
            raised.exception.failure.code,
            "scripted_coding_agent.exhausted",
        )
        self.assertEqual(
            raised.exception.failure.metadata,
            {
                "capability_id": "scripted.coding",
                "capability_version": "1",
            },
        )
        repeating = ScriptedCodingAgent(
            descriptor=coding_descriptor(),
            outcomes=(coding_scenario(),),
            repeat_last=True,
        )
        for index in range(3):
            result = asyncio.run(
                repeating.execute(
                    coding_task(task_id=f"repeat-{index}"),
                    create_context(),
                )
            )
            self.assertTrue(result.is_success)
        self.assertFalse(repeating.is_exhausted)


class ScriptedImageGeneratorTest(unittest.TestCase):
    def test_builds_request_bound_media_lineage_and_producer_identity(self) -> None:
        reference = image_artifact("reference")
        request = image_request(
            references=(reference,),
            media_type="image/webp",
        )
        generator = ScriptedImageGenerator(
            descriptor=image_descriptor(),
            outcomes=(image_scenario(),),
        )
        context = create_context()

        result = asyncio.run(generator.generate(request, context))

        self.assertIsInstance(generator, ImageGenerator)
        self.assertEqual(result.image.media_type, "image/webp")
        self.assertEqual(result.image.producer_run_id, context.run_id)
        self.assertEqual(
            result.image.input_artifact_ids,
            (reference.artifact_id,),
        )
        self.assertEqual(len(generator.calls), 1)
        self.assertIs(generator.calls[0].request, request)
        self.assertIs(generator.calls[0].context, context)

    def test_returns_failures_exhaustion_and_repeat_last(self) -> None:
        failure = provider_failure()
        generator = ScriptedImageGenerator(
            descriptor=image_descriptor(),
            outcomes=(failure,),
        )

        with self.assertRaises(AgentRigError) as raised:
            asyncio.run(generator.generate(image_request(), create_context()))
        self.assertIs(raised.exception.failure, failure)

        with self.assertRaises(AgentRigError) as exhausted:
            asyncio.run(generator.generate(image_request(), create_context()))
        self.assertEqual(
            exhausted.exception.failure.code,
            "scripted_image_generator.exhausted",
        )

        repeating = ScriptedImageGenerator(
            descriptor=image_descriptor(),
            outcomes=(image_scenario(),),
            repeat_last=True,
        )
        for _ in range(3):
            result = asyncio.run(
                repeating.generate(image_request(), create_context())
            )
            self.assertEqual(result.image.artifact_id, ArtifactId("generated-1"))
        self.assertFalse(repeating.is_exhausted)

    def test_preflight_and_constraints_do_not_consume_outcomes(self) -> None:
        generator = ScriptedImageGenerator(
            descriptor=image_descriptor(),
            outcomes=(image_scenario(),),
        )
        unsupported = image_request(
            references=(
                image_artifact("reference-1"),
                image_artifact("reference-2"),
            )
        )

        with self.assertRaises(ValueError):
            asyncio.run(generator.generate(unsupported, create_context()))
        with self.assertRaises(RunCancelled):
            asyncio.run(
                generator.generate(image_request(), cancelled_context())
            )
        expired = Deadline(
            expires_at=datetime(2026, 8, 15, 7, 0, tzinfo=UTC),
            monotonic_deadline=100.0,
        )
        with self.assertRaises(DeadlineExceeded):
            asyncio.run(
                generator.generate(
                    image_request(),
                    create_context(deadline=expired),
                )
            )

        self.assertEqual(generator.calls, ())
        self.assertFalse(generator.is_exhausted)


class ActionContractSuiteTest(unittest.TestCase):
    def test_coding_suite_verifies_shared_portable_semantics(self) -> None:
        agent = ScriptedCodingAgent(
            descriptor=coding_descriptor(),
            outcomes=(coding_scenario(),),
        )
        suite = CodingAgentContractSuite(
            agent=agent,
            supported_task=coding_task(),
            unsupported_task=coding_task(
                task_id="unsupported",
                max_changed_files=3,
            ),
            context=create_context(),
            cancelled_context=cancelled_context(),
            invocation_count=lambda: len(agent.calls),
        )

        result = asyncio.run(suite.verify())

        self.assertTrue(result.is_success)
        self.assertEqual(len(agent.calls), 1)

    def test_image_suite_verifies_shared_portable_semantics(self) -> None:
        generator = ScriptedImageGenerator(
            descriptor=image_descriptor(),
            outcomes=(image_scenario(),),
        )
        reference = image_artifact("reference")
        suite = ImageGeneratorContractSuite(
            generator=generator,
            supported_request=image_request(references=(reference,)),
            unsupported_request=image_request(
                references=(
                    reference,
                    image_artifact("reference-2"),
                )
            ),
            context=create_context(),
            cancelled_context=cancelled_context(),
            invocation_count=lambda: len(generator.calls),
        )

        result = asyncio.run(suite.verify())

        self.assertEqual(
            result.image.input_artifact_ids,
            (reference.artifact_id,),
        )
        self.assertEqual(len(generator.calls), 1)

    def test_rejects_invalid_suite_fixture_configuration(self) -> None:
        agent = ScriptedCodingAgent(
            descriptor=coding_descriptor(),
            outcomes=(coding_scenario(),),
        )
        with self.assertRaisesRegex(ValueError, "must already be cancelled"):
            CodingAgentContractSuite(
                agent=agent,
                supported_task=coding_task(),
                unsupported_task=coding_task(max_changed_files=3),
                context=create_context(),
                cancelled_context=create_context(),
                invocation_count=lambda: len(agent.calls),
            )
        with self.assertRaisesRegex(ValueError, "must be unsupported"):
            CodingAgentContractSuite(
                agent=agent,
                supported_task=coding_task(),
                unsupported_task=coding_task(),
                context=create_context(),
                cancelled_context=cancelled_context(),
                invocation_count=lambda: len(agent.calls),
            )


class ScriptedActionValidationTest(unittest.TestCase):
    def test_rejects_invalid_descriptors_outcomes_and_scenarios(self) -> None:
        with self.assertRaises(ValueError):
            ScriptedCodingAgent(
                descriptor=image_descriptor(),
                outcomes=(coding_scenario(),),
            )
        with self.assertRaises(ValueError):
            ScriptedCodingAgent(descriptor=coding_descriptor(), outcomes=())
        with self.assertRaises(TypeError):
            ScriptedCodingAgent(
                descriptor=coding_descriptor(),
                outcomes=("invalid",),  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            ScriptedCodingScenario()
        with self.assertRaises(ValueError):
            ScriptedCodingScenario(
                validations=(
                    ValidationEvidence(
                        validation_id="unit",
                        status=CodingValidationStatus.FAILED,
                        summary="Validation failed.",
                        exit_code=1,
                    ),
                )
            )
        with self.assertRaises(ValueError):
            ScriptedCodingScenario(blocker=provider_failure())
        with self.assertRaises(TypeError):
            ScriptedImageGenerator(
                descriptor=image_descriptor(),
                outcomes=("invalid",),  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            ScriptedImageGenerator(
                descriptor=coding_descriptor(),
                outcomes=(image_scenario(),),
            )
        with self.assertRaises(TypeError):
            ScriptedImageGeneration(
                artifact_id="invalid",  # type: ignore[arg-type]
                workspace_path="outputs/image.png",
                model=ModelMetadata(provider="scripted", model_id="image-1"),
            )
        with self.assertRaises(ValueError):
            ScriptedImageGeneration(
                artifact_id=ArtifactId("generated"),
                workspace_path="../outputs/image.png",
                model=ModelMetadata(provider="scripted", model_id="image-1"),
            )
        with self.assertRaises(TypeError):
            ScriptedImageGenerator(
                descriptor=image_descriptor(),
                outcomes=(image_scenario(),),
                repeat_last=1,  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
