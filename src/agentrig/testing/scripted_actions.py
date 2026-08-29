"""Scripted coding and image capabilities for deterministic tests."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TypeAlias

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityKind,
    ChangedFileEvidence,
    CodingResult,
    CodingTask,
    CodingValidationStatus,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageUsage,
    ModelMetadata,
    ValidationEvidence,
)
from agentrig.core._validation import require_trimmed_string
from agentrig.core.artifacts import ArtifactId, ArtifactRef
from agentrig.core.context import RunContext
from agentrig.core.errors import AgentRigError, Failure, FailureKind
from agentrig.testing._scripted_capabilities import (
    check_constraints,
    exhaustion_failure,
    require_context,
    require_descriptor_kind,
)
from agentrig.testing._scripted_outcomes import ScriptedOutcomes

_BLOCKING_FAILURE_KINDS = frozenset(
    {
        FailureKind.APPROVAL_REQUIRED,
        FailureKind.WORKFLOW_BLOCKED,
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedCodingScenario:
    """Request-relative evidence for one coding result."""

    changed_files: tuple[ChangedFileEvidence, ...] = ()
    validations: tuple[ValidationEvidence, ...] = ()
    blocker: Failure | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        copied_changes = tuple(self.changed_files)
        if any(
            not isinstance(item, ChangedFileEvidence)
            for item in copied_changes
        ):
            raise TypeError(
                "scripted coding changes must contain ChangedFileEvidence"
            )
        changed_paths = tuple(item.path for item in copied_changes)
        if len(changed_paths) != len(set(changed_paths)):
            raise ValueError(
                "scripted coding changed paths must not contain duplicates"
            )

        copied_validations = tuple(self.validations)
        if any(
            not isinstance(item, ValidationEvidence)
            for item in copied_validations
        ):
            raise TypeError(
                "scripted coding validations must contain ValidationEvidence"
            )
        validation_ids = tuple(
            item.validation_id for item in copied_validations
        )
        if len(validation_ids) != len(set(validation_ids)):
            raise ValueError(
                "scripted coding validation IDs must not contain duplicates"
            )

        if self.blocker is None:
            if not copied_validations:
                raise ValueError(
                    "successful scripted coding requires validation evidence"
                )
            if any(
                item.status is not CodingValidationStatus.PASSED
                for item in copied_validations
            ):
                raise ValueError(
                    "successful scripted coding requires passing validations"
                )
        elif not isinstance(self.blocker, Failure):
            raise TypeError("scripted coding blocker must be a Failure or None")
        elif self.blocker.kind not in _BLOCKING_FAILURE_KINDS:
            raise ValueError(
                "scripted coding blocker must be a blocking failure kind"
            )

        object.__setattr__(self, "changed_files", copied_changes)
        object.__setattr__(self, "validations", copied_validations)


ScriptedCodingOutcome: TypeAlias = ScriptedCodingScenario | Failure


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedCodingAgentCall:
    """One task and context presented to a scripted coding agent."""

    index: int
    task: CodingTask
    context: RunContext


class ScriptedCodingAgent:
    """Build request-bound coding results from predefined scenarios."""

    def __init__(
        self,
        *,
        descriptor: CapabilityDescriptor,
        outcomes: Iterable[ScriptedCodingOutcome],
        repeat_last: bool = False,
    ) -> None:
        require_descriptor_kind(
            descriptor,
            CapabilityKind.CODING,
            "scripted coding agent",
        )
        copied_outcomes = tuple(outcomes)
        if not copied_outcomes:
            raise ValueError(
                "scripted coding agent requires at least one outcome"
            )
        if any(
            not isinstance(outcome, (ScriptedCodingScenario, Failure))
            for outcome in copied_outcomes
        ):
            raise TypeError(
                "scripted coding outcomes must contain "
                "ScriptedCodingScenario or Failure values"
            )

        self._descriptor = descriptor
        self._script = ScriptedOutcomes[
            ScriptedCodingOutcome,
            ScriptedCodingAgentCall,
        ](outcomes=copied_outcomes, repeat_last=repeat_last)

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    @property
    def calls(self) -> tuple[ScriptedCodingAgentCall, ...]:
        """Return a stable snapshot of recorded coding calls."""
        return self._script.calls

    @property
    def is_exhausted(self) -> bool:
        """Whether another call would raise the exhaustion failure."""
        return self._script.is_exhausted

    async def execute(
        self,
        task: CodingTask,
        context: RunContext,
    ) -> CodingResult:
        """Consume one scenario after portable preflight and constraints."""
        if not isinstance(task, CodingTask):
            raise TypeError("scripted coding task must be a CodingTask")
        require_context(context, "scripted coding agent")
        task.require_supported_by(self.descriptor)
        check_constraints(context)

        outcome = self._script.record_and_take(
            lambda index: ScriptedCodingAgentCall(
                index=index,
                task=task,
                context=context,
            )
        )
        if outcome is None:
            raise AgentRigError(
                exhaustion_failure(
                    self.descriptor,
                    code="scripted_coding_agent.exhausted",
                    message="scripted coding agent has no remaining outcomes",
                )
            )
        if isinstance(outcome, Failure):
            raise AgentRigError(outcome)
        if outcome.blocker is not None:
            return CodingResult.blocked(
                task=task,
                blocker=outcome.blocker,
                changed_files=outcome.changed_files,
                validations=outcome.validations,
            )
        return CodingResult.succeeded(
            task=task,
            changed_files=outcome.changed_files,
            validations=outcome.validations,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedImageGeneration:
    """Request-relative artifact identity and model for one image outcome."""

    artifact_id: ArtifactId
    workspace_path: str
    model: ModelMetadata
    usage: ImageUsage = ImageUsage()

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, ArtifactId):
            raise TypeError(
                "scripted image artifact_id must be an ArtifactId"
            )
        require_trimmed_string(
            "scripted image workspace_path",
            self.workspace_path,
        )
        parsed_path = PurePosixPath(self.workspace_path)
        if (
            "\\" in self.workspace_path
            or "\x00" in self.workspace_path
            or parsed_path.is_absolute()
            or ".." in parsed_path.parts
            or parsed_path.as_posix() in ("", ".")
            or parsed_path.as_posix() != self.workspace_path
        ):
            raise ValueError(
                "scripted image workspace_path must be canonical and "
                "workspace-relative"
            )
        if not isinstance(self.model, ModelMetadata):
            raise TypeError("scripted image model must be ModelMetadata")
        if not isinstance(self.usage, ImageUsage):
            raise TypeError("scripted image usage must be ImageUsage")


ScriptedImageGenerationOutcome: TypeAlias = ScriptedImageGeneration | Failure


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedImageGeneratorCall:
    """One request and context presented to a scripted image generator."""

    index: int
    request: ImageGenerationRequest
    context: RunContext


class ScriptedImageGenerator:
    """Build request-bound image artifacts from predefined scenarios."""

    def __init__(
        self,
        *,
        descriptor: CapabilityDescriptor,
        outcomes: Iterable[ScriptedImageGenerationOutcome],
        repeat_last: bool = False,
    ) -> None:
        require_descriptor_kind(
            descriptor,
            CapabilityKind.IMAGE_GENERATION,
            "scripted image generator",
        )
        copied_outcomes = tuple(outcomes)
        if not copied_outcomes:
            raise ValueError(
                "scripted image generator requires at least one outcome"
            )
        if any(
            not isinstance(outcome, (ScriptedImageGeneration, Failure))
            for outcome in copied_outcomes
        ):
            raise TypeError(
                "scripted image outcomes must contain "
                "ScriptedImageGeneration or Failure values"
            )

        self._descriptor = descriptor
        self._script = ScriptedOutcomes[
            ScriptedImageGenerationOutcome,
            ScriptedImageGeneratorCall,
        ](outcomes=copied_outcomes, repeat_last=repeat_last)

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    @property
    def calls(self) -> tuple[ScriptedImageGeneratorCall, ...]:
        """Return a stable snapshot of recorded image calls."""
        return self._script.calls

    @property
    def is_exhausted(self) -> bool:
        """Whether another call would raise the exhaustion failure."""
        return self._script.is_exhausted

    async def generate(
        self,
        request: ImageGenerationRequest,
        context: RunContext,
    ) -> ImageGenerationResult:
        """Consume one scenario after portable preflight and constraints."""
        if not isinstance(request, ImageGenerationRequest):
            raise TypeError(
                "scripted image request must be an ImageGenerationRequest"
            )
        require_context(context, "scripted image generator")
        request.require_supported_by(self.descriptor)
        check_constraints(context)

        outcome = self._script.record_and_take(
            lambda index: ScriptedImageGeneratorCall(
                index=index,
                request=request,
                context=context,
            )
        )
        if outcome is None:
            raise AgentRigError(
                exhaustion_failure(
                    self.descriptor,
                    code="scripted_image_generator.exhausted",
                    message="scripted image generator has no remaining outcomes",
                )
            )
        if isinstance(outcome, Failure):
            raise AgentRigError(outcome)
        image = ArtifactRef(
            artifact_id=outcome.artifact_id,
            kind="image",
            media_type=request.specification.output_media_type,
            producer_run_id=context.run_id,
            workspace_path=outcome.workspace_path,
            input_artifact_ids=request.source_artifact_ids,
        )
        return ImageGenerationResult(
            request=request,
            image=image,
            model=outcome.model,
            usage=outcome.usage,
        )
