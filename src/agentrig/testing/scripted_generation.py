"""Scripted text and structured generators for deterministic tests."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Generic, TypeAlias, TypeVar

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityKind,
    GenerationUsage,
    ModelMetadata,
    StructuredGenerationRequest,
    StructuredGenerationResult,
    TextGenerationFinishReason,
    TextGenerationRequest,
    TextGenerationResult,
)
from agentrig.core._json import JsonValue, freeze_json_value
from agentrig.core.context import RunContext
from agentrig.core.errors import AgentRigError, Failure, FailureKind
from agentrig.testing._scripted_outcomes import ScriptedOutcomes

OutputT = TypeVar("OutputT")

ScriptedTextGenerationOutcome: TypeAlias = TextGenerationResult | Failure


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedTextGeneratorCall:
    """One request and context presented to a scripted text generator."""

    index: int
    request: TextGenerationRequest
    context: RunContext


class ScriptedTextGenerator:
    """Return predefined text results or normalized failures in call order."""

    def __init__(
        self,
        *,
        descriptor: CapabilityDescriptor,
        outcomes: Iterable[ScriptedTextGenerationOutcome],
        repeat_last: bool = False,
    ) -> None:
        _require_descriptor_kind(
            descriptor,
            CapabilityKind.TEXT_GENERATION,
            "scripted text generator",
        )
        copied_outcomes = tuple(outcomes)
        if not copied_outcomes:
            raise ValueError(
                "scripted text generator requires at least one outcome"
            )
        if any(
            not isinstance(outcome, (TextGenerationResult, Failure))
            for outcome in copied_outcomes
        ):
            raise TypeError(
                "scripted text outcomes must contain TextGenerationResult "
                "or Failure values"
            )

        self._descriptor = descriptor
        self._script = ScriptedOutcomes[
            ScriptedTextGenerationOutcome,
            ScriptedTextGeneratorCall,
        ](outcomes=copied_outcomes, repeat_last=repeat_last)

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    @property
    def calls(self) -> tuple[ScriptedTextGeneratorCall, ...]:
        """Return a stable snapshot of recorded generator calls."""
        return self._script.calls

    @property
    def is_exhausted(self) -> bool:
        """Whether another call would raise the exhaustion failure."""
        return self._script.is_exhausted

    async def generate(
        self,
        request: TextGenerationRequest,
        context: RunContext,
    ) -> TextGenerationResult:
        """Consume one result after portable preflight and constraints."""
        if not isinstance(request, TextGenerationRequest):
            raise TypeError(
                "scripted text request must be a TextGenerationRequest"
            )
        _require_context(context, "scripted text generator")
        request.require_supported_by(self.descriptor)
        _check_constraints(context)

        outcome = self._script.record_and_take(
            lambda index: ScriptedTextGeneratorCall(
                index=index,
                request=request,
                context=context,
            )
        )
        if outcome is None:
            raise AgentRigError(
                _exhaustion_failure(
                    self.descriptor,
                    code="scripted_text_generator.exhausted",
                    message="scripted text generator has no remaining outcomes",
                )
            )
        if isinstance(outcome, Failure):
            raise AgentRigError(outcome)
        return outcome


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedStructuredGeneration:
    """Provider JSON and portable metadata for one structured outcome."""

    encoded_output: JsonValue = field(repr=False)
    usage: GenerationUsage
    model: ModelMetadata
    finish_reason: TextGenerationFinishReason

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "encoded_output",
            freeze_json_value(
                "scripted structured generation output",
                self.encoded_output,
            ),
        )
        if not isinstance(self.usage, GenerationUsage):
            raise TypeError(
                "scripted structured generation usage must be GenerationUsage"
            )
        if not isinstance(self.model, ModelMetadata):
            raise TypeError(
                "scripted structured generation model must be ModelMetadata"
            )
        if not isinstance(self.finish_reason, TextGenerationFinishReason):
            raise TypeError(
                "scripted structured generation finish_reason must be a "
                "TextGenerationFinishReason"
            )


ScriptedStructuredGenerationOutcome: TypeAlias = (
    ScriptedStructuredGeneration | Failure
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedStructuredGeneratorCall(Generic[OutputT]):
    """One request and context presented to a structured generator."""

    index: int
    request: StructuredGenerationRequest[OutputT]
    context: RunContext


class ScriptedStructuredGenerator(Generic[OutputT]):
    """Decode predefined provider JSON through each requested schema."""

    def __init__(
        self,
        *,
        descriptor: CapabilityDescriptor,
        outcomes: Iterable[ScriptedStructuredGenerationOutcome],
        repeat_last: bool = False,
    ) -> None:
        _require_descriptor_kind(
            descriptor,
            CapabilityKind.STRUCTURED_GENERATION,
            "scripted structured generator",
        )
        copied_outcomes = tuple(outcomes)
        if not copied_outcomes:
            raise ValueError(
                "scripted structured generator requires at least one outcome"
            )
        if any(
            not isinstance(
                outcome,
                (ScriptedStructuredGeneration, Failure),
            )
            for outcome in copied_outcomes
        ):
            raise TypeError(
                "scripted structured outcomes must contain "
                "ScriptedStructuredGeneration or Failure values"
            )

        self._descriptor = descriptor
        self._script = ScriptedOutcomes[
            ScriptedStructuredGenerationOutcome,
            ScriptedStructuredGeneratorCall[OutputT],
        ](outcomes=copied_outcomes, repeat_last=repeat_last)

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    @property
    def calls(self) -> tuple[ScriptedStructuredGeneratorCall[OutputT], ...]:
        """Return a stable snapshot of recorded generator calls."""
        return self._script.calls

    @property
    def is_exhausted(self) -> bool:
        """Whether another call would raise the exhaustion failure."""
        return self._script.is_exhausted

    async def generate(
        self,
        request: StructuredGenerationRequest[OutputT],
        context: RunContext,
    ) -> StructuredGenerationResult[OutputT]:
        """Consume provider JSON after portable preflight and constraints."""
        if not isinstance(request, StructuredGenerationRequest):
            raise TypeError(
                "scripted structured request must be a "
                "StructuredGenerationRequest"
            )
        _require_context(context, "scripted structured generator")
        request.require_supported_by(self.descriptor)
        _check_constraints(context)

        outcome = self._script.record_and_take(
            lambda index: ScriptedStructuredGeneratorCall(
                index=index,
                request=request,
                context=context,
            )
        )
        if outcome is None:
            raise AgentRigError(
                _exhaustion_failure(
                    self.descriptor,
                    code="scripted_structured_generator.exhausted",
                    message=(
                        "scripted structured generator has no remaining outcomes"
                    ),
                )
            )
        if isinstance(outcome, Failure):
            raise AgentRigError(outcome)
        return StructuredGenerationResult(
            encoded_output=outcome.encoded_output,
            output_schema=request.output_schema,
            usage=outcome.usage,
            model=outcome.model,
            finish_reason=outcome.finish_reason,
        )


def _require_descriptor_kind(
    descriptor: CapabilityDescriptor,
    kind: CapabilityKind,
    label: str,
) -> None:
    if not isinstance(descriptor, CapabilityDescriptor):
        raise TypeError(f"{label} descriptor must be a CapabilityDescriptor")
    if descriptor.kind is not kind:
        raise ValueError(f"{label} descriptor must use the {kind.value} kind")


def _require_context(context: RunContext, label: str) -> None:
    if not isinstance(context, RunContext):
        raise TypeError(f"{label} context must be a RunContext")


def _check_constraints(context: RunContext) -> None:
    context.cancellation.raise_if_cancelled()
    if context.deadline is not None:
        context.deadline.raise_if_expired(context.clock)


def _exhaustion_failure(
    descriptor: CapabilityDescriptor,
    *,
    code: str,
    message: str,
) -> Failure:
    return Failure(
        kind=FailureKind.UNEXPECTED,
        message=message,
        code=code,
        metadata={
            "capability_id": descriptor.capability_id,
            "capability_version": descriptor.version,
        },
    )
