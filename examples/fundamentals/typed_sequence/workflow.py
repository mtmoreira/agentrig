"""Provider-neutral typed sequence composition."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentrig.core import AgentRigError, EffectProfile, Failure, FailureKind, RunContext
from agentrig.workflow import (
    FunctionStep,
    RetryPolicy,
    Sequence,
    Step,
    StepDescriptor,
    Workflow,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class RawRequest:
    """Unnormalized caller input."""

    text: str = field(repr=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class NormalizedRequest:
    """Whitespace-normalized input for an injected classifier."""

    text: str = field(repr=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class ClassifiedRequest:
    """Typed output from the replaceable classification step."""

    text: str = field(repr=False)
    category: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestSummary:
    """Final typed sequence output."""

    message: str = field(repr=False)
    category: str
    character_count: int


async def _normalize(
    request: RawRequest,
    context: RunContext,
) -> NormalizedRequest:
    del context
    normalized = " ".join(request.text.split())
    if not normalized:
        raise AgentRigError(
            Failure(
                kind=FailureKind.INVALID_INPUT,
                message="request text must not be empty",
                code="example.request_empty",
            )
        )
    return NormalizedRequest(text=normalized)


async def _render(
    request: ClassifiedRequest,
    context: RunContext,
) -> RequestSummary:
    del context
    return RequestSummary(
        message=f"{request.category}: {request.text}",
        category=request.category,
        character_count=len(request.text),
    )


def build_typed_sequence(
    *,
    classifier: Step[NormalizedRequest, ClassifiedRequest],
    max_attempts: int = 2,
) -> Workflow[RawRequest, RequestSummary]:
    """Compose typed adjacent steps around one injected implementation."""
    normalize_step: Step[RawRequest, NormalizedRequest] = FunctionStep(
        descriptor=StepDescriptor(
            step_id="request.normalize",
            version="1",
            effect_profile=EffectProfile.READ_ONLY,
        ),
        function=_normalize,
    )
    render_step: Step[ClassifiedRequest, RequestSummary] = FunctionStep(
        descriptor=StepDescriptor(
            step_id="request.render-summary",
            version="1",
            effect_profile=EffectProfile.READ_ONLY,
        ),
        function=_render,
    )
    workflow: Workflow[RawRequest, RequestSummary] = Sequence(
        normalize_step,
        classifier,
        render_step,
        retry_policy=RetryPolicy(max_attempts=max_attempts),
    )
    return workflow
