"""Explicit approval boundary for protected workflow side effects."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, Protocol, TypeVar, runtime_checkable

from agentrig.core._validation import require_trimmed_string
from agentrig.core.cancellation import RunCancelled
from agentrig.core.context import RunContext
from agentrig.core.deadline import DeadlineExceeded
from agentrig.core.errors import AgentRigError, Failure, FailureKind
from agentrig.core.events import EventKind, JsonValue
from agentrig.workflow.execution import _emit, execute_step
from agentrig.workflow.step import Step, StepDescriptor

RequestInputT_co = TypeVar("RequestInputT_co", covariant=True)
AuthorityInputT_contra = TypeVar("AuthorityInputT_contra", contravariant=True)
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class ApprovalDecision(StrEnum):
    """Stable external decisions for one proposed action."""

    APPROVED = "approved"
    DENIED = "denied"


@dataclass(frozen=True, slots=True, kw_only=True)
class ApprovalRequest(Generic[RequestInputT_co]):
    """Explicitly scope a proposed input to one protected action."""

    approval_id: str
    action: StepDescriptor
    summary: str
    proposed_input: RequestInputT_co = field(repr=False)

    def __post_init__(self) -> None:
        require_trimmed_string("approval request ID", self.approval_id)
        if not isinstance(self.action, StepDescriptor):
            raise TypeError("approval request action must be a StepDescriptor")
        require_trimmed_string("approval request summary", self.summary)


@dataclass(frozen=True, slots=True, kw_only=True)
class ApprovalResolution:
    """Explicit external decision associated with one approval request."""

    approval_id: str
    decision: ApprovalDecision
    resolver: str
    reason: str | None = None

    def __post_init__(self) -> None:
        require_trimmed_string("approval resolution ID", self.approval_id)
        if not isinstance(self.decision, ApprovalDecision):
            raise TypeError(
                "approval resolution decision must be an ApprovalDecision"
            )
        require_trimmed_string("approval resolver", self.resolver)
        if self.reason is not None:
            require_trimmed_string("approval resolution reason", self.reason)


@runtime_checkable
class ApprovalAuthority(Protocol[AuthorityInputT_contra]):
    """Resolve one scoped request without owning the protected side effect."""

    async def resolve(
        self,
        request: ApprovalRequest[AuthorityInputT_contra],
        context: RunContext,
    ) -> ApprovalResolution:
        """Return the external decision for this exact request."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ApprovalStepResult(Generic[InputT, OutputT]):
    """Approved request, resolution, and protected action output."""

    request: ApprovalRequest[InputT]
    resolution: ApprovalResolution
    output: OutputT

    def __post_init__(self) -> None:
        if not isinstance(self.request, ApprovalRequest):
            raise TypeError(
                "approval step result request must be an ApprovalRequest"
            )
        if not isinstance(self.resolution, ApprovalResolution):
            raise TypeError(
                "approval step result resolution must be an ApprovalResolution"
            )
        if self.resolution.approval_id != self.request.approval_id:
            raise ValueError(
                "approval step result request and resolution IDs must match"
            )
        if self.resolution.decision is not ApprovalDecision.APPROVED:
            raise ValueError("approval step result requires an approved resolution")


@dataclass(frozen=True, slots=True, kw_only=True)
class ApprovalStep(Generic[InputT, OutputT]):
    """Resolve approval before invoking one explicitly scoped protected step."""

    action: Step[InputT, OutputT] = field(repr=False)
    authority: ApprovalAuthority[InputT] = field(repr=False)
    descriptor: StepDescriptor = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.action, Step):
            raise TypeError("approval action must satisfy Step")
        if not isinstance(self.action.descriptor, StepDescriptor):
            raise TypeError("approval action descriptor must be a StepDescriptor")
        if not isinstance(self.authority, ApprovalAuthority):
            raise TypeError(
                "approval authority must satisfy ApprovalAuthority"
            )
        object.__setattr__(
            self,
            "descriptor",
            StepDescriptor(
                step_id=f"{self.action.descriptor.step_id}.approval",
                version=self.action.descriptor.version,
                effect_profile=self.action.descriptor.effect_profile,
            ),
        )

    async def run(
        self,
        request: ApprovalRequest[InputT],
        context: RunContext,
    ) -> ApprovalStepResult[InputT, OutputT]:
        """Resolve the request and run the action only after explicit approval."""
        if not isinstance(request, ApprovalRequest):
            raise TypeError("approval step input must be an ApprovalRequest")
        if not isinstance(context, RunContext):
            raise TypeError("approval step context must be a RunContext")
        if request.action != self.action.descriptor:
            raise AgentRigError(
                Failure(
                    kind=FailureKind.INVALID_INPUT,
                    message="approval request does not match the protected action",
                    code="approval.action_mismatch",
                    metadata=_approval_metadata(request, self.action.descriptor),
                )
            )

        _check_constraints(context)
        _emit(
            context,
            EventKind.APPROVAL_REQUESTED,
            _request_attributes(request),
        )
        resolution = await _resolve(self.authority, request, context)
        _emit(
            context,
            EventKind.APPROVAL_RESOLVED,
            _resolution_attributes(request, resolution),
        )

        if resolution.decision is ApprovalDecision.DENIED:
            raise AgentRigError(
                Failure(
                    kind=FailureKind.APPROVAL_DENIED,
                    message="proposed action was denied",
                    code="approval.denied",
                    metadata=_approval_metadata(request, request.action),
                )
            )

        _check_constraints(context)
        action_context = context.derive_child(
            correlation={"approval_id": request.approval_id}
        )
        outcome = await execute_step(
            self.action,
            request.proposed_input,
            action_context,
        )
        return ApprovalStepResult(
            request=request,
            resolution=resolution,
            output=outcome.unwrap(),
        )


async def _resolve(
    authority: ApprovalAuthority[InputT],
    request: ApprovalRequest[InputT],
    context: RunContext,
) -> ApprovalResolution:
    try:
        resolution = await authority.resolve(request, context)
    except (asyncio.CancelledError, RunCancelled, DeadlineExceeded, AgentRigError):
        raise
    except Exception as error:
        raise AgentRigError(
            Failure(
                kind=FailureKind.UNEXPECTED,
                message="approval authority could not resolve the request",
                code="approval.resolution_failed",
                metadata=_approval_metadata(request, request.action),
            )
        ) from error

    if (
        not isinstance(resolution, ApprovalResolution)
        or resolution.approval_id != request.approval_id
    ):
        raise AgentRigError(
            Failure(
                kind=FailureKind.UNEXPECTED,
                message="approval authority returned an invalid resolution",
                code="approval.invalid_resolution",
                metadata=_approval_metadata(request, request.action),
            )
        )
    return resolution


def _check_constraints(context: RunContext) -> None:
    context.cancellation.raise_if_cancelled()
    if context.deadline is not None:
        context.deadline.raise_if_expired(context.clock)


def _request_attributes(
    request: ApprovalRequest[object],
) -> dict[str, JsonValue]:
    return {
        "approval_id": request.approval_id,
        "action_id": request.action.step_id,
        "action_version": request.action.version,
        "effect_profile": request.action.effect_profile.value,
    }


def _resolution_attributes(
    request: ApprovalRequest[object],
    resolution: ApprovalResolution,
) -> dict[str, JsonValue]:
    return {
        **_request_attributes(request),
        "decision": resolution.decision.value,
    }


def _approval_metadata(
    request: ApprovalRequest[object],
    action: StepDescriptor,
) -> dict[str, str]:
    return {
        "approval_id": request.approval_id,
        "action_id": action.step_id,
        "action_version": action.version,
    }
