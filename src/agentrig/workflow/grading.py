"""Typed workflow step for grading subjects and applying grade policy."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from agentrig.core.cancellation import RunCancelled
from agentrig.core.context import RunContext
from agentrig.core.deadline import DeadlineExceeded
from agentrig.core.effects import EffectProfile
from agentrig.core.errors import AgentRigError, Failure, FailureKind
from agentrig.core.events import EventKind, JsonValue
from agentrig.core.grading import (
    Grade,
    Grader,
    GraderDescriptor,
    GradingContext,
)
from agentrig.core.policy import (
    GradeDecision,
    GradePolicy,
    GradePolicyDescriptor,
    GradeReference,
)
from agentrig.workflow.execution import _emit
from agentrig.workflow.step import StepDescriptor

SubjectT = TypeVar("SubjectT")


@dataclass(frozen=True, slots=True, kw_only=True)
class GradeStepResult(Generic[SubjectT]):
    """Original subject, individual grades, and separate policy decision."""

    subject: SubjectT
    grades: tuple[Grade, ...]
    policy: GradePolicyDescriptor
    decision: GradeDecision

    def __post_init__(self) -> None:
        copied_grades = tuple(self.grades)
        if not copied_grades:
            raise ValueError("grade step result requires at least one grade")
        if any(not isinstance(grade, Grade) for grade in copied_grades):
            raise TypeError("grade step result grades must contain Grade values")
        references = tuple(
            GradeReference.from_grade(grade) for grade in copied_grades
        )
        if len(references) != len(set(references)):
            raise ValueError("grade step result grade references must be unique")
        if not isinstance(self.policy, GradePolicyDescriptor):
            raise TypeError(
                "grade step result policy must be a GradePolicyDescriptor"
            )
        if not isinstance(self.decision, GradeDecision):
            raise TypeError("grade step result decision must be a GradeDecision")
        object.__setattr__(self, "grades", copied_grades)


@dataclass(frozen=True, slots=True, kw_only=True)
class GradeStep(Generic[SubjectT]):
    """Run configured graders in order, then apply one deterministic policy."""

    graders: tuple[Grader[SubjectT], ...]
    policy: GradePolicy
    descriptor: StepDescriptor = field(init=False)

    def __post_init__(self) -> None:
        copied_graders = tuple(self.graders)
        if not copied_graders:
            raise ValueError("grade step requires at least one grader")
        descriptors: list[GraderDescriptor] = []
        for grader in copied_graders:
            if not isinstance(grader, Grader):
                raise TypeError("grade step graders must satisfy Grader")
            if not isinstance(grader.descriptor, GraderDescriptor):
                raise TypeError(
                    "grade step grader descriptors must be GraderDescriptor values"
                )
            descriptors.append(grader.descriptor)
        if len(descriptors) != len(set(descriptors)):
            raise ValueError("grade step grader descriptors must be unique")
        if not isinstance(self.policy, GradePolicy):
            raise TypeError("grade step policy must satisfy GradePolicy")
        if not isinstance(self.policy.descriptor, GradePolicyDescriptor):
            raise TypeError(
                "grade step policy descriptor must be a GradePolicyDescriptor"
            )

        object.__setattr__(self, "graders", copied_graders)
        object.__setattr__(
            self,
            "descriptor",
            StepDescriptor(
                step_id=self.policy.descriptor.policy_id,
                version=self.policy.descriptor.version,
                effect_profile=EffectProfile.READ_ONLY,
            ),
        )

    async def run(
        self,
        subject: SubjectT,
        context: RunContext,
    ) -> GradeStepResult[SubjectT]:
        """Produce all grades before independently recording policy control flow."""
        if not isinstance(context, RunContext):
            raise TypeError("grade step context must be a RunContext")

        grades: list[Grade] = []
        for grader in self.graders:
            _check_constraints(context)
            grading_run_context = context.derive_child(
                correlation={
                    "grader_id": grader.descriptor.grader_id,
                    "grader_version": grader.descriptor.version,
                }
            )
            grade = await _run_grader(grader, subject, grading_run_context)
            grades.append(grade)
            _emit(
                grading_run_context,
                EventKind.GRADE_PRODUCED,
                _grade_attributes(grade),
            )

        _check_constraints(context)
        decision = _apply_policy(self.policy, tuple(grades))
        _emit(
            context,
            EventKind.GRADE_POLICY_DECIDED,
            {
                "policy_id": self.policy.descriptor.policy_id,
                "policy_version": self.policy.descriptor.version,
                "decision": decision.value,
                "grade_count": len(grades),
            },
        )
        return GradeStepResult(
            subject=subject,
            grades=tuple(grades),
            policy=self.policy.descriptor,
            decision=decision,
        )


async def _run_grader(
    grader: Grader[SubjectT],
    subject: SubjectT,
    run_context: RunContext,
) -> Grade:
    try:
        grade = await grader.grade(
            subject,
            GradingContext(
                run_context=run_context,
                event_sink=run_context.event_sink,
            ),
        )
    except (asyncio.CancelledError, RunCancelled, DeadlineExceeded, AgentRigError):
        raise
    except Exception as error:
        raise AgentRigError(
            Failure(
                kind=FailureKind.GRADER_FAILED,
                message="grader could not evaluate the subject",
                code="grader.execution_failed",
                metadata=_grader_metadata(grader.descriptor),
            )
        ) from error

    if not isinstance(grade, Grade) or grade.grader != grader.descriptor:
        raise AgentRigError(
            Failure(
                kind=FailureKind.GRADER_FAILED,
                message="grader returned an invalid grade",
                code="grader.invalid_result",
                metadata=_grader_metadata(grader.descriptor),
            )
        )
    return grade


def _apply_policy(
    policy: GradePolicy,
    grades: tuple[Grade, ...],
) -> GradeDecision:
    try:
        decision = policy.decide(grades)
    except (asyncio.CancelledError, RunCancelled, DeadlineExceeded, AgentRigError):
        raise
    except Exception as error:
        raise AgentRigError(
            Failure(
                kind=FailureKind.UNEXPECTED,
                message="grade policy could not decide workflow control flow",
                code="grade_policy.execution_failed",
                metadata=_policy_metadata(policy.descriptor),
            )
        ) from error
    if not isinstance(decision, GradeDecision):
        raise AgentRigError(
            Failure(
                kind=FailureKind.UNEXPECTED,
                message="grade policy returned an invalid decision",
                code="grade_policy.invalid_result",
                metadata=_policy_metadata(policy.descriptor),
            )
        )
    return decision


def _check_constraints(context: RunContext) -> None:
    context.cancellation.raise_if_cancelled()
    if context.deadline is not None:
        context.deadline.raise_if_expired(context.clock)


def _grade_attributes(grade: Grade) -> dict[str, JsonValue]:
    attributes: dict[str, JsonValue] = {
        "grader_id": grade.grader.grader_id,
        "grader_version": grade.grader.version,
        "agentic": grade.grader.agentic,
        "metric": grade.metric,
        "status": grade.status.value,
        "classification": grade.classification.value,
        "schema_version": grade.schema_version,
    }
    if grade.score is not None:
        attributes["score"] = grade.score
    return attributes


def _grader_metadata(descriptor: GraderDescriptor) -> dict[str, str]:
    return {
        "grader_id": descriptor.grader_id,
        "grader_version": descriptor.version,
    }


def _policy_metadata(descriptor: GradePolicyDescriptor) -> dict[str, str]:
    return {
        "policy_id": descriptor.policy_id,
        "policy_version": descriptor.version,
    }
