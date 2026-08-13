"""Bounded, evidence-scoped workflow repair control flow."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Generic, NoReturn, TypeVar

from agentrig.core._validation import require_trimmed_string
from agentrig.core.context import RunContext
from agentrig.core.errors import AgentRigError, Failure, FailureKind
from agentrig.core.events import EventKind
from agentrig.core.grading import (
    Grade,
    GradeClassification,
    GradeEvidence,
    GradeStatus,
)
from agentrig.core.policy import GradeDecision, GradeReference
from agentrig.workflow.execution import _emit, execute_step
from agentrig.workflow.grading import GradeStep, GradeStepResult
from agentrig.workflow.step import Step, StepDescriptor

SubjectT = TypeVar("SubjectT")

_CURRENCY_PATTERN = re.compile(r"[A-Z]{3}")


@dataclass(frozen=True, slots=True, kw_only=True)
class RepairBudget:
    """Bound cumulative agentic grading cost for one repair loop."""

    max_grading_cost: float
    currency: str = "USD"

    def __post_init__(self) -> None:
        cost = _require_non_negative_finite_number(
            "repair max_grading_cost",
            self.max_grading_cost,
        )
        require_trimmed_string("repair budget currency", self.currency)
        if _CURRENCY_PATTERN.fullmatch(self.currency) is None:
            raise ValueError(
                "repair budget currency must be a three-letter code"
            )
        object.__setattr__(self, "max_grading_cost", cost)


@dataclass(frozen=True, slots=True, kw_only=True)
class RepairRequest(Generic[SubjectT]):
    """Current subject and only the evidence relevant to its repair."""

    current_subject: SubjectT = field(repr=False)
    repair_attempt: int
    failed_constraints: tuple[Grade, ...] = field(repr=False)
    evidence: tuple[GradeEvidence, ...]
    required_hard_constraints: tuple[GradeReference, ...]

    def __post_init__(self) -> None:
        _require_positive_integer("repair attempt", self.repair_attempt)
        copied_failures = tuple(self.failed_constraints)
        if not copied_failures:
            raise ValueError("repair request requires failed constraints")
        if any(not isinstance(grade, Grade) for grade in copied_failures):
            raise TypeError(
                "repair failed_constraints must contain Grade values"
            )
        if any(grade.status is GradeStatus.PASS for grade in copied_failures):
            raise ValueError("repair failed constraints must not have pass status")
        references = tuple(
            GradeReference.from_grade(grade) for grade in copied_failures
        )
        if len(references) != len(set(references)):
            raise ValueError("repair failed constraint references must be unique")

        copied_evidence = tuple(self.evidence)
        if any(
            not isinstance(item, GradeEvidence) for item in copied_evidence
        ):
            raise TypeError("repair evidence must contain GradeEvidence values")
        if copied_evidence != _evidence_for_grades(copied_failures):
            raise ValueError(
                "repair evidence must exactly match failed constraint evidence"
            )

        copied_required = tuple(self.required_hard_constraints)
        if any(
            not isinstance(reference, GradeReference)
            for reference in copied_required
        ):
            raise TypeError(
                "repair required constraints must contain GradeReference values"
            )
        if len(copied_required) != len(set(copied_required)):
            raise ValueError("repair required hard constraints must be unique")

        object.__setattr__(self, "failed_constraints", copied_failures)
        object.__setattr__(self, "evidence", copied_evidence)
        object.__setattr__(
            self,
            "required_hard_constraints",
            tuple(sorted(copied_required)),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RepairLoopResult(Generic[SubjectT]):
    """Accepted subject, final grading record, and bounded resource usage."""

    subject: SubjectT = field(repr=False)
    grading: GradeStepResult[SubjectT]
    attempts: int
    grading_cost: float
    currency: str
    required_hard_constraints: tuple[GradeReference, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.grading, GradeStepResult):
            raise TypeError(
                "repair loop result grading must be a GradeStepResult"
            )
        if self.grading.decision not in (
            GradeDecision.CONTINUE,
            GradeDecision.CONTINUE_WITH_WARNING,
        ):
            raise ValueError(
                "repair loop result requires an accepted grading decision"
            )
        _require_positive_integer("repair loop result attempts", self.attempts)
        cost = _require_non_negative_finite_number(
            "repair loop result grading_cost",
            self.grading_cost,
        )
        require_trimmed_string("repair loop result currency", self.currency)
        if _CURRENCY_PATTERN.fullmatch(self.currency) is None:
            raise ValueError(
                "repair loop result currency must be a three-letter code"
            )
        copied_required = tuple(self.required_hard_constraints)
        if any(
            not isinstance(reference, GradeReference)
            for reference in copied_required
        ):
            raise TypeError(
                "repair loop result constraints must be GradeReference values"
            )
        if len(copied_required) != len(set(copied_required)):
            raise ValueError(
                "repair loop result required hard constraints must be unique"
            )
        object.__setattr__(self, "grading_cost", cost)
        object.__setattr__(
            self,
            "required_hard_constraints",
            tuple(sorted(copied_required)),
        )

    @property
    def repairs(self) -> int:
        """Return the number of repair executions after initial grading."""
        return self.attempts - 1


@dataclass(frozen=True, slots=True, kw_only=True)
class RepairLoop(Generic[SubjectT]):
    """Re-grade bounded repairs until policy accepts or a limit terminates."""

    repair_step: Step[RepairRequest[SubjectT], SubjectT] = field(repr=False)
    grade_step: GradeStep[SubjectT] = field(repr=False)
    max_attempts: int
    budget: RepairBudget
    descriptor: StepDescriptor = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.repair_step, Step):
            raise TypeError("repair loop repair_step must satisfy Step")
        if not isinstance(self.repair_step.descriptor, StepDescriptor):
            raise TypeError(
                "repair loop repair descriptor must be a StepDescriptor"
            )
        if not isinstance(self.grade_step, GradeStep):
            raise TypeError("repair loop grade_step must be a GradeStep")
        _require_positive_integer("repair loop max_attempts", self.max_attempts)
        if not isinstance(self.budget, RepairBudget):
            raise TypeError("repair loop budget must be a RepairBudget")
        object.__setattr__(
            self,
            "descriptor",
            StepDescriptor(
                step_id=f"{self.repair_step.descriptor.step_id}.repair",
                version=self.repair_step.descriptor.version,
                effect_profile=self.repair_step.descriptor.effect_profile,
            ),
        )

    async def run(
        self,
        subject: SubjectT,
        context: RunContext,
    ) -> RepairLoopResult[SubjectT]:
        """Execute a finite grade-repair sequence under the configured budget."""
        if not isinstance(context, RunContext):
            raise TypeError("repair loop context must be a RunContext")

        current_subject = subject
        grading_cost = 0.0
        required_hard_constraints: set[GradeReference] = set()

        for attempt in range(1, self.max_attempts + 1):
            _check_constraints(context)
            grading_context = context.derive_child(
                correlation={
                    "repair_attempt": str(attempt),
                    "repair_phase": "grade",
                }
            )
            grading = (
                await execute_step(
                    self.grade_step,
                    current_subject,
                    grading_context,
                )
            ).unwrap()
            grading_cost = math.fsum(
                (
                    grading_cost,
                    _grading_cost(grading, self.budget),
                )
            )
            _enforce_budget(
                self.budget,
                grading_cost,
                attempt=attempt,
            )

            grades_by_reference = {
                GradeReference.from_grade(grade): grade
                for grade in grading.grades
            }
            missing_required = required_hard_constraints - grades_by_reference.keys()
            if missing_required:
                _raise_terminal_failure(
                    kind=FailureKind.WORKFLOW_BLOCKED,
                    message="repair grading omitted a required hard constraint",
                    code="repair.required_constraint_missing",
                    attempt=attempt,
                    extra_metadata={"missing_count": str(len(missing_required))},
                )

            regressed_required = tuple(
                grades_by_reference[reference]
                for reference in sorted(required_hard_constraints)
                if grades_by_reference[reference].status is not GradeStatus.PASS
            )
            required_hard_constraints.update(
                GradeReference.from_grade(grade)
                for grade in grading.grades
                if grade.classification is GradeClassification.HARD
                and grade.status is GradeStatus.PASS
            )

            decision = grading.decision
            if decision is GradeDecision.BLOCK:
                _raise_terminal_failure(
                    kind=FailureKind.WORKFLOW_BLOCKED,
                    message="repair grading policy blocked the subject",
                    code="repair.policy_blocked",
                    attempt=attempt,
                )
            if decision is GradeDecision.REQUEST_APPROVAL:
                _raise_terminal_failure(
                    kind=FailureKind.APPROVAL_REQUIRED,
                    message="repair grading policy requires approval",
                    code="repair.approval_required",
                    attempt=attempt,
                )

            needs_repair = (
                decision is GradeDecision.REPAIR or bool(regressed_required)
            )
            if not needs_repair:
                return RepairLoopResult(
                    subject=current_subject,
                    grading=grading,
                    attempts=attempt,
                    grading_cost=grading_cost,
                    currency=self.budget.currency,
                    required_hard_constraints=tuple(
                        required_hard_constraints
                    ),
                )

            if attempt == self.max_attempts:
                _raise_terminal_failure(
                    kind=FailureKind.WORKFLOW_BLOCKED,
                    message="repair attempts were exhausted",
                    code="repair.attempts_exhausted",
                    attempt=attempt,
                )

            failed_constraints = tuple(
                grade
                for grade in grading.grades
                if grade.status is not GradeStatus.PASS
            )
            if not failed_constraints:
                _raise_terminal_failure(
                    kind=FailureKind.WORKFLOW_BLOCKED,
                    message="repair was requested without failed constraints",
                    code="repair.failed_constraints_missing",
                    attempt=attempt,
                )

            repair_request = RepairRequest(
                current_subject=current_subject,
                repair_attempt=attempt,
                failed_constraints=failed_constraints,
                evidence=_evidence_for_grades(failed_constraints),
                required_hard_constraints=tuple(required_hard_constraints),
            )
            _emit(
                context,
                EventKind.PROGRESS_REPORTED,
                {
                    "operation": "repair",
                    "repair_attempt": attempt,
                    "next_attempt": attempt + 1,
                    "max_attempts": self.max_attempts,
                    "failed_constraint_count": len(failed_constraints),
                    "evidence_count": len(repair_request.evidence),
                    "required_hard_constraint_count": len(
                        required_hard_constraints
                    ),
                    "grading_cost": grading_cost,
                    "currency": self.budget.currency,
                },
            )
            _check_constraints(context)
            repair_context = context.derive_child(
                correlation={
                    "repair_attempt": str(attempt),
                    "repair_phase": "repair",
                }
            )
            current_subject = (
                await execute_step(
                    self.repair_step,
                    repair_request,
                    repair_context,
                )
            ).unwrap()

        raise AssertionError("bounded repair loop produced no terminal result")


def _check_constraints(context: RunContext) -> None:
    context.cancellation.raise_if_cancelled()
    if context.deadline is not None:
        context.deadline.raise_if_expired(context.clock)


def _grading_cost(
    grading: GradeStepResult[SubjectT],
    budget: RepairBudget,
) -> float:
    costs: list[float] = []
    for grade in grading.grades:
        if grade.usage is None:
            continue
        if grade.usage.currency != budget.currency:
            raise AgentRigError(
                Failure(
                    kind=FailureKind.INVALID_INPUT,
                    message="repair grading cost currency does not match its budget",
                    code="repair.budget_currency_mismatch",
                    metadata={
                        "budget_currency": budget.currency,
                        "grade_currency": grade.usage.currency,
                        "grader_id": grade.grader.grader_id,
                        "grader_version": grade.grader.version,
                    },
                )
            )
        costs.append(grade.usage.cost)
    return math.fsum(costs)


def _enforce_budget(
    budget: RepairBudget,
    grading_cost: float,
    *,
    attempt: int,
) -> None:
    if grading_cost <= budget.max_grading_cost:
        return
    raise AgentRigError(
        Failure(
            kind=FailureKind.BUDGET_EXHAUSTED,
            message="repair grading budget was exhausted",
            code="repair.budget_exhausted",
            metadata={
                "attempt": str(attempt),
                "currency": budget.currency,
                "max_grading_cost": _format_cost(budget.max_grading_cost),
                "grading_cost": _format_cost(grading_cost),
            },
        )
    )


def _evidence_for_grades(grades: tuple[Grade, ...]) -> tuple[GradeEvidence, ...]:
    evidence: list[GradeEvidence] = []
    for grade in grades:
        for item in grade.evidence:
            if item not in evidence:
                evidence.append(item)
    return tuple(evidence)


def _raise_terminal_failure(
    *,
    kind: FailureKind,
    message: str,
    code: str,
    attempt: int,
    extra_metadata: dict[str, str] | None = None,
) -> NoReturn:
    metadata = {"attempt": str(attempt)}
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    raise AgentRigError(
        Failure(
            kind=kind,
            message=message,
            code=code,
            metadata=metadata,
        )
    )


def _require_positive_integer(field_name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_non_negative_finite_number(
    field_name: str,
    value: object,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a non-negative finite number")
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError(
            f"{field_name} must be a non-negative finite number"
        ) from error
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{field_name} must be a non-negative finite number")
    return converted


def _format_cost(cost: float) -> str:
    return format(cost, ".15g")
