"""Deterministic policies that reduce grades into workflow decisions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from agentrig.core._validation import require_trimmed_string
from agentrig.core.grading import (
    Grade,
    GradeClassification,
    GradeStatus,
)


class GradeDecision(StrEnum):
    """Stable workflow decisions made from grades."""

    CONTINUE = "continue"
    CONTINUE_WITH_WARNING = "continue_with_warning"
    REPAIR = "repair"
    REQUEST_APPROVAL = "request_approval"
    BLOCK = "block"


_DECISION_PRIORITY = {
    GradeDecision.CONTINUE: 0,
    GradeDecision.CONTINUE_WITH_WARNING: 1,
    GradeDecision.REPAIR: 2,
    GradeDecision.REQUEST_APPROVAL: 3,
    GradeDecision.BLOCK: 4,
}

_HARD_FAILURE_DECISIONS = frozenset(
    {
        GradeDecision.REPAIR,
        GradeDecision.REQUEST_APPROVAL,
        GradeDecision.BLOCK,
    }
)

_NONCONTINUING_DECISIONS = frozenset(
    {
        GradeDecision.CONTINUE_WITH_WARNING,
        GradeDecision.REPAIR,
        GradeDecision.REQUEST_APPROVAL,
        GradeDecision.BLOCK,
    }
)


@dataclass(frozen=True, order=True, slots=True, kw_only=True)
class GradePolicyDescriptor:
    """Stable identity for a deterministic grade policy configuration."""

    policy_id: str
    version: str

    def __post_init__(self) -> None:
        require_trimmed_string("grade policy ID", self.policy_id)
        require_trimmed_string("grade policy version", self.version)


@dataclass(frozen=True, order=True, slots=True, kw_only=True)
class GradeReference:
    """Identify one versioned grader metric within a grade collection."""

    grader_id: str
    grader_version: str
    metric: str

    def __post_init__(self) -> None:
        require_trimmed_string("grade reference grader ID", self.grader_id)
        require_trimmed_string(
            "grade reference grader version",
            self.grader_version,
        )
        require_trimmed_string("grade reference metric", self.metric)

    @classmethod
    def from_grade(cls, grade: Grade) -> GradeReference:
        """Create the stable reference identifying a grade."""
        if not isinstance(grade, Grade):
            raise TypeError("grade reference source must be a Grade")
        return cls(
            grader_id=grade.grader.grader_id,
            grader_version=grade.grader.version,
            metric=grade.metric,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class GradeThreshold:
    """Require a minimum score for one versioned grader metric."""

    grade: GradeReference
    minimum_score: float
    failure_decision: GradeDecision = GradeDecision.REPAIR

    def __post_init__(self) -> None:
        if not isinstance(self.grade, GradeReference):
            raise TypeError("grade threshold grade must be a GradeReference")
        minimum_score = _require_finite_score(
            "grade threshold minimum_score",
            self.minimum_score,
        )
        object.__setattr__(self, "minimum_score", minimum_score)
        _require_decision(
            "grade threshold failure_decision",
            self.failure_decision,
            allowed=_NONCONTINUING_DECISIONS,
        )


@runtime_checkable
class GradePolicy(Protocol):
    """Deterministically reduce grades without performing side effects."""

    @property
    def descriptor(self) -> GradePolicyDescriptor:
        """Return the stable identity of this policy configuration."""
        ...

    def decide(self, grades: Sequence[Grade]) -> GradeDecision:
        """Return the workflow decision for a complete grade collection."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ThresholdGradePolicy:
    """Apply hard constraints, grade statuses, and calibrated score thresholds."""

    descriptor: GradePolicyDescriptor
    thresholds: tuple[GradeThreshold, ...]
    hard_failure_decision: GradeDecision = GradeDecision.BLOCK
    soft_failure_decision: GradeDecision = GradeDecision.REPAIR
    missing_grade_decision: GradeDecision = GradeDecision.BLOCK

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, GradePolicyDescriptor):
            raise TypeError(
                "threshold policy descriptor must be a GradePolicyDescriptor"
            )
        copied_thresholds = tuple(self.thresholds)
        if not copied_thresholds:
            raise ValueError("threshold policy requires at least one threshold")
        if any(
            not isinstance(threshold, GradeThreshold)
            for threshold in copied_thresholds
        ):
            raise TypeError(
                "threshold policy thresholds must contain GradeThreshold values"
            )
        references = tuple(threshold.grade for threshold in copied_thresholds)
        if len(references) != len(set(references)):
            raise ValueError("threshold policy grade references must be unique")
        object.__setattr__(
            self,
            "thresholds",
            tuple(sorted(copied_thresholds, key=lambda item: item.grade)),
        )

        _require_decision(
            "hard_failure_decision",
            self.hard_failure_decision,
            allowed=_HARD_FAILURE_DECISIONS,
        )
        _require_decision(
            "soft_failure_decision",
            self.soft_failure_decision,
            allowed=_NONCONTINUING_DECISIONS,
        )
        _require_decision(
            "missing_grade_decision",
            self.missing_grade_decision,
            allowed=_HARD_FAILURE_DECISIONS,
        )

    def decide(self, grades: Sequence[Grade]) -> GradeDecision:
        """Return the strongest deterministic decision triggered by the grades."""
        validated_grades = _validate_grades(grades)
        grade_by_reference = {
            GradeReference.from_grade(grade): grade for grade in validated_grades
        }
        decisions = [_decision_for_status(grade, self) for grade in validated_grades]

        for threshold in self.thresholds:
            grade = grade_by_reference.get(threshold.grade)
            if grade is None or grade.score is None or grade.score_range is None:
                decisions.append(self.missing_grade_decision)
                continue
            if not (
                grade.score_range.minimum
                <= threshold.minimum_score
                <= grade.score_range.maximum
            ):
                raise ValueError(
                    "grade threshold lies outside the grade's calibrated range: "
                    f"{threshold.grade!r}"
                )
            if grade.score < threshold.minimum_score:
                decisions.append(threshold.failure_decision)

        return _strongest_decision(decisions)


@dataclass(frozen=True, slots=True, init=False)
class CompositeGradePolicy:
    """Combine policy decisions while independently protecting hard failures."""

    descriptor: GradePolicyDescriptor
    policies: tuple[GradePolicy, ...]
    hard_failure_decision: GradeDecision

    def __init__(
        self,
        *policies: GradePolicy,
        descriptor: GradePolicyDescriptor,
        hard_failure_decision: GradeDecision = GradeDecision.BLOCK,
    ) -> None:
        if not isinstance(descriptor, GradePolicyDescriptor):
            raise TypeError(
                "composite policy descriptor must be a GradePolicyDescriptor"
            )
        if not policies:
            raise ValueError("composite grade policy requires at least one policy")
        copied_policies = tuple(policies)
        for policy in copied_policies:
            if not isinstance(policy, GradePolicy):
                raise TypeError(
                    "composite policies must implement the GradePolicy protocol"
                )
            if not isinstance(policy.descriptor, GradePolicyDescriptor):
                raise TypeError(
                    "composite child descriptors must be GradePolicyDescriptor values"
                )
        descriptors = tuple(policy.descriptor for policy in copied_policies)
        if len(descriptors) != len(set(descriptors)):
            raise ValueError("composite child policy descriptors must be unique")
        _require_decision(
            "hard_failure_decision",
            hard_failure_decision,
            allowed=_HARD_FAILURE_DECISIONS,
        )

        object.__setattr__(self, "descriptor", descriptor)
        object.__setattr__(
            self,
            "policies",
            tuple(sorted(copied_policies, key=lambda policy: policy.descriptor)),
        )
        object.__setattr__(
            self,
            "hard_failure_decision",
            hard_failure_decision,
        )

    def decide(self, grades: Sequence[Grade]) -> GradeDecision:
        """Return the strongest child decision, never laundering hard failures."""
        validated_grades = _validate_grades(grades)
        decisions: list[GradeDecision] = []
        for policy in self.policies:
            decision = policy.decide(validated_grades)
            if not isinstance(decision, GradeDecision):
                raise TypeError(
                    "composite child policies must return a GradeDecision"
                )
            decisions.append(decision)

        if any(
            grade.classification is GradeClassification.HARD
            and grade.status is GradeStatus.FAILURE
            for grade in validated_grades
        ):
            decisions.append(self.hard_failure_decision)

        return _strongest_decision(decisions)


def _decision_for_status(
    grade: Grade,
    policy: ThresholdGradePolicy,
) -> GradeDecision:
    if grade.status is GradeStatus.WARNING:
        return GradeDecision.CONTINUE_WITH_WARNING
    if grade.status is GradeStatus.FAILURE:
        if grade.classification is GradeClassification.HARD:
            return policy.hard_failure_decision
        return policy.soft_failure_decision
    return GradeDecision.CONTINUE


def _validate_grades(grades: Sequence[Grade]) -> tuple[Grade, ...]:
    copied_grades = tuple(grades)
    if any(not isinstance(grade, Grade) for grade in copied_grades):
        raise TypeError("grade policies require Grade values")
    references = tuple(GradeReference.from_grade(grade) for grade in copied_grades)
    if len(references) != len(set(references)):
        raise ValueError("grade collections must have unique references")
    return copied_grades


def _strongest_decision(decisions: Sequence[GradeDecision]) -> GradeDecision:
    return max(
        decisions,
        key=lambda decision: _DECISION_PRIORITY[decision],
        default=GradeDecision.CONTINUE,
    )


def _require_decision(
    field_name: str,
    value: object,
    *,
    allowed: frozenset[GradeDecision],
) -> GradeDecision:
    if not isinstance(value, GradeDecision):
        raise TypeError(f"{field_name} must be a GradeDecision")
    if value not in allowed:
        allowed_values = ", ".join(sorted(decision.value for decision in allowed))
        raise ValueError(f"{field_name} must be one of: {allowed_values}")
    return value


def _require_finite_score(field_name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError(f"{field_name} must be a finite number") from error
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be a finite number")
    return converted
