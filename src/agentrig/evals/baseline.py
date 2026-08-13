"""Versioned eval baselines, deterministic comparisons, and promotion policy."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, cast, runtime_checkable

from agentrig.core._json import JsonValue
from agentrig.core._validation import require_trimmed_string
from agentrig.core.grading import GradeStatus, GraderDescriptor, ScoreRange
from agentrig.core.outcomes import ExecutionStatus
from agentrig.core.policy import GradeReference
from agentrig.core.redaction import DEFAULT_REDACTION_POLICY
from agentrig.evals.report import (
    EvalReport,
    EvalReportCase,
    EvalReportGrade,
    EvalReportRetention,
)

EVAL_BASELINE_SCHEMA_VERSION = 1


class EvalChangeKind(StrEnum):
    """Stable direction of one comparison change."""

    REGRESSION = "regression"
    IMPROVEMENT = "improvement"


class EvalMetric(StrEnum):
    """Metrics that can change relative to an approved baseline."""

    CASE_STATUS = "case_status"
    GRADE_STATUS = "grade_status"
    GRADE_SCORE = "grade_score"
    DURATION_SECONDS = "duration_seconds"
    GRADER_LATENCY_SECONDS = "grader_latency_seconds"
    GRADER_COST = "grader_cost"


class EvalInconclusiveReason(StrEnum):
    """Operational conditions that prohibit a promotion decision."""

    BLOCKED_CASE = "blocked_case"
    CANCELLED_CASE = "cancelled_case"
    GRADER_FAILURE = "grader_failure"


class PromotionDecision(StrEnum):
    """Stable release decisions over baseline comparison evidence."""

    PROMOTE = "promote"
    REJECT = "reject"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, order=True, slots=True, kw_only=True)
class PromotionPolicyDescriptor:
    """Stable identity of one deterministic promotion policy."""

    policy_id: str
    version: str

    def __post_init__(self) -> None:
        require_trimmed_string("promotion policy ID", self.policy_id)
        require_trimmed_string("promotion policy version", self.version)


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalComparisonTolerance:
    """Explicit bounds for numeric regressions relative to a baseline."""

    max_grade_score_drop: float = 0.0
    max_duration_increase_ratio: float = 0.0
    max_grader_latency_increase_ratio: float = 0.0
    max_grader_cost_increase_ratio: float = 0.0

    def __post_init__(self) -> None:
        for field_name, value in (
            ("max_grade_score_drop", self.max_grade_score_drop),
            (
                "max_duration_increase_ratio",
                self.max_duration_increase_ratio,
            ),
            (
                "max_grader_latency_increase_ratio",
                self.max_grader_latency_increase_ratio,
            ),
            (
                "max_grader_cost_increase_ratio",
                self.max_grader_cost_increase_ratio,
            ),
        ):
            object.__setattr__(
                self,
                field_name,
                _non_negative_number(f"eval comparison {field_name}", value),
            )


MetricValue = str | float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalMetricChange:
    """One regression or improvement with structured comparison evidence."""

    kind: EvalChangeKind
    metric: EvalMetric
    baseline_value: MetricValue
    candidate_value: MetricValue
    allowed_regression: float = 0.0
    case_id: str | None = None
    grade: GradeReference | None = None
    currency: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EvalChangeKind):
            raise TypeError("eval metric change kind must be an EvalChangeKind")
        if not isinstance(self.metric, EvalMetric):
            raise TypeError("eval metric change metric must be an EvalMetric")
        _validate_metric_value("baseline metric value", self.baseline_value)
        _validate_metric_value("candidate metric value", self.candidate_value)
        object.__setattr__(
            self,
            "allowed_regression",
            _non_negative_number(
                "eval metric allowed_regression",
                self.allowed_regression,
            ),
        )
        if self.case_id is not None:
            require_trimmed_string("eval metric case ID", self.case_id)
        if self.grade is not None and not isinstance(self.grade, GradeReference):
            raise TypeError("eval metric grade must be a GradeReference or None")
        if self.currency is not None:
            _require_currency("eval metric currency", self.currency)

        if self.metric is EvalMetric.CASE_STATUS:
            if self.case_id is None or self.grade is not None:
                raise ValueError("case status changes require only a case ID")
            if self.currency is not None:
                raise ValueError("case status changes cannot include currency")
        elif self.metric in (EvalMetric.GRADE_STATUS, EvalMetric.GRADE_SCORE):
            if self.case_id is None or self.grade is None:
                raise ValueError("grade changes require a case ID and grade")
            if self.currency is not None:
                raise ValueError("grade changes cannot include currency")
        elif self.metric is EvalMetric.GRADER_COST:
            if self.currency is None:
                raise ValueError("grader cost changes require currency")
            if self.case_id is not None or self.grade is not None:
                raise ValueError("aggregate cost changes cannot identify a case")
        elif any(
            value is not None
            for value in (self.case_id, self.grade, self.currency)
        ):
            raise ValueError("aggregate metric changes cannot identify evidence")


@dataclass(frozen=True, order=True, slots=True, kw_only=True)
class EvalInconclusive:
    """One case-level reason that prevents promotion or rejection."""

    reason: EvalInconclusiveReason
    case_id: str
    grader: GraderDescriptor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reason, EvalInconclusiveReason):
            raise TypeError(
                "eval inconclusive reason must be an EvalInconclusiveReason"
            )
        require_trimmed_string("eval inconclusive case ID", self.case_id)
        if self.grader is not None and not isinstance(
            self.grader,
            GraderDescriptor,
        ):
            raise TypeError(
                "eval inconclusive grader must be a GraderDescriptor or None"
            )
        requires_grader = self.reason is EvalInconclusiveReason.GRADER_FAILURE
        if requires_grader != (self.grader is not None):
            raise ValueError(
                "only grader failure inconclusive evidence identifies a grader"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalComparison:
    """Deterministic evidence comparing one candidate with one baseline."""

    baseline_id: str
    baseline_version: str
    baseline_target_version: str
    candidate_target_version: str
    changes: tuple[EvalMetricChange, ...] = ()
    inconclusive: tuple[EvalInconclusive, ...] = ()

    def __post_init__(self) -> None:
        require_trimmed_string("comparison baseline ID", self.baseline_id)
        require_trimmed_string(
            "comparison baseline version",
            self.baseline_version,
        )
        require_trimmed_string(
            "comparison baseline target version",
            self.baseline_target_version,
        )
        require_trimmed_string(
            "comparison candidate target version",
            self.candidate_target_version,
        )
        copied_changes = tuple(self.changes)
        if any(
            not isinstance(change, EvalMetricChange)
            for change in copied_changes
        ):
            raise TypeError(
                "eval comparison changes must contain EvalMetricChange values"
            )
        copied_inconclusive = tuple(self.inconclusive)
        if any(
            not isinstance(item, EvalInconclusive)
            for item in copied_inconclusive
        ):
            raise TypeError(
                "eval comparison inconclusive must contain EvalInconclusive values"
            )
        if len(copied_inconclusive) != len(set(copied_inconclusive)):
            raise ValueError("eval comparison inconclusive evidence must be unique")
        object.__setattr__(self, "changes", copied_changes)
        object.__setattr__(self, "inconclusive", copied_inconclusive)

    @property
    def regressions(self) -> tuple[EvalMetricChange, ...]:
        return tuple(
            change
            for change in self.changes
            if change.kind is EvalChangeKind.REGRESSION
        )

    @property
    def improvements(self) -> tuple[EvalMetricChange, ...]:
        return tuple(
            change
            for change in self.changes
            if change.kind is EvalChangeKind.IMPROVEMENT
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalBaseline:
    """A versioned, payload-free snapshot of one approved eval report."""

    baseline_id: str
    version: str
    report: EvalReport
    schema_version: int = EVAL_BASELINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_trimmed_string("eval baseline ID", self.baseline_id)
        require_trimmed_string("eval baseline version", self.version)
        if not isinstance(self.report, EvalReport):
            raise TypeError("eval baseline report must be an EvalReport")
        if (
            isinstance(self.schema_version, bool)
            or self.schema_version != EVAL_BASELINE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported eval baseline schema version: {self.schema_version!r}"
            )
        if self.report.retention != EvalReportRetention():
            raise ValueError("eval baseline report must omit retained payloads")
        if self.report.environment:
            raise ValueError("eval baseline report must omit environment data")
        if self.report.summary.blocked_cases:
            raise ValueError("eval baseline cannot approve blocked cases")
        if self.report.summary.cancelled_cases:
            raise ValueError("eval baseline cannot approve cancelled cases")
        if self.report.summary.grader_failure_count:
            raise ValueError("eval baseline cannot approve grader failures")

        redacted = DEFAULT_REDACTION_POLICY.redact_json_object(
            self.report.to_data()
        )
        object.__setattr__(self, "report", EvalReport.from_data(redacted))

    @classmethod
    def from_report(
        cls,
        *,
        baseline_id: str,
        version: str,
        report: EvalReport,
    ) -> EvalBaseline:
        """Strip optional payloads and environment from an approved report."""
        if not isinstance(report, EvalReport):
            raise TypeError("eval baseline source must be an EvalReport")
        stripped_cases = tuple(_strip_case(case) for case in report.cases)
        stripped_report = EvalReport(
            dataset_id=report.dataset_id,
            dataset_version=report.dataset_version,
            target=report.target,
            retention=EvalReportRetention(),
            cases=stripped_cases,
            summary=report.summary,
            environment={},
        )
        return cls(
            baseline_id=baseline_id,
            version=version,
            report=stripped_report,
        )

    def to_data(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "baseline_id": self.baseline_id,
            "version": self.version,
            "report": self.report.to_data(),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_data(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> EvalBaseline:
        _require_exact_fields(
            data,
            {"schema_version", "baseline_id", "version", "report"},
            "eval baseline",
        )
        schema_version = data["schema_version"]
        if isinstance(schema_version, bool) or not isinstance(
            schema_version,
            int,
        ):
            raise ValueError("eval baseline schema version must be an integer")
        return cls(
            baseline_id=require_trimmed_string(
                "eval baseline ID",
                data["baseline_id"],
            ),
            version=require_trimmed_string(
                "eval baseline version",
                data["version"],
            ),
            report=EvalReport.from_data(
                _require_mapping("eval baseline report", data["report"])
            ),
            schema_version=schema_version,
        )

    @classmethod
    def from_json(cls, serialized: str) -> EvalBaseline:
        decoded = json.loads(serialized)
        if not isinstance(decoded, dict):
            raise ValueError("serialized eval baseline must contain a JSON object")
        return cls.from_data(decoded)


@runtime_checkable
class PromotionPolicy(Protocol):
    """Reduce complete baseline comparison evidence into a release decision."""

    @property
    def descriptor(self) -> PromotionPolicyDescriptor:
        """Return the stable identity of this policy configuration."""
        ...

    def decide(self, comparison: EvalComparison) -> PromotionDecision:
        """Return a deterministic decision without side effects."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class DeterministicPromotionPolicy:
    """Promote conclusive candidates only when no metric regresses."""

    descriptor: PromotionPolicyDescriptor

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, PromotionPolicyDescriptor):
            raise TypeError(
                "promotion policy descriptor must be a PromotionPolicyDescriptor"
            )

    def decide(self, comparison: EvalComparison) -> PromotionDecision:
        if not isinstance(comparison, EvalComparison):
            raise TypeError("promotion policy requires an EvalComparison")
        if comparison.inconclusive:
            return PromotionDecision.INCONCLUSIVE
        if comparison.regressions:
            return PromotionDecision.REJECT
        return PromotionDecision.PROMOTE


def compare_to_baseline(
    baseline: EvalBaseline,
    candidate: EvalReport,
    *,
    tolerance: EvalComparisonTolerance | None = None,
) -> EvalComparison:
    """Compare compatible reports and return ordered structured evidence."""
    if not isinstance(baseline, EvalBaseline):
        raise TypeError("baseline comparison requires an EvalBaseline")
    if not isinstance(candidate, EvalReport):
        raise TypeError("baseline comparison candidate must be an EvalReport")
    effective_tolerance = (
        tolerance if tolerance is not None else EvalComparisonTolerance()
    )
    if not isinstance(effective_tolerance, EvalComparisonTolerance):
        raise TypeError(
            "baseline comparison tolerance must be EvalComparisonTolerance"
        )

    approved = baseline.report
    _require_compatible_reports(approved, candidate)
    changes: list[EvalMetricChange] = []
    inconclusive: list[EvalInconclusive] = []

    for baseline_case, candidate_case in zip(
        approved.cases,
        candidate.cases,
        strict=True,
    ):
        _collect_inconclusive(candidate_case, inconclusive)
        if candidate_case.status in (
            ExecutionStatus.BLOCKED,
            ExecutionStatus.CANCELLED,
        ):
            continue
        _compare_case(baseline_case, candidate_case, changes, effective_tolerance)

    _compare_lower_is_better(
        metric=EvalMetric.DURATION_SECONDS,
        baseline_value=approved.summary.duration_seconds,
        candidate_value=candidate.summary.duration_seconds,
        allowed_ratio=effective_tolerance.max_duration_increase_ratio,
        changes=changes,
    )
    _compare_lower_is_better(
        metric=EvalMetric.GRADER_LATENCY_SECONDS,
        baseline_value=approved.summary.grader_latency_seconds,
        candidate_value=candidate.summary.grader_latency_seconds,
        allowed_ratio=(
            effective_tolerance.max_grader_latency_increase_ratio
        ),
        changes=changes,
    )
    _compare_costs(approved, candidate, effective_tolerance, changes)

    return EvalComparison(
        baseline_id=baseline.baseline_id,
        baseline_version=baseline.version,
        baseline_target_version=approved.target.version,
        candidate_target_version=candidate.target.version,
        changes=tuple(changes),
        inconclusive=tuple(inconclusive),
    )


def _strip_case(case: EvalReportCase) -> EvalReportCase:
    return EvalReportCase(
        case_id=case.case_id,
        case_version=case.case_version,
        run_id=case.run_id,
        status=case.status,
        failure=case.failure,
        output_retained=False,
        output=None,
        artifacts=(),
        grades=tuple(
            EvalReportGrade(
                grader=grade.grader,
                metric=grade.metric,
                status=grade.status,
                classification=grade.classification,
                score=grade.score,
                score_range=grade.score_range,
                usage=grade.usage,
                explanation=None,
                evidence=(),
                schema_version=grade.schema_version,
            )
            for grade in case.grades
        ),
        grader_failures=case.grader_failures,
        duration_seconds=case.duration_seconds,
    )


def _require_compatible_reports(
    baseline: EvalReport,
    candidate: EvalReport,
) -> None:
    if (
        candidate.dataset_id != baseline.dataset_id
        or candidate.dataset_version != baseline.dataset_version
    ):
        raise ValueError("candidate dataset does not match the baseline")
    if (
        candidate.target.target_id != baseline.target.target_id
        or candidate.target.kind is not baseline.target.kind
    ):
        raise ValueError("candidate target does not match the baseline")
    baseline_cases = tuple(
        (case.case_id, case.case_version) for case in baseline.cases
    )
    candidate_cases = tuple(
        (case.case_id, case.case_version) for case in candidate.cases
    )
    if candidate_cases != baseline_cases:
        raise ValueError("candidate case identities do not match the baseline")


def _collect_inconclusive(
    case: EvalReportCase,
    evidence: list[EvalInconclusive],
) -> None:
    if case.status is ExecutionStatus.BLOCKED:
        evidence.append(
            EvalInconclusive(
                reason=EvalInconclusiveReason.BLOCKED_CASE,
                case_id=case.case_id,
            )
        )
    elif case.status is ExecutionStatus.CANCELLED:
        evidence.append(
            EvalInconclusive(
                reason=EvalInconclusiveReason.CANCELLED_CASE,
                case_id=case.case_id,
            )
        )
    evidence.extend(
        EvalInconclusive(
            reason=EvalInconclusiveReason.GRADER_FAILURE,
            case_id=case.case_id,
            grader=failure.grader,
        )
        for failure in case.grader_failures
    )


def _compare_case(
    baseline: EvalReportCase,
    candidate: EvalReportCase,
    changes: list[EvalMetricChange],
    tolerance: EvalComparisonTolerance,
) -> None:
    if baseline.status is not candidate.status:
        kind = (
            EvalChangeKind.IMPROVEMENT
            if candidate.status is ExecutionStatus.SUCCEEDED
            else EvalChangeKind.REGRESSION
        )
        changes.append(
            EvalMetricChange(
                kind=kind,
                metric=EvalMetric.CASE_STATUS,
                baseline_value=baseline.status.value,
                candidate_value=candidate.status.value,
                case_id=baseline.case_id,
            )
        )
    elif (
        baseline.status is not ExecutionStatus.SUCCEEDED
        and baseline.failure is not None
        and candidate.failure is not None
        and baseline.failure.kind is not candidate.failure.kind
    ):
        changes.append(
            EvalMetricChange(
                kind=EvalChangeKind.REGRESSION,
                metric=EvalMetric.CASE_STATUS,
                baseline_value=(
                    f"{baseline.status.value}:{baseline.failure.kind.value}"
                ),
                candidate_value=(
                    f"{candidate.status.value}:{candidate.failure.kind.value}"
                ),
                case_id=baseline.case_id,
            )
        )
    if (
        baseline.status is not ExecutionStatus.SUCCEEDED
        or candidate.status is not ExecutionStatus.SUCCEEDED
    ):
        return
    _compare_grades(baseline, candidate, changes, tolerance)


def _compare_grades(
    baseline: EvalReportCase,
    candidate: EvalReportCase,
    changes: list[EvalMetricChange],
    tolerance: EvalComparisonTolerance,
) -> None:
    baseline_grades = {
        _grade_reference(grade): grade for grade in baseline.grades
    }
    candidate_grades = {
        _grade_reference(grade): grade for grade in candidate.grades
    }
    for reference, baseline_grade in baseline_grades.items():
        candidate_grade = candidate_grades.get(reference)
        if candidate_grade is None:
            changes.append(
                EvalMetricChange(
                    kind=EvalChangeKind.REGRESSION,
                    metric=EvalMetric.GRADE_STATUS,
                    baseline_value=baseline_grade.status.value,
                    candidate_value=None,
                    case_id=baseline.case_id,
                    grade=reference,
                )
            )
            continue
        if candidate_grade.grader != baseline_grade.grader:
            raise ValueError(
                "candidate grader descriptor does not match baseline grade: "
                f"{reference!r}"
            )
        if candidate_grade.classification is not baseline_grade.classification:
            raise ValueError(
                "candidate grade classification does not match baseline grade: "
                f"{reference!r}"
            )
        _compare_grade_status(
            baseline.case_id,
            reference,
            baseline_grade,
            candidate_grade,
            changes,
        )
        _compare_grade_score(
            baseline.case_id,
            reference,
            baseline_grade,
            candidate_grade,
            changes,
            tolerance,
        )

    for reference, candidate_grade in candidate_grades.items():
        if reference in baseline_grades or candidate_grade.status is GradeStatus.PASS:
            continue
        changes.append(
            EvalMetricChange(
                kind=EvalChangeKind.REGRESSION,
                metric=EvalMetric.GRADE_STATUS,
                baseline_value=None,
                candidate_value=candidate_grade.status.value,
                case_id=baseline.case_id,
                grade=reference,
            )
        )


def _compare_grade_status(
    case_id: str,
    reference: GradeReference,
    baseline: EvalReportGrade,
    candidate: EvalReportGrade,
    changes: list[EvalMetricChange],
) -> None:
    ranks = {
        GradeStatus.FAILURE: 0,
        GradeStatus.WARNING: 1,
        GradeStatus.PASS: 2,
    }
    if baseline.status is candidate.status:
        return
    changes.append(
        EvalMetricChange(
            kind=(
                EvalChangeKind.IMPROVEMENT
                if ranks[candidate.status] > ranks[baseline.status]
                else EvalChangeKind.REGRESSION
            ),
            metric=EvalMetric.GRADE_STATUS,
            baseline_value=baseline.status.value,
            candidate_value=candidate.status.value,
            case_id=case_id,
            grade=reference,
        )
    )


def _compare_grade_score(
    case_id: str,
    reference: GradeReference,
    baseline: EvalReportGrade,
    candidate: EvalReportGrade,
    changes: list[EvalMetricChange],
    tolerance: EvalComparisonTolerance,
) -> None:
    if baseline.score is None:
        return
    if candidate.score is None:
        changes.append(
            EvalMetricChange(
                kind=EvalChangeKind.REGRESSION,
                metric=EvalMetric.GRADE_SCORE,
                baseline_value=baseline.score,
                candidate_value=None,
                allowed_regression=tolerance.max_grade_score_drop,
                case_id=case_id,
                grade=reference,
            )
        )
        return
    _require_matching_score_ranges(reference, baseline, candidate)
    if candidate.score < baseline.score - tolerance.max_grade_score_drop:
        kind = EvalChangeKind.REGRESSION
    elif candidate.score > baseline.score:
        kind = EvalChangeKind.IMPROVEMENT
    else:
        return
    changes.append(
        EvalMetricChange(
            kind=kind,
            metric=EvalMetric.GRADE_SCORE,
            baseline_value=baseline.score,
            candidate_value=candidate.score,
            allowed_regression=tolerance.max_grade_score_drop,
            case_id=case_id,
            grade=reference,
        )
    )


def _require_matching_score_ranges(
    reference: GradeReference,
    baseline: EvalReportGrade,
    candidate: EvalReportGrade,
) -> None:
    if not isinstance(baseline.score_range, ScoreRange):
        raise AssertionError("scored baseline grade has no range")
    if candidate.score_range != baseline.score_range:
        raise ValueError(
            "candidate score range does not match baseline grade: "
            f"{reference!r}"
        )


def _compare_lower_is_better(
    *,
    metric: EvalMetric,
    baseline_value: float,
    candidate_value: float,
    allowed_ratio: float,
    changes: list[EvalMetricChange],
    currency: str | None = None,
) -> None:
    allowed_regression = baseline_value * allowed_ratio
    if candidate_value > baseline_value + allowed_regression:
        kind = EvalChangeKind.REGRESSION
    elif candidate_value < baseline_value:
        kind = EvalChangeKind.IMPROVEMENT
    else:
        return
    changes.append(
        EvalMetricChange(
            kind=kind,
            metric=metric,
            baseline_value=baseline_value,
            candidate_value=candidate_value,
            allowed_regression=allowed_regression,
            currency=currency,
        )
    )


def _compare_costs(
    baseline: EvalReport,
    candidate: EvalReport,
    tolerance: EvalComparisonTolerance,
    changes: list[EvalMetricChange],
) -> None:
    baseline_costs = {
        cost.currency: cost.amount for cost in baseline.summary.grader_costs
    }
    candidate_costs = {
        cost.currency: cost.amount for cost in candidate.summary.grader_costs
    }
    for currency in sorted(set(baseline_costs) | set(candidate_costs)):
        _compare_lower_is_better(
            metric=EvalMetric.GRADER_COST,
            baseline_value=baseline_costs.get(currency, 0.0),
            candidate_value=candidate_costs.get(currency, 0.0),
            allowed_ratio=tolerance.max_grader_cost_increase_ratio,
            changes=changes,
            currency=currency,
        )


def _grade_reference(grade: EvalReportGrade) -> GradeReference:
    return GradeReference(
        grader_id=grade.grader.grader_id,
        grader_version=grade.grader.version,
        metric=grade.metric,
    )


def _validate_metric_value(field_name: str, value: MetricValue) -> None:
    if value is None or isinstance(value, str):
        if isinstance(value, str):
            require_trimmed_string(field_name, value)
        return
    _finite_number(field_name, value)


def _finite_number(field_name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be a finite number")
    return converted


def _non_negative_number(field_name: str, value: object) -> float:
    converted = _finite_number(field_name, value)
    if converted < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return converted


def _require_currency(field_name: str, value: object) -> str:
    currency = require_trimmed_string(field_name, value)
    if (
        len(currency) != 3
        or not currency.isascii()
        or not currency.isalpha()
        or not currency.isupper()
    ):
        raise ValueError(f"{field_name} must be a three-letter code")
    return currency


def _require_exact_fields(
    data: Mapping[str, object],
    expected: set[str],
    field_name: str,
) -> None:
    actual = set(data)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{field_name} fields do not match schema; "
            f"missing={missing}, unknown={unknown}"
        )


def _require_mapping(field_name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{field_name} must be an object")
    return cast(Mapping[str, object], value)
