"""Deterministic evaluation execution with isolated case contexts."""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from agentrig.core._validation import require_trimmed_string
from agentrig.core.artifacts import ArtifactRef
from agentrig.core.cancellation import RunCancelled
from agentrig.core.context import RunContext
from agentrig.core.deadline import DeadlineExceeded
from agentrig.core.errors import (
    AgentRigError,
    Failure,
    FailureKind,
    normalize_exception,
)
from agentrig.core.events import Event, EventKind, JsonValue
from agentrig.core.grading import (
    Grade,
    GradeStatus,
    Grader,
    GraderDescriptor,
    GradingContext,
)
from agentrig.core.identity import RunId
from agentrig.core.outcomes import ExecutionOutcome, ExecutionStatus
from agentrig.evals.case import EvalCase
from agentrig.evals.dataset import EvalDataset
from agentrig.evals.target import EvalTarget, EvalTargetDescriptor

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalSubject(Generic[InputT, OutputT]):
    """One successful target output together with its versioned case contract."""

    case: EvalCase[InputT] = field(repr=False)
    target: EvalTargetDescriptor
    output: OutputT = field(repr=False)
    artifacts: tuple[ArtifactRef, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.case, EvalCase):
            raise TypeError("eval subject case must be an EvalCase")
        if not isinstance(self.target, EvalTargetDescriptor):
            raise TypeError("eval subject target must be an EvalTargetDescriptor")
        copied_artifacts = tuple(self.artifacts)
        if any(not isinstance(item, ArtifactRef) for item in copied_artifacts):
            raise TypeError("eval subject artifacts must contain ArtifactRef values")
        artifact_ids = tuple(item.artifact_id for item in copied_artifacts)
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("eval subject artifacts must have unique IDs")
        object.__setattr__(self, "artifacts", copied_artifacts)


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalGraderFailure:
    """One normalized grader execution failure retained separately from grades."""

    grader: GraderDescriptor
    failure: Failure

    def __post_init__(self) -> None:
        if not isinstance(self.grader, GraderDescriptor):
            raise TypeError("eval grader failure grader must be a GraderDescriptor")
        if not isinstance(self.failure, Failure):
            raise TypeError("eval grader failure must contain a Failure")
        if self.failure.kind is not FailureKind.GRADER_FAILED:
            raise ValueError("eval grader failures must use the grader_failed kind")


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalCaseResult(Generic[OutputT]):
    """Target outcome, grades, and grader failures for one isolated case run."""

    case_id: str
    case_version: str
    run_id: RunId
    outcome: ExecutionOutcome[OutputT] = field(repr=False)
    grades: tuple[Grade, ...] = field(default=(), repr=False)
    grader_failures: tuple[EvalGraderFailure, ...] = field(
        default=(),
        repr=False,
    )
    duration_seconds: float

    def __post_init__(self) -> None:
        require_trimmed_string("eval result case ID", self.case_id)
        require_trimmed_string("eval result case version", self.case_version)
        if not isinstance(self.run_id, RunId):
            raise TypeError("eval case result run_id must be a RunId")
        if not isinstance(self.outcome, ExecutionOutcome):
            raise TypeError("eval case result outcome must be an ExecutionOutcome")

        copied_grades = tuple(self.grades)
        if any(not isinstance(grade, Grade) for grade in copied_grades):
            raise TypeError("eval case result grades must contain Grade values")
        grade_descriptors = tuple(grade.grader for grade in copied_grades)
        if len(grade_descriptors) != len(set(grade_descriptors)):
            raise ValueError("eval case result grade descriptors must be unique")

        copied_failures = tuple(self.grader_failures)
        if any(
            not isinstance(item, EvalGraderFailure) for item in copied_failures
        ):
            raise TypeError(
                "eval case result grader_failures must contain "
                "EvalGraderFailure values"
            )
        failure_descriptors = tuple(item.grader for item in copied_failures)
        if len(failure_descriptors) != len(set(failure_descriptors)):
            raise ValueError(
                "eval case result grader failure descriptors must be unique"
            )
        if set(grade_descriptors).intersection(failure_descriptors):
            raise ValueError("one eval grader cannot both succeed and fail")

        duration = _require_non_negative_number(
            "eval case duration_seconds",
            self.duration_seconds,
        )
        object.__setattr__(self, "grades", copied_grades)
        object.__setattr__(self, "grader_failures", copied_failures)
        object.__setattr__(self, "duration_seconds", duration)


@dataclass(frozen=True, order=True, slots=True, kw_only=True)
class EvalCost:
    """Cumulative agentic grading cost in one explicit currency."""

    currency: str
    amount: float

    def __post_init__(self) -> None:
        require_trimmed_string("eval cost currency", self.currency)
        if (
            len(self.currency) != 3
            or not self.currency.isascii()
            or not self.currency.isalpha()
            or not self.currency.isupper()
        ):
            raise ValueError("eval cost currency must be a three-letter code")
        object.__setattr__(
            self,
            "amount",
            _require_non_negative_number("eval cost amount", self.amount),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalSummary:
    """Deterministic outcome, grade, error, latency, and cost aggregates."""

    case_count: int
    succeeded_cases: int
    failed_cases: int
    blocked_cases: int
    cancelled_cases: int
    passing_grades: int
    warning_grades: int
    failing_grades: int
    grader_failure_count: int
    duration_seconds: float
    grader_latency_seconds: float
    grader_costs: tuple[EvalCost, ...] = ()

    def __post_init__(self) -> None:
        count_fields = (
            ("eval summary case_count", self.case_count),
            ("eval summary succeeded_cases", self.succeeded_cases),
            ("eval summary failed_cases", self.failed_cases),
            ("eval summary blocked_cases", self.blocked_cases),
            ("eval summary cancelled_cases", self.cancelled_cases),
            ("eval summary passing_grades", self.passing_grades),
            ("eval summary warning_grades", self.warning_grades),
            ("eval summary failing_grades", self.failing_grades),
            ("eval summary grader_failure_count", self.grader_failure_count),
        )
        for field_name, value in count_fields:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if (
            self.succeeded_cases
            + self.failed_cases
            + self.blocked_cases
            + self.cancelled_cases
            != self.case_count
        ):
            raise ValueError("eval summary outcome counts must equal case_count")

        object.__setattr__(
            self,
            "duration_seconds",
            _require_non_negative_number(
                "eval summary duration_seconds",
                self.duration_seconds,
            ),
        )
        object.__setattr__(
            self,
            "grader_latency_seconds",
            _require_non_negative_number(
                "eval summary grader_latency_seconds",
                self.grader_latency_seconds,
            ),
        )
        copied_costs = tuple(self.grader_costs)
        if any(not isinstance(cost, EvalCost) for cost in copied_costs):
            raise TypeError("eval summary grader_costs must contain EvalCost values")
        currencies = tuple(cost.currency for cost in copied_costs)
        if len(currencies) != len(set(currencies)):
            raise ValueError("eval summary grader cost currencies must be unique")
        if currencies != tuple(sorted(currencies)):
            raise ValueError("eval summary grader costs must be sorted by currency")
        object.__setattr__(self, "grader_costs", copied_costs)

    @classmethod
    def from_cases(
        cls,
        cases: tuple[EvalCaseResult[OutputT], ...],
    ) -> EvalSummary:
        """Aggregate one stable summary from an ordered set of case results."""
        status_counts = {status: 0 for status in ExecutionStatus}
        grade_counts = {status: 0 for status in GradeStatus}
        grader_failures = 0
        costs: dict[str, list[float]] = defaultdict(list)
        latencies: list[float] = []
        durations: list[float] = []

        for case in cases:
            if not isinstance(case, EvalCaseResult):
                raise TypeError("eval summary cases must contain EvalCaseResult values")
            status_counts[case.outcome.status] += 1
            durations.append(case.duration_seconds)
            grader_failures += len(case.grader_failures)
            for grade in case.grades:
                grade_counts[grade.status] += 1
                if grade.usage is not None:
                    latencies.append(grade.usage.latency_seconds)
                    costs[grade.usage.currency].append(grade.usage.cost)

        return cls(
            case_count=len(cases),
            succeeded_cases=status_counts[ExecutionStatus.SUCCEEDED],
            failed_cases=status_counts[ExecutionStatus.FAILED],
            blocked_cases=status_counts[ExecutionStatus.BLOCKED],
            cancelled_cases=status_counts[ExecutionStatus.CANCELLED],
            passing_grades=grade_counts[GradeStatus.PASS],
            warning_grades=grade_counts[GradeStatus.WARNING],
            failing_grades=grade_counts[GradeStatus.FAILURE],
            grader_failure_count=grader_failures,
            duration_seconds=math.fsum(durations),
            grader_latency_seconds=math.fsum(latencies),
            grader_costs=tuple(
                EvalCost(currency=currency, amount=math.fsum(costs[currency]))
                for currency in sorted(costs)
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalRunResult(Generic[OutputT]):
    """One dataset execution before report serialization and promotion policy."""

    dataset_id: str
    dataset_version: str
    target: EvalTargetDescriptor
    cases: tuple[EvalCaseResult[OutputT], ...]
    summary: EvalSummary

    def __post_init__(self) -> None:
        require_trimmed_string("eval run dataset ID", self.dataset_id)
        require_trimmed_string("eval run dataset version", self.dataset_version)
        if not isinstance(self.target, EvalTargetDescriptor):
            raise TypeError("eval run target must be an EvalTargetDescriptor")
        copied_cases = tuple(self.cases)
        if not copied_cases:
            raise ValueError("eval run requires at least one case result")
        if any(not isinstance(case, EvalCaseResult) for case in copied_cases):
            raise TypeError("eval run cases must contain EvalCaseResult values")
        case_ids = tuple(case.case_id for case in copied_cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("eval run case IDs must be unique")
        if not isinstance(self.summary, EvalSummary):
            raise TypeError("eval run summary must be an EvalSummary")
        expected_summary = EvalSummary.from_cases(copied_cases)
        if self.summary != expected_summary:
            raise ValueError("eval run summary must match its case results")
        object.__setattr__(self, "cases", copied_cases)


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalRunner(Generic[InputT, OutputT]):
    """Run a typed target over a dataset and invoke configured graders."""

    target: EvalTarget[InputT, OutputT]
    graders: tuple[Grader[EvalSubject[InputT, OutputT]], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.target, EvalTarget):
            raise TypeError("eval runner target must satisfy EvalTarget")
        if not isinstance(self.target.descriptor, EvalTargetDescriptor):
            raise TypeError(
                "eval runner target descriptor must be an EvalTargetDescriptor"
            )
        copied_graders = tuple(self.graders)
        if not copied_graders:
            raise ValueError("eval runner requires at least one grader")
        descriptors: list[GraderDescriptor] = []
        for grader in copied_graders:
            if not isinstance(grader, Grader):
                raise TypeError("eval runner graders must satisfy Grader")
            if not isinstance(grader.descriptor, GraderDescriptor):
                raise TypeError(
                    "eval runner grader descriptors must be GraderDescriptor values"
                )
            descriptors.append(grader.descriptor)
        if len(descriptors) != len(set(descriptors)):
            raise ValueError("eval runner grader descriptors must be unique")
        object.__setattr__(self, "graders", copied_graders)

    async def run(
        self,
        dataset: EvalDataset[InputT],
        context: RunContext,
    ) -> EvalRunResult[OutputT]:
        """Execute cases in dataset order using sibling isolated contexts."""
        if not isinstance(dataset, EvalDataset):
            raise TypeError("eval runner dataset must be an EvalDataset")
        if not isinstance(context, RunContext):
            raise TypeError("eval runner context must be a RunContext")

        case_results: list[EvalCaseResult[OutputT]] = []
        for case in dataset.cases:
            _check_constraints(context)
            case_context = context.derive_child(
                correlation={
                    "eval_dataset_id": dataset.dataset_id,
                    "eval_dataset_version": dataset.version,
                    "eval_case_id": case.case_id,
                    "eval_case_version": case.version,
                    "eval_target_id": self.target.descriptor.target_id,
                    "eval_target_version": self.target.descriptor.version,
                }
            )
            case_results.append(await self._run_case(case, case_context))

        copied_results = tuple(case_results)
        return EvalRunResult(
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.version,
            target=self.target.descriptor,
            cases=copied_results,
            summary=EvalSummary.from_cases(copied_results),
        )

    async def _run_case(
        self,
        case: EvalCase[InputT],
        context: RunContext,
    ) -> EvalCaseResult[OutputT]:
        started = context.clock.monotonic()
        attributes = _case_attributes(case, self.target.descriptor)
        _emit(context, EventKind.RUN_STARTED, attributes)
        grades: list[Grade] = []
        grader_failures: list[EvalGraderFailure] = []
        try:
            outcome = await _run_target(self.target, case.input, context)
            _check_constraints(context)

            if outcome.is_success:
                subject = EvalSubject(
                    case=case,
                    target=self.target.descriptor,
                    output=outcome.unwrap(),
                    artifacts=outcome.artifacts,
                )
                for grader in self.graders:
                    _check_constraints(context)
                    grading_context = context.derive_child(
                        correlation={
                            "grader_id": grader.descriptor.grader_id,
                            "grader_version": grader.descriptor.version,
                        }
                    )
                    grade, failure = await _run_grader(
                        grader,
                        subject,
                        grading_context,
                    )
                    _check_constraints(context)
                    if grade is not None:
                        grades.append(grade)
                        _emit(
                            grading_context,
                            EventKind.GRADE_PRODUCED,
                            _grade_attributes(grade),
                        )
                    elif failure is not None:
                        grader_failures.append(
                            EvalGraderFailure(
                                grader=grader.descriptor,
                                failure=failure,
                            )
                        )
        except (asyncio.CancelledError, RunCancelled, DeadlineExceeded) as error:
            duration = _elapsed_seconds(started, context.clock.monotonic())
            _emit_case_completed(
                context,
                attributes,
                ExecutionOutcome.from_failure(normalize_exception(error)),
                duration_seconds=duration,
                grade_count=len(grades),
                grader_failure_count=len(grader_failures),
            )
            raise

        duration = _elapsed_seconds(started, context.clock.monotonic())
        result = EvalCaseResult(
            case_id=case.case_id,
            case_version=case.version,
            run_id=context.run_id,
            outcome=outcome,
            grades=tuple(grades),
            grader_failures=tuple(grader_failures),
            duration_seconds=duration,
        )
        _emit_case_completed(
            context,
            attributes,
            outcome,
            duration_seconds=duration,
            grade_count=len(grades),
            grader_failure_count=len(grader_failures),
        )
        return result


async def _run_target(
    target: EvalTarget[InputT, OutputT],
    input: InputT,
    context: RunContext,
) -> ExecutionOutcome[OutputT]:
    try:
        outcome = await target.run(input, context)
    except (asyncio.CancelledError, RunCancelled, DeadlineExceeded):
        raise
    except Exception as error:
        return ExecutionOutcome.from_failure(normalize_exception(error))
    if not isinstance(outcome, ExecutionOutcome):
        return ExecutionOutcome.from_failure(
            Failure(
                kind=FailureKind.UNEXPECTED,
                message="eval target returned an invalid outcome",
                code="eval_target.invalid_result",
                metadata={
                    "target_id": target.descriptor.target_id,
                    "target_version": target.descriptor.version,
                },
            )
        )
    return outcome


async def _run_grader(
    grader: Grader[EvalSubject[InputT, OutputT]],
    subject: EvalSubject[InputT, OutputT],
    context: RunContext,
) -> tuple[Grade | None, Failure | None]:
    try:
        grade = await grader.grade(
            subject,
            GradingContext(
                run_context=context,
                event_sink=context.event_sink,
            ),
        )
    except (asyncio.CancelledError, RunCancelled, DeadlineExceeded):
        raise
    except AgentRigError as error:
        if error.failure.kind is FailureKind.GRADER_FAILED:
            return None, error.failure
        return None, _grader_failure(grader.descriptor, "grader.execution_failed")
    except Exception:
        return None, _grader_failure(grader.descriptor, "grader.execution_failed")

    if not isinstance(grade, Grade) or grade.grader != grader.descriptor:
        return None, _grader_failure(grader.descriptor, "grader.invalid_result")
    return grade, None


def _grader_failure(descriptor: GraderDescriptor, code: str) -> Failure:
    return Failure(
        kind=FailureKind.GRADER_FAILED,
        message="grader could not evaluate the eval subject",
        code=code,
        metadata={
            "grader_id": descriptor.grader_id,
            "grader_version": descriptor.version,
        },
    )


def _case_attributes(
    case: EvalCase[InputT],
    target: EvalTargetDescriptor,
) -> dict[str, JsonValue]:
    return {
        "eval_case_id": case.case_id,
        "eval_case_version": case.version,
        "eval_target_id": target.target_id,
        "eval_target_version": target.version,
        "eval_target_kind": target.kind.value,
    }


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


def _emit_case_completed(
    context: RunContext,
    attributes: dict[str, JsonValue],
    outcome: ExecutionOutcome[OutputT],
    *,
    duration_seconds: float,
    grade_count: int,
    grader_failure_count: int,
) -> None:
    terminal_attributes: dict[str, JsonValue] = {
        **attributes,
        "status": outcome.status.value,
        "duration_seconds": duration_seconds,
        "grade_count": grade_count,
        "grader_failure_count": grader_failure_count,
    }
    failure = outcome.failure
    if failure is not None:
        terminal_attributes["failure_kind"] = failure.kind.value
        if failure.code is not None:
            terminal_attributes["failure_code"] = failure.code
    _emit(context, _terminal_event_kind(outcome.status), terminal_attributes)


def _terminal_event_kind(status: ExecutionStatus) -> EventKind:
    if status is ExecutionStatus.SUCCEEDED:
        return EventKind.RUN_COMPLETED
    if status is ExecutionStatus.BLOCKED:
        return EventKind.RUN_BLOCKED
    if status is ExecutionStatus.CANCELLED:
        return EventKind.RUN_CANCELLED
    return EventKind.RUN_FAILED


def _emit(
    context: RunContext,
    kind: EventKind,
    attributes: dict[str, JsonValue],
) -> None:
    context.event_sink.emit(
        Event.from_context(
            event_id=context.event_id_generator.generate(),
            kind=kind,
            context=context,
            attributes=attributes,
        )
    )


def _check_constraints(context: RunContext) -> None:
    context.cancellation.raise_if_cancelled()
    if context.deadline is not None:
        context.deadline.raise_if_expired(context.clock)


def _elapsed_seconds(started: float, finished: float) -> float:
    if not math.isfinite(started) or not math.isfinite(finished):
        raise ValueError("eval clock monotonic values must be finite")
    if finished < started:
        raise ValueError("eval clock monotonic time must not move backwards")
    return finished - started


def _require_non_negative_number(field_name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite non-negative number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return converted
