"""Versioned, privacy-aware JSON reports for deterministic eval runs."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeVar, cast

from agentrig.core._json import (
    JsonValue,
    freeze_json_object,
    freeze_json_value,
    thaw_json_value,
)
from agentrig.core._validation import freeze_string_map, require_trimmed_string
from agentrig.core.artifacts import ArtifactRef
from agentrig.core.errors import Failure, FailureKind
from agentrig.core.grading import (
    GRADE_SCHEMA_VERSION,
    Grade,
    GradeClassification,
    GradeEvidence,
    GradeStatus,
    GraderDescriptor,
    GraderUsage,
    ScoreRange,
)
from agentrig.core.identity import RunId
from agentrig.core.outcomes import ExecutionOutcome, ExecutionStatus
from agentrig.core.redaction import (
    DEFAULT_REDACTION_POLICY,
    JsonRedactionPolicy,
)
from agentrig.evals.runner import (
    EvalCaseResult,
    EvalCost,
    EvalGraderFailure,
    EvalRunResult,
    EvalSummary,
)
from agentrig.evals.target import EvalTargetDescriptor, EvalTargetKind

EVAL_REPORT_SCHEMA_VERSION = 1

OutputT = TypeVar("OutputT")
EnumT = TypeVar("EnumT", bound=StrEnum)


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalReportRetention:
    """Explicit controls for potentially private report payloads."""

    outputs: bool = False
    artifacts: bool = False
    grade_details: bool = False

    def __post_init__(self) -> None:
        for field_name, value in (
            ("outputs", self.outputs),
            ("artifacts", self.artifacts),
            ("grade_details", self.grade_details),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"eval report retention {field_name} must be a bool")

    def to_data(self) -> dict[str, JsonValue]:
        return {
            "outputs": self.outputs,
            "artifacts": self.artifacts,
            "grade_details": self.grade_details,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> EvalReportRetention:
        _require_exact_fields(
            data,
            {"outputs", "artifacts", "grade_details"},
            "eval report retention",
        )
        return cls(
            outputs=_require_bool("retention outputs", data["outputs"]),
            artifacts=_require_bool("retention artifacts", data["artifacts"]),
            grade_details=_require_bool(
                "retention grade_details",
                data["grade_details"],
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalReportArtifact:
    """A retained artifact reference represented without dereferencing content."""

    artifact_id: str
    kind: str
    media_type: str
    producer_run_id: str
    uri: str | None = None
    workspace_path: str | None = None
    content_digest: str | None = None
    input_artifact_ids: tuple[str, ...] = ()
    labels: Mapping[str, str] = field(default_factory=dict)
    provider_lineage: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_trimmed_string("report artifact ID", self.artifact_id)
        require_trimmed_string("report artifact kind", self.kind)
        require_trimmed_string("report artifact media type", self.media_type)
        require_trimmed_string(
            "report artifact producer run ID",
            self.producer_run_id,
        )
        if (self.uri is None) == (self.workspace_path is None):
            raise ValueError(
                "report artifact requires exactly one URI or workspace path"
            )
        if self.uri is not None:
            require_trimmed_string("report artifact URI", self.uri)
        if self.workspace_path is not None:
            require_trimmed_string(
                "report artifact workspace path",
                self.workspace_path,
            )
        if self.content_digest is not None:
            require_trimmed_string(
                "report artifact content digest",
                self.content_digest,
            )
        copied_inputs = tuple(self.input_artifact_ids)
        for artifact_id in copied_inputs:
            require_trimmed_string("report input artifact ID", artifact_id)
        if len(copied_inputs) != len(set(copied_inputs)):
            raise ValueError("report input artifact IDs must be unique")
        if self.artifact_id in copied_inputs:
            raise ValueError("report artifact cannot reference itself as input")
        object.__setattr__(self, "input_artifact_ids", copied_inputs)
        object.__setattr__(
            self,
            "labels",
            freeze_string_map("report artifact labels", self.labels),
        )
        object.__setattr__(
            self,
            "provider_lineage",
            freeze_string_map(
                "report artifact provider lineage",
                self.provider_lineage,
            ),
        )

    @classmethod
    def from_artifact(cls, artifact: ArtifactRef) -> EvalReportArtifact:
        if not isinstance(artifact, ArtifactRef):
            raise TypeError("report artifact source must be an ArtifactRef")
        return cls(
            artifact_id=artifact.artifact_id.value,
            kind=artifact.kind,
            media_type=artifact.media_type,
            producer_run_id=artifact.producer_run_id.value,
            uri=artifact.uri,
            workspace_path=artifact.workspace_path,
            content_digest=(
                str(artifact.content_digest)
                if artifact.content_digest is not None
                else None
            ),
            input_artifact_ids=tuple(
                artifact_id.value for artifact_id in artifact.input_artifact_ids
            ),
            labels=artifact.labels,
            provider_lineage=artifact.provider_lineage,
        )

    def to_data(self) -> dict[str, JsonValue]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "media_type": self.media_type,
            "producer_run_id": self.producer_run_id,
            "uri": self.uri,
            "workspace_path": self.workspace_path,
            "content_digest": self.content_digest,
            "input_artifact_ids": list(self.input_artifact_ids),
            "labels": dict(self.labels),
            "provider_lineage": dict(self.provider_lineage),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> EvalReportArtifact:
        _require_exact_fields(
            data,
            {
                "artifact_id",
                "kind",
                "media_type",
                "producer_run_id",
                "uri",
                "workspace_path",
                "content_digest",
                "input_artifact_ids",
                "labels",
                "provider_lineage",
            },
            "report artifact",
        )
        return cls(
            artifact_id=require_trimmed_string(
                "report artifact ID",
                data["artifact_id"],
            ),
            kind=require_trimmed_string("report artifact kind", data["kind"]),
            media_type=require_trimmed_string(
                "report artifact media type",
                data["media_type"],
            ),
            producer_run_id=require_trimmed_string(
                "report artifact producer run ID",
                data["producer_run_id"],
            ),
            uri=_optional_string("report artifact URI", data["uri"]),
            workspace_path=_optional_string(
                "report artifact workspace path",
                data["workspace_path"],
            ),
            content_digest=_optional_string(
                "report artifact content digest",
                data["content_digest"],
            ),
            input_artifact_ids=_string_tuple(
                "report input artifact IDs",
                data["input_artifact_ids"],
            ),
            labels=_string_map("report artifact labels", data["labels"]),
            provider_lineage=_string_map(
                "report artifact provider lineage",
                data["provider_lineage"],
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalReportGrade:
    """A grade summary with details retained only by explicit policy."""

    grader: GraderDescriptor
    metric: str
    status: GradeStatus
    classification: GradeClassification
    score: float | None = None
    score_range: ScoreRange | None = None
    usage: GraderUsage | None = None
    explanation: str | None = field(default=None, repr=False)
    evidence: tuple[GradeEvidence, ...] = field(default=(), repr=False)
    schema_version: int = GRADE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.grader, GraderDescriptor):
            raise TypeError("report grade grader must be a GraderDescriptor")
        require_trimmed_string("report grade metric", self.metric)
        if not isinstance(self.status, GradeStatus):
            raise TypeError("report grade status must be a GradeStatus")
        if not isinstance(self.classification, GradeClassification):
            raise TypeError(
                "report grade classification must be a GradeClassification"
            )
        if (
            isinstance(self.schema_version, bool)
            or self.schema_version != GRADE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported report grade schema version: {self.schema_version!r}"
            )
        if (self.score is None) != (self.score_range is None):
            raise ValueError(
                "report grade score and score_range must be provided together"
            )
        if self.score is not None:
            score = _finite_number("report grade score", self.score)
            if not isinstance(self.score_range, ScoreRange):
                raise TypeError("report grade score_range must be a ScoreRange")
            if not self.score_range.minimum <= score <= self.score_range.maximum:
                raise ValueError("report grade score must be inside its range")
            object.__setattr__(self, "score", score)
        if self.usage is not None and not isinstance(self.usage, GraderUsage):
            raise TypeError("report grade usage must be GraderUsage or None")
        if self.grader.agentic and self.usage is None:
            raise ValueError("agentic report grades must include usage")
        if self.explanation is not None:
            require_trimmed_string("report grade explanation", self.explanation)
        copied_evidence = tuple(self.evidence)
        if any(not isinstance(item, GradeEvidence) for item in copied_evidence):
            raise TypeError(
                "report grade evidence must contain GradeEvidence values"
            )
        if self.explanation is None and copied_evidence:
            raise ValueError(
                "report grade evidence requires retained grade details"
            )
        object.__setattr__(self, "evidence", copied_evidence)

    @classmethod
    def from_grade(
        cls,
        grade: Grade,
        *,
        retain_details: bool,
    ) -> EvalReportGrade:
        if not isinstance(grade, Grade):
            raise TypeError("report grade source must be a Grade")
        if not isinstance(retain_details, bool):
            raise TypeError("retain_details must be a bool")
        return cls(
            grader=grade.grader,
            metric=grade.metric,
            status=grade.status,
            classification=grade.classification,
            score=grade.score,
            score_range=grade.score_range,
            usage=grade.usage,
            explanation=grade.explanation if retain_details else None,
            evidence=grade.evidence if retain_details else (),
            schema_version=grade.schema_version,
        )

    def to_data(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "grader": _grader_to_data(self.grader),
            "metric": self.metric,
            "status": self.status.value,
            "classification": self.classification.value,
            "score": self.score,
            "score_range": (
                None
                if self.score_range is None
                else {
                    "minimum": self.score_range.minimum,
                    "maximum": self.score_range.maximum,
                }
            ),
            "usage": (
                None
                if self.usage is None
                else {
                    "latency_seconds": self.usage.latency_seconds,
                    "cost": self.usage.cost,
                    "currency": self.usage.currency,
                }
            ),
            "explanation": self.explanation,
            "evidence": [
                cast(JsonValue, item.to_data()) for item in self.evidence
            ],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> EvalReportGrade:
        _require_exact_fields(
            data,
            {
                "schema_version",
                "grader",
                "metric",
                "status",
                "classification",
                "score",
                "score_range",
                "usage",
                "explanation",
                "evidence",
            },
            "report grade",
        )
        score_range_data = data["score_range"]
        score_range = (
            None
            if score_range_data is None
            else _score_range_from_data(
                _require_mapping("report grade score_range", score_range_data)
            )
        )
        usage_data = data["usage"]
        usage = (
            None
            if usage_data is None
            else _usage_from_data(
                _require_mapping("report grade usage", usage_data)
            )
        )
        evidence_data = _require_list("report grade evidence", data["evidence"])
        schema_version = _require_int(
            "report grade schema version",
            data["schema_version"],
        )
        return cls(
            grader=_grader_from_data(
                _require_mapping("report grade grader", data["grader"])
            ),
            metric=require_trimmed_string(
                "report grade metric",
                data["metric"],
            ),
            status=_enum_value(GradeStatus, "report grade status", data["status"]),
            classification=_enum_value(
                GradeClassification,
                "report grade classification",
                data["classification"],
            ),
            score=(
                None
                if data["score"] is None
                else _finite_number("report grade score", data["score"])
            ),
            score_range=score_range,
            usage=usage,
            explanation=_optional_string(
                "report grade explanation",
                data["explanation"],
            ),
            evidence=tuple(
                GradeEvidence.from_data(
                    _require_mapping("report grade evidence", item)
                )
                for item in evidence_data
            ),
            schema_version=schema_version,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalReportCase:
    """One JSON-serializable case result under explicit retention controls."""

    case_id: str
    case_version: str
    run_id: RunId
    status: ExecutionStatus
    failure: Failure | None = None
    output_retained: bool = False
    output: JsonValue = field(default=None, repr=False)
    artifacts: tuple[EvalReportArtifact, ...] = field(default=(), repr=False)
    grades: tuple[EvalReportGrade, ...] = field(default=(), repr=False)
    grader_failures: tuple[EvalGraderFailure, ...] = field(
        default=(),
        repr=False,
    )
    duration_seconds: float

    def __post_init__(self) -> None:
        require_trimmed_string("report case ID", self.case_id)
        require_trimmed_string("report case version", self.case_version)
        if not isinstance(self.run_id, RunId):
            raise TypeError("report case run_id must be a RunId")
        if not isinstance(self.status, ExecutionStatus):
            raise TypeError("report case status must be an ExecutionStatus")
        if self.status is ExecutionStatus.SUCCEEDED:
            if self.failure is not None:
                raise ValueError("successful report case must not have a failure")
        else:
            if not isinstance(self.failure, Failure):
                raise ValueError("non-successful report case requires a failure")
            expected = ExecutionOutcome.from_failure(self.failure).status
            if self.status is not expected:
                raise ValueError(
                    "report case status must match its failure classification"
                )
        if not isinstance(self.output_retained, bool):
            raise TypeError("report case output_retained must be a bool")
        if self.status is not ExecutionStatus.SUCCEEDED and self.output_retained:
            raise ValueError("non-successful report cases cannot retain output")
        if not self.output_retained and self.output is not None:
            raise ValueError("omitted report output must be None")
        object.__setattr__(
            self,
            "output",
            freeze_json_value("eval report output", self.output),
        )

        copied_artifacts = tuple(self.artifacts)
        if any(
            not isinstance(item, EvalReportArtifact) for item in copied_artifacts
        ):
            raise TypeError(
                "report case artifacts must contain EvalReportArtifact values"
            )
        artifact_ids = tuple(item.artifact_id for item in copied_artifacts)
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("report case artifact IDs must be unique")

        copied_grades = tuple(self.grades)
        if any(not isinstance(item, EvalReportGrade) for item in copied_grades):
            raise TypeError(
                "report case grades must contain EvalReportGrade values"
            )
        grade_descriptors = tuple(item.grader for item in copied_grades)
        if len(grade_descriptors) != len(set(grade_descriptors)):
            raise ValueError("report case grade descriptors must be unique")

        copied_failures = tuple(self.grader_failures)
        if any(
            not isinstance(item, EvalGraderFailure) for item in copied_failures
        ):
            raise TypeError(
                "report case grader failures must contain EvalGraderFailure values"
            )
        failure_descriptors = tuple(item.grader for item in copied_failures)
        if len(failure_descriptors) != len(set(failure_descriptors)):
            raise ValueError(
                "report case grader failure descriptors must be unique"
            )
        if set(grade_descriptors).intersection(failure_descriptors):
            raise ValueError("one report grader cannot both succeed and fail")

        object.__setattr__(self, "artifacts", copied_artifacts)
        object.__setattr__(self, "grades", copied_grades)
        object.__setattr__(self, "grader_failures", copied_failures)
        object.__setattr__(
            self,
            "duration_seconds",
            _non_negative_number(
                "report case duration_seconds",
                self.duration_seconds,
            ),
        )

    @classmethod
    def from_case_result(
        cls,
        result: EvalCaseResult[OutputT],
        *,
        retention: EvalReportRetention,
    ) -> EvalReportCase:
        if not isinstance(result, EvalCaseResult):
            raise TypeError("report case source must be an EvalCaseResult")
        if not isinstance(retention, EvalReportRetention):
            raise TypeError("report case retention must be EvalReportRetention")
        retain_output = retention.outputs and result.outcome.is_success
        return cls(
            case_id=result.case_id,
            case_version=result.case_version,
            run_id=result.run_id,
            status=result.outcome.status,
            failure=result.outcome.failure,
            output_retained=retain_output,
            output=(
                cast(JsonValue, result.outcome.output)
                if retain_output
                else None
            ),
            artifacts=(
                tuple(
                    EvalReportArtifact.from_artifact(artifact)
                    for artifact in result.outcome.artifacts
                )
                if retention.artifacts
                else ()
            ),
            grades=tuple(
                EvalReportGrade.from_grade(
                    grade,
                    retain_details=retention.grade_details,
                )
                for grade in result.grades
            ),
            grader_failures=result.grader_failures,
            duration_seconds=result.duration_seconds,
        )

    def to_data(self) -> dict[str, JsonValue]:
        return {
            "case_id": self.case_id,
            "case_version": self.case_version,
            "run_id": self.run_id.value,
            "status": self.status.value,
            "failure": (
                None if self.failure is None else _failure_to_data(self.failure)
            ),
            "output_retained": self.output_retained,
            "output": thaw_json_value(self.output),
            "artifacts": [item.to_data() for item in self.artifacts],
            "grades": [item.to_data() for item in self.grades],
            "grader_failures": [
                {
                    "grader": _grader_to_data(item.grader),
                    "failure": _failure_to_data(item.failure),
                }
                for item in self.grader_failures
            ],
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> EvalReportCase:
        _require_exact_fields(
            data,
            {
                "case_id",
                "case_version",
                "run_id",
                "status",
                "failure",
                "output_retained",
                "output",
                "artifacts",
                "grades",
                "grader_failures",
                "duration_seconds",
            },
            "eval report case",
        )
        failure_data = data["failure"]
        failure = (
            None
            if failure_data is None
            else _failure_from_data(
                _require_mapping("report case failure", failure_data)
            )
        )
        return cls(
            case_id=require_trimmed_string("report case ID", data["case_id"]),
            case_version=require_trimmed_string(
                "report case version",
                data["case_version"],
            ),
            run_id=RunId(require_trimmed_string("report run ID", data["run_id"])),
            status=_enum_value(
                ExecutionStatus,
                "report case status",
                data["status"],
            ),
            failure=failure,
            output_retained=_require_bool(
                "report case output_retained",
                data["output_retained"],
            ),
            output=cast(JsonValue, data["output"]),
            artifacts=tuple(
                EvalReportArtifact.from_data(
                    _require_mapping("report case artifact", item)
                )
                for item in _require_list(
                    "report case artifacts",
                    data["artifacts"],
                )
            ),
            grades=tuple(
                EvalReportGrade.from_data(
                    _require_mapping("report case grade", item)
                )
                for item in _require_list("report case grades", data["grades"])
            ),
            grader_failures=tuple(
                _grader_failure_from_data(
                    _require_mapping("report grader failure", item)
                )
                for item in _require_list(
                    "report case grader failures",
                    data["grader_failures"],
                )
            ),
            duration_seconds=_non_negative_number(
                "report case duration_seconds",
                data["duration_seconds"],
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalReport:
    """A strict, deterministic, and redacted machine-readable eval report."""

    dataset_id: str
    dataset_version: str
    target: EvalTargetDescriptor
    retention: EvalReportRetention
    cases: tuple[EvalReportCase, ...]
    summary: EvalSummary
    environment: Mapping[str, JsonValue] = field(default_factory=dict, repr=False)
    schema_version: int = EVAL_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_trimmed_string("eval report dataset ID", self.dataset_id)
        require_trimmed_string(
            "eval report dataset version",
            self.dataset_version,
        )
        if not isinstance(self.target, EvalTargetDescriptor):
            raise TypeError("eval report target must be an EvalTargetDescriptor")
        if not isinstance(self.retention, EvalReportRetention):
            raise TypeError(
                "eval report retention must be an EvalReportRetention"
            )
        if (
            isinstance(self.schema_version, bool)
            or self.schema_version != EVAL_REPORT_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported eval report schema version: {self.schema_version!r}"
            )
        copied_cases = tuple(self.cases)
        if not copied_cases:
            raise ValueError("eval report requires at least one case")
        if any(not isinstance(case, EvalReportCase) for case in copied_cases):
            raise TypeError("eval report cases must contain EvalReportCase values")
        case_ids = tuple(case.case_id for case in copied_cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("eval report case IDs must be unique")
        if not isinstance(self.summary, EvalSummary):
            raise TypeError("eval report summary must be an EvalSummary")
        if self.summary != _summarize_report_cases(copied_cases):
            raise ValueError("eval report summary must match its case results")
        if not self.retention.outputs and any(
            case.output_retained for case in copied_cases
        ):
            raise ValueError("eval report retained output violates its policy")
        if not self.retention.artifacts and any(
            case.artifacts for case in copied_cases
        ):
            raise ValueError("eval report retained artifacts violate its policy")
        if not self.retention.grade_details and any(
            grade.explanation is not None or grade.evidence
            for case in copied_cases
            for grade in case.grades
        ):
            raise ValueError("eval report retained grade details violate its policy")
        object.__setattr__(self, "cases", copied_cases)
        object.__setattr__(
            self,
            "environment",
            freeze_json_object("eval report environment", self.environment),
        )

    @classmethod
    def from_run(
        cls,
        run: EvalRunResult[OutputT],
        *,
        retention: EvalReportRetention | None = None,
        environment: Mapping[str, JsonValue] | None = None,
        redaction_policy: JsonRedactionPolicy = DEFAULT_REDACTION_POLICY,
    ) -> EvalReport:
        """Build a report, then redact every retained JSON field."""
        if not isinstance(run, EvalRunResult):
            raise TypeError("eval report source must be an EvalRunResult")
        effective_retention = (
            retention if retention is not None else EvalReportRetention()
        )
        if not isinstance(effective_retention, EvalReportRetention):
            raise TypeError("eval report retention must be EvalReportRetention")
        if not callable(getattr(redaction_policy, "redact_json_object", None)):
            raise TypeError(
                "eval report redaction policy must implement redact_json_object"
            )
        unredacted = cls(
            dataset_id=run.dataset_id,
            dataset_version=run.dataset_version,
            target=run.target,
            retention=effective_retention,
            cases=tuple(
                EvalReportCase.from_case_result(
                    case,
                    retention=effective_retention,
                )
                for case in run.cases
            ),
            summary=run.summary,
            environment=environment if environment is not None else {},
        )
        redacted = redaction_policy.redact_json_object(unredacted.to_data())
        return cls.from_data(redacted)

    def to_data(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "dataset": {
                "dataset_id": self.dataset_id,
                "version": self.dataset_version,
            },
            "target": {
                "target_id": self.target.target_id,
                "version": self.target.version,
                "kind": self.target.kind.value,
            },
            "retention": self.retention.to_data(),
            "environment": thaw_json_value(self.environment),
            "cases": [case.to_data() for case in self.cases],
            "summary": _summary_to_data(self.summary),
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
    def from_data(cls, data: Mapping[str, object]) -> EvalReport:
        _require_exact_fields(
            data,
            {
                "schema_version",
                "dataset",
                "target",
                "retention",
                "environment",
                "cases",
                "summary",
            },
            "eval report",
        )
        dataset = _require_mapping("eval report dataset", data["dataset"])
        _require_exact_fields(
            dataset,
            {"dataset_id", "version"},
            "eval report dataset",
        )
        target = _require_mapping("eval report target", data["target"])
        _require_exact_fields(
            target,
            {"target_id", "version", "kind"},
            "eval report target",
        )
        return cls(
            dataset_id=require_trimmed_string(
                "eval report dataset ID",
                dataset["dataset_id"],
            ),
            dataset_version=require_trimmed_string(
                "eval report dataset version",
                dataset["version"],
            ),
            target=EvalTargetDescriptor(
                target_id=require_trimmed_string(
                    "eval report target ID",
                    target["target_id"],
                ),
                version=require_trimmed_string(
                    "eval report target version",
                    target["version"],
                ),
                kind=_enum_value(
                    EvalTargetKind,
                    "eval report target kind",
                    target["kind"],
                ),
            ),
            retention=EvalReportRetention.from_data(
                _require_mapping(
                    "eval report retention",
                    data["retention"],
                )
            ),
            cases=tuple(
                EvalReportCase.from_data(
                    _require_mapping("eval report case", item)
                )
                for item in _require_list("eval report cases", data["cases"])
            ),
            summary=_summary_from_data(
                _require_mapping("eval report summary", data["summary"])
            ),
            environment=cast(
                Mapping[str, JsonValue],
                _require_mapping("eval report environment", data["environment"]),
            ),
            schema_version=_require_int(
                "eval report schema version",
                data["schema_version"],
            ),
        )

    @classmethod
    def from_json(cls, serialized: str) -> EvalReport:
        decoded = json.loads(serialized)
        if not isinstance(decoded, dict):
            raise ValueError("serialized eval report must contain a JSON object")
        return cls.from_data(decoded)


def _summarize_report_cases(cases: tuple[EvalReportCase, ...]) -> EvalSummary:
    outcome_counts = {status: 0 for status in ExecutionStatus}
    grade_counts = {status: 0 for status in GradeStatus}
    grader_failures = 0
    costs: dict[str, list[float]] = defaultdict(list)
    latencies: list[float] = []
    durations: list[float] = []
    for case in cases:
        outcome_counts[case.status] += 1
        grader_failures += len(case.grader_failures)
        durations.append(case.duration_seconds)
        for grade in case.grades:
            grade_counts[grade.status] += 1
            if grade.usage is not None:
                latencies.append(grade.usage.latency_seconds)
                costs[grade.usage.currency].append(grade.usage.cost)
    return EvalSummary(
        case_count=len(cases),
        succeeded_cases=outcome_counts[ExecutionStatus.SUCCEEDED],
        failed_cases=outcome_counts[ExecutionStatus.FAILED],
        blocked_cases=outcome_counts[ExecutionStatus.BLOCKED],
        cancelled_cases=outcome_counts[ExecutionStatus.CANCELLED],
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


def _summary_to_data(summary: EvalSummary) -> dict[str, JsonValue]:
    return {
        "case_count": summary.case_count,
        "succeeded_cases": summary.succeeded_cases,
        "failed_cases": summary.failed_cases,
        "blocked_cases": summary.blocked_cases,
        "cancelled_cases": summary.cancelled_cases,
        "passing_grades": summary.passing_grades,
        "warning_grades": summary.warning_grades,
        "failing_grades": summary.failing_grades,
        "grader_failure_count": summary.grader_failure_count,
        "duration_seconds": summary.duration_seconds,
        "grader_latency_seconds": summary.grader_latency_seconds,
        "grader_costs": [
            {"currency": cost.currency, "amount": cost.amount}
            for cost in summary.grader_costs
        ],
    }


def _summary_from_data(data: Mapping[str, object]) -> EvalSummary:
    fields = {
        "case_count",
        "succeeded_cases",
        "failed_cases",
        "blocked_cases",
        "cancelled_cases",
        "passing_grades",
        "warning_grades",
        "failing_grades",
        "grader_failure_count",
        "duration_seconds",
        "grader_latency_seconds",
        "grader_costs",
    }
    _require_exact_fields(data, fields, "eval report summary")
    return EvalSummary(
        case_count=_require_int("summary case_count", data["case_count"]),
        succeeded_cases=_require_int(
            "summary succeeded_cases",
            data["succeeded_cases"],
        ),
        failed_cases=_require_int("summary failed_cases", data["failed_cases"]),
        blocked_cases=_require_int(
            "summary blocked_cases",
            data["blocked_cases"],
        ),
        cancelled_cases=_require_int(
            "summary cancelled_cases",
            data["cancelled_cases"],
        ),
        passing_grades=_require_int(
            "summary passing_grades",
            data["passing_grades"],
        ),
        warning_grades=_require_int(
            "summary warning_grades",
            data["warning_grades"],
        ),
        failing_grades=_require_int(
            "summary failing_grades",
            data["failing_grades"],
        ),
        grader_failure_count=_require_int(
            "summary grader_failure_count",
            data["grader_failure_count"],
        ),
        duration_seconds=_non_negative_number(
            "summary duration_seconds",
            data["duration_seconds"],
        ),
        grader_latency_seconds=_non_negative_number(
            "summary grader_latency_seconds",
            data["grader_latency_seconds"],
        ),
        grader_costs=tuple(
            _cost_from_data(_require_mapping("summary grader cost", item))
            for item in _require_list(
                "summary grader costs",
                data["grader_costs"],
            )
        ),
    )


def _cost_from_data(data: Mapping[str, object]) -> EvalCost:
    _require_exact_fields(data, {"currency", "amount"}, "eval report cost")
    return EvalCost(
        currency=require_trimmed_string("eval cost currency", data["currency"]),
        amount=_non_negative_number("eval cost amount", data["amount"]),
    )


def _grader_to_data(descriptor: GraderDescriptor) -> dict[str, JsonValue]:
    return {
        "grader_id": descriptor.grader_id,
        "version": descriptor.version,
        "agentic": descriptor.agentic,
    }


def _grader_from_data(data: Mapping[str, object]) -> GraderDescriptor:
    _require_exact_fields(
        data,
        {"grader_id", "version", "agentic"},
        "report grader",
    )
    return GraderDescriptor(
        grader_id=require_trimmed_string("report grader ID", data["grader_id"]),
        version=require_trimmed_string(
            "report grader version",
            data["version"],
        ),
        agentic=_require_bool("report grader agentic", data["agentic"]),
    )


def _failure_to_data(failure: Failure) -> dict[str, JsonValue]:
    return {
        "kind": failure.kind.value,
        "message": failure.message,
        "code": failure.code,
        "metadata": dict(failure.metadata),
    }


def _failure_from_data(data: Mapping[str, object]) -> Failure:
    _require_exact_fields(
        data,
        {"kind", "message", "code", "metadata"},
        "report failure",
    )
    return Failure(
        kind=_enum_value(FailureKind, "report failure kind", data["kind"]),
        message=require_trimmed_string(
            "report failure message",
            data["message"],
        ),
        code=_optional_string("report failure code", data["code"]),
        metadata=_string_map("report failure metadata", data["metadata"]),
    )


def _grader_failure_from_data(
    data: Mapping[str, object],
) -> EvalGraderFailure:
    _require_exact_fields(
        data,
        {"grader", "failure"},
        "report grader failure",
    )
    return EvalGraderFailure(
        grader=_grader_from_data(
            _require_mapping("report failed grader", data["grader"])
        ),
        failure=_failure_from_data(
            _require_mapping("report grader failure", data["failure"])
        ),
    )


def _score_range_from_data(data: Mapping[str, object]) -> ScoreRange:
    _require_exact_fields(data, {"minimum", "maximum"}, "report score range")
    return ScoreRange(
        minimum=_finite_number("report score minimum", data["minimum"]),
        maximum=_finite_number("report score maximum", data["maximum"]),
    )


def _usage_from_data(data: Mapping[str, object]) -> GraderUsage:
    _require_exact_fields(
        data,
        {"latency_seconds", "cost", "currency"},
        "report grader usage",
    )
    return GraderUsage(
        latency_seconds=_non_negative_number(
            "report grader latency",
            data["latency_seconds"],
        ),
        cost=_non_negative_number("report grader cost", data["cost"]),
        currency=require_trimmed_string(
            "report grader currency",
            data["currency"],
        ),
    )


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


def _require_list(field_name: str, value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    return value


def _require_bool(field_name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a bool")
    return value


def _require_int(field_name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


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


def _optional_string(field_name: str, value: object) -> str | None:
    if value is None:
        return None
    return require_trimmed_string(field_name, value)


def _string_tuple(field_name: str, value: object) -> tuple[str, ...]:
    return tuple(
        require_trimmed_string(field_name, item)
        for item in _require_list(field_name, value)
    )


def _string_map(field_name: str, value: object) -> Mapping[str, str]:
    mapping = _require_mapping(field_name, value)
    copied: dict[str, str] = {}
    for key, item in mapping.items():
        copied[require_trimmed_string(f"{field_name} key", key)] = (
            require_trimmed_string(f"{field_name} value", item)
        )
    return copied


def _enum_value(
    enum_type: type[EnumT],
    field_name: str,
    value: object,
) -> EnumT:
    string_value = require_trimmed_string(field_name, value)
    try:
        return enum_type(string_value)
    except ValueError as error:
        raise ValueError(f"unknown {field_name}: {string_value!r}") from error
