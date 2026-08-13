"""Portable grading records and the provider-independent grader contract."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, TypeVar, cast, runtime_checkable

from agentrig.core._validation import require_trimmed_string
from agentrig.core.artifacts import ArtifactId
from agentrig.core.context import RunContext
from agentrig.core.observability import (
    NOOP_EVENT_SINK,
    EventSink,
)

GRADE_SCHEMA_VERSION = 1

_CURRENCY_PATTERN = re.compile(r"[A-Z]{3}")

SubjectT = TypeVar("SubjectT", contravariant=True)
FieldPathComponent = str | int


class GradeStatus(StrEnum):
    """Stable subject-evaluation result produced by a grader."""

    PASS = "pass"
    WARNING = "warning"
    FAILURE = "failure"


class GradeClassification(StrEnum):
    """Whether a grade is a mandatory constraint or advisory signal."""

    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True, slots=True, kw_only=True)
class GraderDescriptor:
    """Stable identity and execution characteristics for a grader."""

    grader_id: str
    version: str
    agentic: bool = False

    def __post_init__(self) -> None:
        require_trimmed_string("grader ID", self.grader_id)
        require_trimmed_string("grader version", self.version)
        if not isinstance(self.agentic, bool):
            raise TypeError("grader agentic flag must be a bool")


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoreRange:
    """Inclusive calibrated range giving a numeric grade meaning."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        minimum = _require_finite_number("score range minimum", self.minimum)
        maximum = _require_finite_number("score range maximum", self.maximum)
        if minimum >= maximum:
            raise ValueError("score range minimum must be less than maximum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)


@dataclass(frozen=True, slots=True, kw_only=True)
class GradeEvidence:
    """Reference an artifact or structured subject field, never hidden reasoning."""

    artifact_id: ArtifactId | None = None
    field_path: tuple[FieldPathComponent, ...] | None = None

    def __post_init__(self) -> None:
        if (self.artifact_id is None) == (self.field_path is None):
            raise ValueError(
                "grade evidence must reference exactly one artifact or field"
            )
        if self.artifact_id is not None:
            if not isinstance(self.artifact_id, ArtifactId):
                raise TypeError("evidence artifact_id must be an ArtifactId")
            return

        if self.field_path is None:
            raise AssertionError("validated field evidence has no path")
        copied_path = tuple(self.field_path)
        if not copied_path:
            raise ValueError("evidence field path must not be empty")
        for component in copied_path:
            if isinstance(component, bool) or not isinstance(
                component,
                (str, int),
            ):
                raise TypeError(
                    "evidence field path components must be strings or integers"
                )
            if isinstance(component, str):
                require_trimmed_string("evidence field path component", component)
            elif component < 0:
                raise ValueError(
                    "evidence field path integer components must be non-negative"
                )
        object.__setattr__(self, "field_path", copied_path)

    def to_data(self) -> dict[str, object]:
        """Return the stable wire representation for this reference."""
        return {
            "artifact_id": (
                self.artifact_id.value if self.artifact_id is not None else None
            ),
            "field_path": (
                list(self.field_path) if self.field_path is not None else None
            ),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> GradeEvidence:
        """Validate and restore an evidence reference."""
        _require_exact_fields(data, {"artifact_id", "field_path"}, "evidence")
        artifact_id_value = data["artifact_id"]
        field_path_value = data["field_path"]
        artifact_id = (
            None
            if artifact_id_value is None
            else ArtifactId(require_trimmed_string("artifact ID", artifact_id_value))
        )
        if field_path_value is None:
            field_path = None
        elif isinstance(field_path_value, list):
            field_path = tuple(cast(list[FieldPathComponent], field_path_value))
        else:
            raise ValueError("evidence field_path must be an array or null")
        return cls(artifact_id=artifact_id, field_path=field_path)


@dataclass(frozen=True, slots=True, kw_only=True)
class GraderUsage:
    """Cost and latency attributable to one grading operation."""

    latency_seconds: float
    cost: float
    currency: str = "USD"

    def __post_init__(self) -> None:
        latency = _require_finite_number(
            "grader latency_seconds",
            self.latency_seconds,
        )
        cost = _require_finite_number("grader cost", self.cost)
        if latency < 0:
            raise ValueError("grader latency_seconds must be non-negative")
        if cost < 0:
            raise ValueError("grader cost must be non-negative")
        require_trimmed_string("grader cost currency", self.currency)
        if _CURRENCY_PATTERN.fullmatch(self.currency) is None:
            raise ValueError("grader cost currency must be a three-letter code")
        object.__setattr__(self, "latency_seconds", latency)
        object.__setattr__(self, "cost", cost)


@dataclass(frozen=True, slots=True, kw_only=True)
class Grade:
    """Immutable, versioned evaluation of a subject or one of its artifacts."""

    grader: GraderDescriptor
    metric: str
    status: GradeStatus
    classification: GradeClassification
    explanation: str
    evidence: tuple[GradeEvidence, ...] = ()
    score: float | None = None
    score_range: ScoreRange | None = None
    usage: GraderUsage | None = None
    schema_version: int = GRADE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.grader, GraderDescriptor):
            raise TypeError("grade grader must be a GraderDescriptor")
        if not isinstance(self.status, GradeStatus):
            raise TypeError("grade status must be a GradeStatus")
        if not isinstance(self.classification, GradeClassification):
            raise TypeError(
                "grade classification must be a GradeClassification"
            )
        require_trimmed_string("grade metric", self.metric)
        require_trimmed_string("grade explanation", self.explanation)
        if (
            isinstance(self.schema_version, bool)
            or self.schema_version != GRADE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported grade schema version: {self.schema_version!r}"
            )

        copied_evidence = tuple(self.evidence)
        if any(not isinstance(item, GradeEvidence) for item in copied_evidence):
            raise TypeError("grade evidence must contain GradeEvidence values")
        object.__setattr__(self, "evidence", copied_evidence)

        if (self.score is None) != (self.score_range is None):
            raise ValueError("grade score and score_range must be provided together")
        if self.score is not None:
            score = _require_finite_number("grade score", self.score)
            if not isinstance(self.score_range, ScoreRange):
                raise TypeError("grade score_range must be a ScoreRange")
            if not self.score_range.minimum <= score <= self.score_range.maximum:
                raise ValueError("grade score must be inside its calibrated range")
            object.__setattr__(self, "score", score)

        if self.usage is not None and not isinstance(self.usage, GraderUsage):
            raise TypeError("grade usage must be GraderUsage or None")
        if self.grader.agentic and self.usage is None:
            raise ValueError("agentic grades must include cost and latency usage")

    def to_data(self) -> dict[str, object]:
        """Return a detached JSON-compatible representation."""
        return {
            "schema_version": self.schema_version,
            "grader": {
                "grader_id": self.grader.grader_id,
                "version": self.grader.version,
                "agentic": self.grader.agentic,
            },
            "metric": self.metric,
            "status": self.status.value,
            "classification": self.classification.value,
            "explanation": self.explanation,
            "evidence": [item.to_data() for item in self.evidence],
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
        }

    def to_json(self) -> str:
        """Serialize using a stable compact JSON representation."""
        return json.dumps(
            self.to_data(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> Grade:
        """Validate and restore a grade from decoded JSON data."""
        _require_exact_fields(
            data,
            {
                "schema_version",
                "grader",
                "metric",
                "status",
                "classification",
                "explanation",
                "evidence",
                "score",
                "score_range",
                "usage",
            },
            "grade",
        )
        grader_data = _require_mapping("grade grader", data["grader"])
        _require_exact_fields(
            grader_data,
            {"grader_id", "version", "agentic"},
            "grader",
        )
        agentic = grader_data["agentic"]
        if not isinstance(agentic, bool):
            raise ValueError("grader agentic flag must be a bool")
        grader = GraderDescriptor(
            grader_id=require_trimmed_string(
                "grader ID",
                grader_data["grader_id"],
            ),
            version=require_trimmed_string(
                "grader version",
                grader_data["version"],
            ),
            agentic=agentic,
        )

        evidence_data = data["evidence"]
        if not isinstance(evidence_data, list):
            raise ValueError("grade evidence must be an array")
        evidence = tuple(
            GradeEvidence.from_data(_require_mapping("grade evidence", item))
            for item in evidence_data
        )

        score_range_data = data["score_range"]
        if score_range_data is None:
            score_range = None
        else:
            score_range_mapping = _require_mapping(
                "grade score_range",
                score_range_data,
            )
            _require_exact_fields(
                score_range_mapping,
                {"minimum", "maximum"},
                "score_range",
            )
            score_range = ScoreRange(
                minimum=cast(float, score_range_mapping["minimum"]),
                maximum=cast(float, score_range_mapping["maximum"]),
            )

        usage_data = data["usage"]
        if usage_data is None:
            usage = None
        else:
            usage_mapping = _require_mapping("grade usage", usage_data)
            _require_exact_fields(
                usage_mapping,
                {"latency_seconds", "cost", "currency"},
                "usage",
            )
            usage = GraderUsage(
                latency_seconds=cast(float, usage_mapping["latency_seconds"]),
                cost=cast(float, usage_mapping["cost"]),
                currency=require_trimmed_string(
                    "grader cost currency",
                    usage_mapping["currency"],
                ),
            )

        schema_version = data["schema_version"]
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ValueError("grade schema version must be an integer")
        score_data = data["score"]
        score = None if score_data is None else cast(float, score_data)
        try:
            status = GradeStatus(
                require_trimmed_string("grade status", data["status"])
            )
        except ValueError as error:
            raise ValueError(f"unknown grade status: {data['status']!r}") from error
        try:
            classification = GradeClassification(
                require_trimmed_string(
                    "grade classification",
                    data["classification"],
                )
            )
        except ValueError as error:
            raise ValueError(
                f"unknown grade classification: {data['classification']!r}"
            ) from error

        return cls(
            grader=grader,
            metric=require_trimmed_string("grade metric", data["metric"]),
            status=status,
            classification=classification,
            explanation=require_trimmed_string(
                "grade explanation",
                data["explanation"],
            ),
            evidence=evidence,
            score=score,
            score_range=score_range,
            usage=usage,
            schema_version=schema_version,
        )

    @classmethod
    def from_json(cls, serialized: str) -> Grade:
        """Decode and validate a grade from JSON."""
        decoded = json.loads(serialized)
        if not isinstance(decoded, dict):
            raise ValueError("serialized grade must contain a JSON object")
        return cls.from_data(decoded)


@dataclass(frozen=True, slots=True, kw_only=True)
class GradingContext:
    """Execution dependencies available to a grader invocation."""

    run_context: RunContext
    event_sink: EventSink = field(default=NOOP_EVENT_SINK)

    def __post_init__(self) -> None:
        if not isinstance(self.run_context, RunContext):
            raise TypeError("grading run_context must be a RunContext")
        if not callable(getattr(self.event_sink, "emit", None)):
            raise TypeError("grading event_sink must implement emit(event)")


@runtime_checkable
class Grader(Protocol[SubjectT]):
    """Evaluate a subject without deciding workflow control flow."""

    @property
    def descriptor(self) -> GraderDescriptor:
        """Return the stable identity of this grader."""
        ...

    async def grade(
        self,
        subject: SubjectT,
        context: GradingContext,
    ) -> Grade:
        """Return a grade or raise a normalized grader failure."""
        ...


def _require_finite_number(field_name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError(f"{field_name} must be a finite number") from error
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be a finite number")
    return converted


def _require_mapping(field_name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return cast(Mapping[str, object], value)


def _require_exact_fields(
    data: Mapping[str, object],
    expected: set[str],
    record_name: str,
) -> None:
    actual = set(data)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"invalid {record_name} fields; "
            f"missing={missing!r}, unknown={unknown!r}"
        )
