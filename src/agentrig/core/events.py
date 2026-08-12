"""Versioned, provider-independent execution events."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import TypeAlias, cast

from agentrig.core._validation import freeze_string_map, require_trimmed_string
from agentrig.core.context import RunContext
from agentrig.core.identity import RunId

EVENT_SCHEMA_VERSION = 1

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = (
    JsonScalar
    | Mapping[str, "JsonValue"]
    | list["JsonValue"]
    | tuple["JsonValue", ...]
)


class EventKind(StrEnum):
    """Stable vocabulary for execution lifecycle and observability."""

    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_BLOCKED = "run.blocked"
    RUN_CANCELLED = "run.cancelled"
    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"
    PROGRESS_REPORTED = "progress.reported"
    PROVIDER_CALL_STARTED = "provider_call.started"
    PROVIDER_CALL_COMPLETED = "provider_call.completed"
    TOOL_CALL_STARTED = "tool_call.started"
    TOOL_CALL_COMPLETED = "tool_call.completed"
    ARTIFACT_PRODUCED = "artifact.produced"
    GRADE_PRODUCED = "grade.produced"
    RETRY_SCHEDULED = "retry.scheduled"
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_RESOLVED = "approval.resolved"
    USAGE_REPORTED = "usage.reported"


@dataclass(frozen=True, order=True, slots=True)
class EventId:
    """Opaque, serializable identity for one event."""

    value: str

    def __post_init__(self) -> None:
        require_trimmed_string("event ID", self.value)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """Immutable event envelope with JSON-safe attributes."""

    event_id: EventId
    kind: EventKind
    occurred_at: datetime
    run_id: RunId
    parent_run_id: RunId | None = None
    correlation: Mapping[str, str] = field(default_factory=dict)
    attributes: Mapping[str, JsonValue] = field(default_factory=dict)
    schema_version: int = EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, EventId):
            raise TypeError("event_id must be an EventId")
        if not isinstance(self.kind, EventKind):
            raise TypeError("kind must be an EventKind")
        if not isinstance(self.run_id, RunId):
            raise TypeError("run_id must be a RunId")
        if self.parent_run_id is not None and not isinstance(
            self.parent_run_id,
            RunId,
        ):
            raise TypeError("parent_run_id must be a RunId or None")
        if not isinstance(self.occurred_at, datetime):
            raise TypeError("occurred_at must be a datetime")
        if (
            self.occurred_at.tzinfo is None
            or self.occurred_at.utcoffset() is None
        ):
            raise ValueError("event timestamp must be timezone-aware")
        if (
            isinstance(self.schema_version, bool)
            or self.schema_version != EVENT_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported event schema version: {self.schema_version!r}"
            )

        object.__setattr__(
            self,
            "occurred_at",
            self.occurred_at.astimezone(UTC),
        )
        object.__setattr__(
            self,
            "correlation",
            freeze_string_map("event correlation", self.correlation),
        )
        object.__setattr__(
            self,
            "attributes",
            _freeze_json_object(self.attributes),
        )

    @classmethod
    def from_context(
        cls,
        *,
        event_id: EventId,
        kind: EventKind,
        context: RunContext,
        attributes: Mapping[str, JsonValue] | None = None,
    ) -> Event:
        """Create an event using the context's injected clock and lineage."""
        return cls(
            event_id=event_id,
            kind=kind,
            occurred_at=context.clock.now(),
            run_id=context.run_id,
            parent_run_id=context.parent_run_id,
            correlation=context.correlation,
            attributes=attributes if attributes is not None else {},
        )

    def to_data(self) -> dict[str, JsonValue]:
        """Return a detached JSON-compatible representation."""
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id.value,
            "kind": self.kind.value,
            "occurred_at": self.occurred_at.isoformat().replace("+00:00", "Z"),
            "run_id": self.run_id.value,
            "parent_run_id": (
                self.parent_run_id.value
                if self.parent_run_id is not None
                else None
            ),
            "correlation": dict(self.correlation),
            "attributes": _thaw_json_value(self.attributes),
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
    def from_data(cls, data: Mapping[str, object]) -> Event:
        """Validate and restore an event from decoded JSON data."""
        expected_fields = {
            "schema_version",
            "event_id",
            "kind",
            "occurred_at",
            "run_id",
            "parent_run_id",
            "correlation",
            "attributes",
        }
        actual_fields = set(data)
        if actual_fields != expected_fields:
            missing = sorted(expected_fields - actual_fields)
            unknown = sorted(actual_fields - expected_fields)
            raise ValueError(
                f"invalid event fields; missing={missing!r}, unknown={unknown!r}"
            )

        schema_version = data["schema_version"]
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ValueError("event schema version must be an integer")

        kind_value = require_trimmed_string("event kind", data["kind"])
        try:
            kind = EventKind(kind_value)
        except ValueError as error:
            raise ValueError(f"unknown event kind: {kind_value!r}") from error

        occurred_at_value = require_trimmed_string(
            "event timestamp",
            data["occurred_at"],
        )
        try:
            occurred_at = datetime.fromisoformat(occurred_at_value)
        except ValueError as error:
            raise ValueError("event timestamp must use ISO 8601") from error

        parent_run_id_value = data["parent_run_id"]
        parent_run_id = (
            None
            if parent_run_id_value is None
            else RunId(
                require_trimmed_string("parent run ID", parent_run_id_value)
            )
        )
        correlation = _require_object("event correlation", data["correlation"])
        attributes = _require_object("event attributes", data["attributes"])

        return cls(
            event_id=EventId(
                require_trimmed_string("event ID", data["event_id"])
            ),
            kind=kind,
            occurred_at=occurred_at,
            run_id=RunId(require_trimmed_string("run ID", data["run_id"])),
            parent_run_id=parent_run_id,
            correlation=cast(Mapping[str, str], correlation),
            attributes=cast(Mapping[str, JsonValue], attributes),
            schema_version=schema_version,
        )

    @classmethod
    def from_json(cls, serialized: str) -> Event:
        """Decode and validate an event from JSON."""
        decoded = json.loads(serialized)
        if not isinstance(decoded, dict):
            raise ValueError("serialized event must contain a JSON object")
        return cls.from_data(decoded)


def _freeze_json_object(
    value: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    frozen = _freeze_json_value(value, path="attributes", active_ids=set())
    return cast(Mapping[str, JsonValue], frozen)


def _freeze_json_value(
    value: JsonValue,
    *,
    path: str,
    active_ids: set[int],
) -> JsonValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        return _freeze_json_mapping(value, path=path, active_ids=active_ids)
    if isinstance(value, (list, tuple)):
        return _freeze_json_sequence(value, path=path, active_ids=active_ids)
    raise ValueError(f"{path} contains a non-JSON value: {type(value).__name__}")


def _freeze_json_mapping(
    value: Mapping[str, JsonValue],
    *,
    path: str,
    active_ids: set[int],
) -> Mapping[str, JsonValue]:
    identity = id(value)
    if identity in active_ids:
        raise ValueError(f"{path} contains a reference cycle")
    active_ids.add(identity)
    try:
        frozen: dict[str, JsonValue] = {}
        for key, child in value.items():
            validated_key = require_trimmed_string(f"{path} key", key)
            frozen[validated_key] = _freeze_json_value(
                child,
                path=f"{path}.{validated_key}",
                active_ids=active_ids,
            )
        return MappingProxyType(frozen)
    finally:
        active_ids.remove(identity)


def _freeze_json_sequence(
    value: list[JsonValue] | tuple[JsonValue, ...],
    *,
    path: str,
    active_ids: set[int],
) -> tuple[JsonValue, ...]:
    identity = id(value)
    if identity in active_ids:
        raise ValueError(f"{path} contains a reference cycle")
    active_ids.add(identity)
    try:
        return tuple(
            _freeze_json_value(
                child,
                path=f"{path}[{index}]",
                active_ids=active_ids,
            )
            for index, child in enumerate(value)
        )
    finally:
        active_ids.remove(identity)


def _thaw_json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json_value(child) for child in value]
    return value


def _require_object(field_name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    return cast(Mapping[str, object], value)
