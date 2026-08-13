from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone

from agentrig.core import (
    EVENT_SCHEMA_VERSION,
    CancellationSource,
    Event,
    EventId,
    EventKind,
    RunContext,
    RunId,
)


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 12, 19, 30, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


@dataclass
class SequentialRunIdGenerator:
    next_value: int = 1

    def generate(self) -> RunId:
        run_id = RunId(f"run-{self.next_value}")
        self.next_value += 1
        return run_id


def create_context() -> RunContext:
    source = CancellationSource()
    generator = SequentialRunIdGenerator()
    root = RunContext.create_root(
        clock=FixedClock(),
        id_generator=generator,
        cancellation=source.token,
        correlation={"request_id": "request-1"},
    )
    return root.derive_child(correlation={"step_id": "step-1"})


def create_event(**overrides: object) -> Event:
    values: dict[str, object] = {
        "event_id": EventId("event-1"),
        "kind": EventKind.STEP_STARTED,
        "occurred_at": datetime(2026, 8, 12, 19, 30, tzinfo=UTC),
        "run_id": RunId("run-2"),
        "parent_run_id": RunId("run-1"),
        "correlation": {"request_id": "request-1"},
        "attributes": {"step": "draft", "attempt": 1},
    }
    values.update(overrides)
    return Event(**values)  # type: ignore[arg-type]


class EventKindTest(unittest.TestCase):
    def test_vocabulary_has_stable_wire_values(self) -> None:
        self.assertEqual(
            {kind.value for kind in EventKind},
            {
                "approval.requested",
                "approval.resolved",
                "artifact.produced",
                "grade.produced",
                "grade_policy.decided",
                "progress.reported",
                "provider_call.completed",
                "provider_call.started",
                "retry.scheduled",
                "run.blocked",
                "run.cancelled",
                "run.completed",
                "run.failed",
                "run.started",
                "step.completed",
                "step.started",
                "tool_call.completed",
                "tool_call.started",
                "usage.reported",
            },
        )


class EventTest(unittest.TestCase):
    def test_from_context_uses_injected_time_lineage_and_correlation(self) -> None:
        context = create_context()

        event = Event.from_context(
            event_id=EventId("event-1"),
            kind=EventKind.STEP_STARTED,
            context=context,
            attributes={"step": "draft"},
        )

        self.assertEqual(event.occurred_at, FixedClock().now())
        self.assertEqual(event.run_id, context.run_id)
        self.assertEqual(event.parent_run_id, context.parent_run_id)
        self.assertEqual(
            event.correlation,
            {"request_id": "request-1", "step_id": "step-1"},
        )

    def test_attributes_are_recursively_copied_and_frozen(self) -> None:
        attributes = {
            "usage": {"input_tokens": 10},
            "messages": ["started", {"percent": 0.5}],
        }

        event = create_event(attributes=attributes)
        attributes["usage"]["input_tokens"] = 99  # type: ignore[index]
        attributes["messages"].append("mutated")  # type: ignore[union-attr]

        self.assertEqual(event.attributes["usage"], {"input_tokens": 10})
        self.assertEqual(
            event.attributes["messages"],
            ("started", {"percent": 0.5}),
        )
        with self.assertRaises(TypeError):
            event.attributes["new"] = "value"  # type: ignore[index]

    def test_timestamp_is_normalized_to_utc(self) -> None:
        eastern = timezone(timedelta(hours=-4))
        event = create_event(
            occurred_at=datetime(2026, 8, 12, 15, 30, tzinfo=eastern)
        )

        self.assertEqual(
            event.occurred_at,
            datetime(2026, 8, 12, 19, 30, tzinfo=UTC),
        )

    def test_json_serialization_round_trips(self) -> None:
        event = create_event(
            attributes={
                "message": "Olá",
                "values": [True, None, 3, 2.5],
            }
        )

        serialized = event.to_json()
        restored = Event.from_json(serialized)

        self.assertEqual(restored, event)
        self.assertIn('"schema_version":1', serialized)
        self.assertIn('"occurred_at":"2026-08-12T19:30:00Z"', serialized)

    def test_to_data_returns_detached_mutable_containers(self) -> None:
        event = create_event(attributes={"nested": {"value": 1}})

        data = event.to_data()
        attributes = data["attributes"]
        self.assertIsInstance(attributes, dict)
        attributes["nested"]["value"] = 2  # type: ignore[index]

        self.assertEqual(event.attributes["nested"], {"value": 1})

    def test_rejects_non_json_values_numbers_and_cycles(self) -> None:
        cyclic: list[object] = []
        cyclic.append(cyclic)
        invalid_values = (
            object(),
            math.nan,
            math.inf,
            cyclic,
        )

        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaises(ValueError):
                    create_event(attributes={"invalid": invalid_value})

    def test_rejects_invalid_timestamp_and_schema_version(self) -> None:
        with self.assertRaises(ValueError):
            create_event(occurred_at=datetime(2026, 8, 12, 19, 30))
        for invalid_version in (0, 2, True):
            with self.subTest(invalid_version=invalid_version):
                with self.assertRaises(ValueError):
                    create_event(schema_version=invalid_version)

    def test_from_data_rejects_unknown_fields_and_event_kinds(self) -> None:
        data = create_event().to_data()
        data["unexpected"] = True
        with self.assertRaises(ValueError):
            Event.from_data(data)

        data = create_event().to_data()
        data["kind"] = "vendor.unknown"
        with self.assertRaises(ValueError):
            Event.from_data(data)

    def test_event_id_requires_a_trimmed_nonempty_value(self) -> None:
        for value in ("", " event-1", "event-1 "):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    EventId(value)

    def test_schema_version_constant_matches_events(self) -> None:
        self.assertEqual(create_event().schema_version, EVENT_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
