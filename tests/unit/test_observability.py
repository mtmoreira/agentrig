from __future__ import annotations

import unittest
from datetime import UTC, datetime

from agentrig.core import (
    REDACTED_VALUE,
    CompositeEventSink,
    Event,
    EventId,
    EventKind,
    EventSink,
    InMemoryEventSink,
    NOOP_EVENT_SINK,
    NoOpRedactionPolicy,
    RedactingEventSink,
    RunId,
)


def create_event(
    event_id: str = "event-1",
    *,
    attributes: dict[str, object] | None = None,
) -> Event:
    return Event(
        event_id=EventId(event_id),
        kind=EventKind.PROGRESS_REPORTED,
        occurred_at=datetime(2026, 8, 12, 20, 30, tzinfo=UTC),
        run_id=RunId("run-1"),
        attributes=attributes or {},  # type: ignore[arg-type]
    )


class RecordingSink:
    def __init__(self, name: str, calls: list[str] | None = None) -> None:
        self.name = name
        self.calls = calls if calls is not None else []
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.calls.append(self.name)
        self.events.append(event)


class FailingSink:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    def emit(self, event: Event) -> None:
        del event
        self.calls.append(self.name)
        raise RuntimeError(f"{self.name} failed")


def emit_through_protocol(sink: EventSink, event: Event) -> None:
    sink.emit(event)


class EventSinkTest(unittest.TestCase):
    def test_noop_sink_satisfies_protocol(self) -> None:
        emit_through_protocol(NOOP_EVENT_SINK, create_event())

    def test_in_memory_sink_redacts_by_default(self) -> None:
        sink = InMemoryEventSink()
        event = create_event(
            attributes={"password": "private", "input_tokens": 10}
        )

        sink.emit(event)

        self.assertEqual(sink.events[0].attributes["password"], REDACTED_VALUE)
        self.assertEqual(sink.events[0].attributes["input_tokens"], 10)
        self.assertEqual(event.attributes["password"], "private")

    def test_in_memory_snapshots_are_stable_and_clearable(self) -> None:
        sink = InMemoryEventSink(redaction_policy=NoOpRedactionPolicy())
        sink.emit(create_event("event-1"))
        snapshot = sink.events

        sink.emit(create_event("event-2"))

        self.assertEqual(
            tuple(event.event_id for event in snapshot),
            (EventId("event-1"),),
        )
        self.assertEqual(
            tuple(event.event_id for event in sink.events),
            (EventId("event-1"), EventId("event-2")),
        )
        sink.clear()
        self.assertEqual(sink.events, ())

    def test_redacting_wrapper_protects_a_raw_sink(self) -> None:
        raw_sink = RecordingSink("raw")
        sink = RedactingEventSink(raw_sink)

        sink.emit(create_event(attributes={"api_key": "private"}))

        self.assertEqual(
            raw_sink.events[0].attributes["api_key"],
            REDACTED_VALUE,
        )

    def test_composite_delivers_to_every_sink_in_order(self) -> None:
        calls: list[str] = []
        first = RecordingSink("first", calls)
        second = RecordingSink("second", calls)
        sink = CompositeEventSink(first, second)
        event = create_event()

        sink.emit(event)

        self.assertEqual(calls, ["first", "second"])
        self.assertEqual(first.events, [event])
        self.assertEqual(second.events, [event])

    def test_composite_redacts_before_dispatching_to_raw_sinks(self) -> None:
        raw_sink = RecordingSink("raw")
        sink = CompositeEventSink(raw_sink)

        sink.emit(create_event(attributes={"password": "private"}))

        self.assertEqual(
            raw_sink.events[0].attributes["password"],
            REDACTED_VALUE,
        )

    def test_composite_attempts_all_sinks_before_raising(self) -> None:
        calls: list[str] = []
        first = FailingSink("first", calls)
        second = RecordingSink("second", calls)
        third = FailingSink("third", calls)
        sink = CompositeEventSink(first, second, third)

        with self.assertRaises(ExceptionGroup) as raised:
            sink.emit(create_event())

        self.assertEqual(calls, ["first", "second", "third"])
        self.assertEqual(len(raised.exception.exceptions), 2)
        self.assertIsInstance(raised.exception.exceptions[0], RuntimeError)
        self.assertEqual(len(second.events), 1)

    def test_composite_requires_at_least_one_sink(self) -> None:
        with self.assertRaises(ValueError):
            CompositeEventSink()


if __name__ == "__main__":
    unittest.main()
