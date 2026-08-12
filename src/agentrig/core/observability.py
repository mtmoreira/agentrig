"""Composable event sinks for local execution observability."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Protocol

from agentrig.core.events import Event
from agentrig.core.redaction import DEFAULT_REDACTION_POLICY, RedactionPolicy


class EventSink(Protocol):
    """Accept events without performing unbounded blocking work."""

    def emit(self, event: Event) -> None:
        """Record or enqueue an event, raising if delivery fails."""
        ...


@dataclass(frozen=True, slots=True)
class NoOpEventSink:
    """Discard all events."""

    def emit(self, event: Event) -> None:
        del event


NOOP_EVENT_SINK: EventSink = NoOpEventSink()


@dataclass(frozen=True, slots=True)
class RedactingEventSink:
    """Apply a redaction policy before forwarding to another sink."""

    sink: EventSink
    redaction_policy: RedactionPolicy = DEFAULT_REDACTION_POLICY

    def emit(self, event: Event) -> None:
        self.sink.emit(self.redaction_policy.redact(event))


class InMemoryEventSink:
    """Capture redacted events in emission order for tests and local runs."""

    def __init__(
        self,
        *,
        redaction_policy: RedactionPolicy = DEFAULT_REDACTION_POLICY,
    ) -> None:
        self._redaction_policy = redaction_policy
        self._events: list[Event] = []
        self._lock = Lock()

    @property
    def events(self) -> tuple[Event, ...]:
        """Return a stable snapshot of captured events."""
        with self._lock:
            return tuple(self._events)

    def emit(self, event: Event) -> None:
        redacted = self._redaction_policy.redact(event)
        with self._lock:
            self._events.append(redacted)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


@dataclass(frozen=True, slots=True, init=False)
class CompositeEventSink:
    """Deliver to every sink in order and aggregate delivery failures."""

    sinks: tuple[EventSink, ...]
    redaction_policy: RedactionPolicy

    def __init__(
        self,
        *sinks: EventSink,
        redaction_policy: RedactionPolicy = DEFAULT_REDACTION_POLICY,
    ) -> None:
        if not sinks:
            raise ValueError("composite event sink requires at least one sink")
        object.__setattr__(self, "sinks", tuple(sinks))
        object.__setattr__(self, "redaction_policy", redaction_policy)

    def emit(self, event: Event) -> None:
        redacted = self.redaction_policy.redact(event)
        failures: list[Exception] = []
        for sink in self.sinks:
            try:
                sink.emit(redacted)
            except Exception as error:
                failures.append(error)

        if failures:
            raise ExceptionGroup("one or more event sinks failed", failures)
