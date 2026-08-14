"""Internal ordered-outcome state shared by scripted test doubles."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Generic, TypeVar

OutcomeT = TypeVar("OutcomeT")
CallT = TypeVar("CallT")


class ScriptedOutcomes(Generic[OutcomeT, CallT]):
    """Record calls and consume immutable outcomes under one lock."""

    def __init__(
        self,
        *,
        outcomes: tuple[OutcomeT, ...],
        repeat_last: bool,
    ) -> None:
        if not isinstance(repeat_last, bool):
            raise TypeError("scripted outcomes repeat_last must be a bool")
        self._outcomes = outcomes
        self._repeat_last = repeat_last
        self._next_outcome = 0
        self._calls: list[CallT] = []
        self._lock = Lock()

    @property
    def calls(self) -> tuple[CallT, ...]:
        """Return a stable snapshot of recorded calls."""
        with self._lock:
            return tuple(self._calls)

    @property
    def is_exhausted(self) -> bool:
        """Whether another call would receive no scripted outcome."""
        with self._lock:
            return (
                not self._repeat_last
                and self._next_outcome >= len(self._outcomes)
            )

    def record_and_take(
        self,
        create_call: Callable[[int], CallT],
    ) -> OutcomeT | None:
        """Record one indexed call and atomically select its outcome."""
        with self._lock:
            call = create_call(len(self._calls))
            self._calls.append(call)
            if self._next_outcome >= len(self._outcomes):
                return None
            outcome = self._outcomes[self._next_outcome]
            if not (
                self._repeat_last
                and self._next_outcome == len(self._outcomes) - 1
            ):
                self._next_outcome += 1
            return outcome
