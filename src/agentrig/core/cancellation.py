"""Cooperative cancellation with deterministic parent propagation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock

CancellationCallback = Callable[["Cancellation"], None]
Unsubscribe = Callable[[], None]


@dataclass(frozen=True, slots=True)
class Cancellation:
    """The first cancellation request observed by an execution."""

    reason: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("cancellation reason must not be empty")
        if self.reason != self.reason.strip():
            raise ValueError("cancellation reason must not have surrounding whitespace")


class RunCancelled(asyncio.CancelledError):
    """Raised when cooperative execution observes cancellation."""

    def __init__(self, cancellation: Cancellation) -> None:
        self.cancellation = cancellation
        super().__init__(cancellation.reason)


class _CancellationState:
    def __init__(self) -> None:
        self._lock = Lock()
        self._cancellation: Cancellation | None = None
        self._callbacks: list[CancellationCallback] = []

    @property
    def cancellation(self) -> Cancellation | None:
        with self._lock:
            return self._cancellation

    def cancel(self, cancellation: Cancellation) -> bool:
        with self._lock:
            if self._cancellation is not None:
                return False

            self._cancellation = cancellation
            callbacks = tuple(self._callbacks)
            self._callbacks.clear()

        for callback in callbacks:
            callback(cancellation)

        return True

    def subscribe(self, callback: CancellationCallback) -> Unsubscribe:
        with self._lock:
            cancellation = self._cancellation
            if cancellation is None:
                self._callbacks.append(callback)

        if cancellation is not None:
            callback(cancellation)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._callbacks:
                    self._callbacks.remove(callback)

        return unsubscribe


class CancellationToken:
    """Read-only view of cooperative cancellation state."""

    def __init__(self, state: _CancellationState) -> None:
        self._state = state

    @property
    def cancellation(self) -> Cancellation | None:
        return self._state.cancellation

    @property
    def is_cancelled(self) -> bool:
        return self.cancellation is not None

    def raise_if_cancelled(self) -> None:
        cancellation = self.cancellation
        if cancellation is not None:
            raise RunCancelled(cancellation)

    async def wait_cancelled(self) -> Cancellation:
        """Wait until cancellation without polling or binding at construction."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Cancellation] = loop.create_future()

        def notify(cancellation: Cancellation) -> None:
            def resolve() -> None:
                if not future.done():
                    future.set_result(cancellation)

            try:
                loop.call_soon_threadsafe(resolve)
            except RuntimeError:
                # The waiter can lose a race with event-loop shutdown.
                pass

        unsubscribe = self._state.subscribe(notify)
        try:
            return await future
        finally:
            unsubscribe()

    def _subscribe(self, callback: CancellationCallback) -> Unsubscribe:
        return self._state.subscribe(callback)


class CancellationSource:
    """Owns cancellation authority for one execution scope."""

    def __init__(self, parent: CancellationToken | None = None) -> None:
        self._state = _CancellationState()
        self._token = CancellationToken(self._state)
        self._parent_subscription: Unsubscribe | None = None

        if parent is not None:
            def propagate(cancellation: Cancellation) -> None:
                if self._state.cancel(cancellation):
                    self._detach_from_parent()

            self._parent_subscription = parent._subscribe(propagate)
            if self._state.cancellation is not None:
                self._detach_from_parent()

    @property
    def token(self) -> CancellationToken:
        return self._token

    def cancel(self, reason: str = "cancelled") -> bool:
        """Request cancellation once; the first request wins."""
        cancellation_recorded = self._state.cancel(Cancellation(reason))
        if cancellation_recorded:
            self._detach_from_parent()
        return cancellation_recorded

    def create_child(self) -> CancellationSource:
        """Create a source cancelled by this source, but not vice versa."""
        return CancellationSource(parent=self.token)

    def _detach_from_parent(self) -> None:
        unsubscribe = self._parent_subscription
        self._parent_subscription = None
        if unsubscribe is not None:
            unsubscribe()
