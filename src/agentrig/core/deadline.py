"""Monotonic execution deadlines with UTC diagnostic timestamps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from agentrig.core.clock import Clock


@dataclass(frozen=True, slots=True)
class Deadline:
    """An absolute monotonic deadline and its wall-clock representation."""

    expires_at: datetime
    monotonic_deadline: float

    def __post_init__(self) -> None:
        if self.expires_at.tzinfo is None or self.expires_at.utcoffset() is None:
            raise ValueError("deadline timestamp must be timezone-aware")
        if not math.isfinite(self.monotonic_deadline):
            raise ValueError("monotonic deadline must be finite")

        object.__setattr__(self, "expires_at", self.expires_at.astimezone(UTC))

    @classmethod
    def after(cls, timeout_seconds: float, clock: Clock) -> Deadline:
        """Create a deadline relative to injected wall and monotonic clocks."""
        if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("timeout must be finite and non-negative")

        return cls(
            expires_at=clock.now() + timedelta(seconds=timeout_seconds),
            monotonic_deadline=clock.monotonic() + timeout_seconds,
        )

    def remaining_seconds(self, clock: Clock) -> float:
        """Return remaining monotonic time, clamped to zero."""
        return max(0.0, self.monotonic_deadline - clock.monotonic())

    def is_expired(self, clock: Clock) -> bool:
        return clock.monotonic() >= self.monotonic_deadline

    def raise_if_expired(self, clock: Clock) -> None:
        if self.is_expired(clock):
            raise DeadlineExceeded(self)

    @staticmethod
    def earliest(*deadlines: Deadline | None) -> Deadline | None:
        """Return the strictest deadline, ignoring absent constraints."""
        present = tuple(deadline for deadline in deadlines if deadline is not None)
        if not present:
            return None
        return min(present, key=lambda deadline: deadline.monotonic_deadline)


class DeadlineExceeded(TimeoutError):
    """Raised when execution observes an expired deadline."""

    def __init__(self, deadline: Deadline) -> None:
        self.deadline = deadline
        super().__init__(f"deadline exceeded at {deadline.expires_at.isoformat()}")
