"""Clock contracts used by execution and observability."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    """Provides related wall and monotonic time readings."""

    def now(self) -> datetime:
        """Return an aware wall-clock timestamp."""
        ...

    def monotonic(self) -> float:
        """Return monotonic seconds for durations and deadlines."""
        ...


class SystemClock:
    """Read UTC wall time and the host's monotonic clock."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()
