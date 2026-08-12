from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import UTC, datetime

from agentrig.core import Clock, SystemClock


@dataclass(frozen=True)
class FixedClock:
    wall_time: datetime
    monotonic_time: float

    def now(self) -> datetime:
        return self.wall_time

    def monotonic(self) -> float:
        return self.monotonic_time


def read_clock(clock: Clock) -> tuple[datetime, float]:
    return clock.now(), clock.monotonic()


class ClockTest(unittest.TestCase):
    def test_clock_protocol_accepts_an_injected_implementation(self) -> None:
        wall_time = datetime(2026, 8, 12, tzinfo=UTC)

        self.assertEqual(read_clock(FixedClock(wall_time, 42.0)), (wall_time, 42.0))

    def test_system_clock_returns_aware_utc_and_monotonic_time(self) -> None:
        clock = SystemClock()
        before = clock.monotonic()
        wall_time = clock.now()
        after = clock.monotonic()

        self.assertIs(wall_time.tzinfo, UTC)
        self.assertLessEqual(before, after)


if __name__ == "__main__":
    unittest.main()
