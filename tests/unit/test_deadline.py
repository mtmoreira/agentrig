from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import UTC, datetime

from agentrig.core import Clock, Deadline, DeadlineExceeded


@dataclass(frozen=True)
class FixedClock:
    wall_time: datetime
    monotonic_time: float

    def now(self) -> datetime:
        return self.wall_time

    def monotonic(self) -> float:
        return self.monotonic_time


def make_clock(monotonic_time: float) -> Clock:
    return FixedClock(datetime(2026, 8, 12, 12, 0, tzinfo=UTC), monotonic_time)


class DeadlineTest(unittest.TestCase):
    def test_after_uses_injected_wall_and_monotonic_clocks(self) -> None:
        deadline = Deadline.after(30.0, make_clock(100.0))

        self.assertEqual(
            deadline.expires_at,
            datetime(2026, 8, 12, 12, 0, 30, tzinfo=UTC),
        )
        self.assertEqual(deadline.monotonic_deadline, 130.0)

    def test_remaining_time_is_clamped_and_expiry_is_inclusive(self) -> None:
        deadline = Deadline.after(30.0, make_clock(100.0))

        self.assertEqual(deadline.remaining_seconds(make_clock(115.0)), 15.0)
        self.assertEqual(deadline.remaining_seconds(make_clock(131.0)), 0.0)
        self.assertFalse(deadline.is_expired(make_clock(129.9)))
        self.assertTrue(deadline.is_expired(make_clock(130.0)))

    def test_raise_if_expired_carries_deadline(self) -> None:
        deadline = Deadline.after(0.0, make_clock(100.0))

        with self.assertRaises(DeadlineExceeded) as raised:
            deadline.raise_if_expired(make_clock(100.0))

        self.assertIs(raised.exception.deadline, deadline)

    def test_earliest_ignores_absent_constraints(self) -> None:
        later = Deadline.after(20.0, make_clock(100.0))
        earlier = Deadline.after(10.0, make_clock(100.0))

        self.assertIs(Deadline.earliest(None, later, earlier), earlier)
        self.assertIsNone(Deadline.earliest(None))

    def test_invalid_timeouts_are_rejected(self) -> None:
        for timeout in (-1.0, float("inf"), float("nan")):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    Deadline.after(timeout, make_clock(100.0))

    def test_naive_wall_timestamp_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Deadline(datetime(2026, 8, 12, 12, 0), 100.0)


if __name__ == "__main__":
    unittest.main()
