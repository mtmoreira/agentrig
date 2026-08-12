from __future__ import annotations

import asyncio
import unittest

from agentrig.core import Cancellation, CancellationSource, RunCancelled


class CancellationSourceTest(unittest.IsolatedAsyncioTestCase):
    def test_first_cancellation_request_wins(self) -> None:
        source = CancellationSource()

        self.assertTrue(source.cancel("user requested stop"))
        self.assertFalse(source.cancel("later request"))
        self.assertEqual(
            source.token.cancellation,
            Cancellation("user requested stop"),
        )

    def test_parent_cancellation_propagates_to_child(self) -> None:
        parent = CancellationSource()
        child = parent.create_child()

        parent.cancel("parent stopped")

        self.assertEqual(child.token.cancellation, Cancellation("parent stopped"))

    def test_child_created_after_parent_cancellation_starts_cancelled(self) -> None:
        parent = CancellationSource()
        parent.cancel("parent already stopped")

        child = parent.create_child()

        self.assertEqual(
            child.token.cancellation,
            Cancellation("parent already stopped"),
        )

    def test_child_cancellation_does_not_cancel_parent(self) -> None:
        parent = CancellationSource()
        child = parent.create_child()

        child.cancel("child stopped")

        self.assertFalse(parent.token.is_cancelled)
        parent.cancel("parent stopped later")
        self.assertEqual(child.token.cancellation, Cancellation("child stopped"))

    def test_raise_if_cancelled_preserves_reason(self) -> None:
        source = CancellationSource()
        source.cancel("approval denied")

        with self.assertRaises(RunCancelled) as raised:
            source.token.raise_if_cancelled()

        self.assertEqual(raised.exception.cancellation, Cancellation("approval denied"))

    async def test_wait_cancelled_observes_later_parent_request(self) -> None:
        parent = CancellationSource()
        child = parent.create_child()
        waiter = asyncio.create_task(child.token.wait_cancelled())

        await asyncio.sleep(0)
        parent.cancel("deadline owner stopped")

        self.assertEqual(
            await waiter,
            Cancellation("deadline owner stopped"),
        )

    async def test_abandoned_waiter_unsubscribes_safely(self) -> None:
        source = CancellationSource()
        waiter = asyncio.create_task(source.token.wait_cancelled())
        await asyncio.sleep(0)

        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter

        self.assertTrue(source.cancel("cancelled after waiter left"))

    def test_cancellation_reason_must_be_nonempty_and_trimmed(self) -> None:
        for reason in ("", " padded", "padded "):
            with self.subTest(reason=reason):
                with self.assertRaises(ValueError):
                    Cancellation(reason)


if __name__ == "__main__":
    unittest.main()
