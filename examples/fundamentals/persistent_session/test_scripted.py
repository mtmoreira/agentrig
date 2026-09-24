from __future__ import annotations

import asyncio
import json
import unittest

from examples.fundamentals.persistent_session.scripted import run_example


class PersistentSessionExampleTest(unittest.TestCase):
    def test_two_calls_share_one_process_and_shutdown_cleanly(self) -> None:
        report = asyncio.run(run_example())

        self.assertEqual(report.first_request_count, 1)
        self.assertEqual(report.second_request_count, 2)
        self.assertEqual(report.observed_values, ("alpha", "beta"))
        self.assertEqual(report.final_state, "closed")
        self.assertEqual(json.loads(report.to_json())["second_request_count"], 2)


if __name__ == "__main__":
    unittest.main()
