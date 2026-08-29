from __future__ import annotations

import asyncio
import json
import unittest

from agentrig.core import FailureKind

from examples.capabilities.bounded_image_edit.scripted import (
    example_request,
    report,
    run_scripted_example,
)


class BoundedImageEditExampleTest(unittest.TestCase):
    def test_visible_report_proves_lineage_retry_and_single_route(self) -> None:
        run = asyncio.run(run_scripted_example())
        value = report(run)

        self.assertTrue(run.execution.outcome.is_success)
        self.assertEqual(run.selected_calls, 2)
        self.assertEqual(run.unselected_calls, 0)
        self.assertEqual(
            tuple(item.failure_kind for item in run.execution.attempts),
            (FailureKind.TRANSIENT_PROVIDER, None),
        )
        self.assertEqual(
            run.execution.outcome.unwrap().image.input_artifact_ids,
            example_request().source_artifact_ids,
        )
        self.assertIsNone(value["usage_cost"])
        encoded = json.dumps(value, sort_keys=True)
        self.assertNotIn("Keep both fictional", encoded)
        self.assertNotIn("scripted route is temporarily busy", encoded)


if __name__ == "__main__":
    unittest.main()
