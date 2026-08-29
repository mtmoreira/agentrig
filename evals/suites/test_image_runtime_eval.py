from __future__ import annotations

import asyncio
import unittest

from agentrig.capabilities import ImageInputRole
from agentrig.core import FailureKind

from examples.capabilities.bounded_image_edit.scripted import (
    example_request,
    run_scripted_example,
)


class ImageRuntimeEval(unittest.TestCase):
    def test_bounded_edit_qualification_cases(self) -> None:
        run = asyncio.run(run_scripted_example())
        request = example_request()
        output = run.execution.outcome.unwrap()
        cases = {
            "explicit_edit_base": request.inputs[0].role
            is ImageInputRole.EDIT_BASE,
            "explicit_mask": request.inputs[-1].role
            is ImageInputRole.EDIT_MASK,
            "complete_lineage": output.image.input_artifact_ids
            == request.source_artifact_ids,
            "bounded_attempts": len(run.execution.attempts) == 2,
            "declared_retry_only": tuple(
                item.failure_kind for item in run.execution.attempts
            )
            == (FailureKind.TRANSIENT_PROVIDER, None),
            "selected_route_only": run.selected_calls == 2
            and run.unselected_calls == 0,
            "unknown_usage_preserved": output.usage.cost is None,
            "offline": True,
        }
        self.assertEqual(len(cases), 8)
        self.assertTrue(all(cases.values()), cases)


if __name__ == "__main__":
    unittest.main()
