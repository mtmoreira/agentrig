from __future__ import annotations

import asyncio
from collections.abc import Mapping
import unittest

from agentrig.agents import AgentExecutionResult
from agentrig.capabilities import CodingAgent, CodingStatus
from agentrig.core import AgentRigError, EventKind, FailureKind, JsonValue

from examples.capabilities.bounded_coding.scripted import (
    SCRIPTED_RUNTIME_CAPABILITY_ID,
    SCRIPTED_TOOL_ID,
    create_context,
    create_scripted_example,
    example_task,
    run_scripted_example,
)


class BoundedCodingExampleTest(unittest.TestCase):
    def test_same_coding_capability_runs_against_the_scripted_runtime(self) -> None:
        configured = create_scripted_example()

        run = asyncio.run(run_scripted_example(configured))

        self.assertIsInstance(configured.agent, CodingAgent)
        self.assertEqual(run.result.status, CodingStatus.SUCCEEDED)
        self.assertEqual(
            tuple(change.path for change in run.result.changed_files),
            ("greeting.py",),
        )
        self.assertEqual(
            tuple(item.validation_id for item in run.result.validations),
            ("python-compile",),
        )
        self.assertEqual(len(run.runtime.calls), 1)
        call = run.runtime.calls[0]
        self.assertEqual(
            call.request.contract.allowed_capabilities,
            (SCRIPTED_RUNTIME_CAPABILITY_ID,),
        )
        self.assertEqual(
            call.request.contract.allowed_tools,
            (SCRIPTED_TOOL_ID,),
        )
        self.assertEqual(call.request.contract.limits.max_turns, 1)
        self.assertEqual(call.request.contract.limits.max_tool_calls, 8)
        self.assertEqual(
            call.request.contract.permissions,
            {"workspace": "read_write", "network": "denied"},
        )
        self.assertIsInstance(call.request.input, Mapping)
        if not isinstance(call.request.input, Mapping):
            raise AssertionError("encoded coding task must be an object")
        self.assertEqual(call.request.input["max_changed_files"], 1)
        self.assertEqual(
            tuple(event.kind for event in run.events),
            (
                EventKind.PROVIDER_CALL_STARTED,
                EventKind.PROVIDER_CALL_COMPLETED,
            ),
        )
        serialized_events = "\n".join(event.to_json() for event in run.events)
        self.assertNotIn("Create greeting.py", serialized_events)
        self.assertNotIn("compiled successfully", serialized_events)

    def test_task_capacity_fails_before_runtime_execution(self) -> None:
        configured = create_scripted_example()
        context, _ = create_context()

        with self.assertRaises(ValueError):
            asyncio.run(
                configured.agent.execute(
                    example_task(max_changed_files=2),
                    context,
                )
            )

        self.assertEqual(configured.runtime.calls, ())

    def test_reported_change_outside_authorization_is_safely_rejected(self) -> None:
        configured = create_scripted_example(
            AgentExecutionResult.succeeded(
                {
                    "changed_files": (
                        {"path": "../outside.py", "change_kind": "added"},
                    ),
                    "validations": (
                        {
                            "validation_id": "python-compile",
                            "summary": "Provider claimed success.",
                            "exit_code": 0,
                        },
                    ),
                }
            )
        )
        context, _ = create_context()

        with self.assertRaises(AgentRigError) as raised:
            asyncio.run(configured.agent.execute(example_task(), context))

        self.assertEqual(raised.exception.failure.kind, FailureKind.INVALID_INPUT)
        self.assertEqual(
            raised.exception.failure.code,
            "agent.output_schema_mismatch",
        )
        self.assertNotIn("../outside.py", raised.exception.failure.message)

    def test_task_relative_bound_rejects_extra_reported_files(self) -> None:
        encoded: JsonValue = {
            "changed_files": (
                {"path": "greeting.py", "change_kind": "added"},
                {"path": "notes.txt", "change_kind": "added"},
            ),
            "validations": (
                {
                    "validation_id": "python-compile",
                    "summary": "Provider claimed success.",
                    "exit_code": 0,
                },
            ),
        }
        configured = create_scripted_example(
            AgentExecutionResult.succeeded(encoded)
        )
        context, _ = create_context()

        with self.assertRaises(AgentRigError) as raised:
            asyncio.run(configured.agent.execute(example_task(), context))

        self.assertEqual(
            raised.exception.failure.code,
            "agent.output_schema_mismatch",
        )


if __name__ == "__main__":
    unittest.main()
