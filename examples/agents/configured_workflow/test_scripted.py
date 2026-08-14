from __future__ import annotations

import asyncio
import unittest

from agentrig.agents import Agent, AgentExecutionResult, AgentStatus
from agentrig.core import EventKind, FailureKind, RunId
from agentrig.workflow import Workflow

from examples.agents.configured_workflow.scripted import (
    create_scripted_example,
    run_scripted_example,
)


class ConfiguredWorkflowExampleTest(unittest.TestCase):
    def test_configured_agent_round_trips_through_a_workflow_agent(self) -> None:
        configured = create_scripted_example()
        run = asyncio.run(run_scripted_example(configured))

        self.assertIsInstance(configured.workflow, Workflow)
        self.assertIsInstance(configured.agent, Agent)
        self.assertEqual(configured.agent.contract.agent_id, "research-delivery")
        self.assertEqual(configured.agent.contract.allowed_tools, ("search",))
        self.assertEqual(
            configured.agent.contract.permissions["network"],
            "search_only",
        )
        self.assertEqual(run.result.status, AgentStatus.SUCCEEDED)
        delivered = run.result.unwrap()
        self.assertEqual(delivered.word_count, 10)
        self.assertEqual(
            delivered.brief.sources,
            ("https://example.test/contracts",),
        )

        self.assertEqual(len(run.runtime.calls), 1)
        runtime_call = run.runtime.calls[0]
        self.assertEqual(
            runtime_call.request.input,
            {"topic": "portable agent contracts"},
        )
        self.assertEqual(runtime_call.request.contract.agent_id, "researcher")
        self.assertEqual(runtime_call.request.contract.allowed_tools, ("search",))
        self.assertEqual(
            runtime_call.request.contract.permissions["network"],
            "search_only",
        )

        self.assertEqual(
            tuple(event.kind for event in run.events),
            (
                EventKind.STEP_STARTED,
                EventKind.PROVIDER_CALL_STARTED,
                EventKind.PROGRESS_REPORTED,
                EventKind.TOOL_CALL_STARTED,
                EventKind.TOOL_CALL_COMPLETED,
                EventKind.PROVIDER_CALL_COMPLETED,
                EventKind.STEP_COMPLETED,
                EventKind.STEP_STARTED,
                EventKind.STEP_COMPLETED,
            ),
        )
        self.assertEqual(
            tuple(
                event.run_id
                for event in run.events
                if event.kind is EventKind.STEP_STARTED
            ),
            (RunId("run-2"), RunId("run-3")),
        )
        serialized_events = " ".join(event.to_json() for event in run.events)
        self.assertNotIn("portable agent contracts", serialized_events)
        self.assertNotIn("Typed agent contracts", serialized_events)
        self.assertFalse(hasattr(run.result, "provider_metadata"))

    def test_malformed_runtime_output_stops_before_downstream_step(self) -> None:
        configured = create_scripted_example(
            AgentExecutionResult.succeeded({"answer": "missing sources"})
        )

        run = asyncio.run(run_scripted_example(configured))

        self.assertEqual(run.result.status, AgentStatus.FAILED)
        self.assertIsNotNone(run.result.failure)
        if run.result.failure is None:
            raise AssertionError("failed result has no failure")
        self.assertEqual(run.result.failure.kind, FailureKind.INVALID_INPUT)
        self.assertEqual(
            run.result.failure.code,
            "agent.output_schema_mismatch",
        )
        self.assertEqual(
            tuple(event.kind for event in run.events).count(
                EventKind.STEP_STARTED
            ),
            1,
        )

    def test_disallowed_runtime_tool_is_rejected_before_execution(self) -> None:
        configured = create_scripted_example(tool_name="write")

        run = asyncio.run(run_scripted_example(configured))

        self.assertEqual(run.result.status, AgentStatus.FAILED)
        self.assertIsNotNone(run.result.failure)
        if run.result.failure is None:
            raise AssertionError("failed result has no failure")
        self.assertEqual(run.result.failure.kind, FailureKind.INVALID_INPUT)
        self.assertEqual(run.result.failure.code, "agent.tool_not_allowed")
        self.assertNotIn(
            EventKind.TOOL_CALL_STARTED,
            tuple(event.kind for event in run.events),
        )


if __name__ == "__main__":
    unittest.main()
