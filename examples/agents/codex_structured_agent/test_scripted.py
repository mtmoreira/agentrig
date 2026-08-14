from __future__ import annotations

import asyncio
import unittest

from agentrig.agents import AgentExecutionResult, AgentStatus
from agentrig.core import EventKind, FailureKind, JsonValue

from examples.agents.codex_structured_agent.scripted import (
    SCRIPTED_CAPABILITY_ID,
    create_scripted_example,
    run_scripted_example,
)


class CodexStructuredAgentExampleTest(unittest.TestCase):
    def test_same_typed_agent_runs_against_the_scripted_runtime(self) -> None:
        configured = create_scripted_example()

        run = asyncio.run(run_scripted_example(configured))

        brief = run.result.unwrap()
        self.assertEqual(brief.recommendation, "proceed")
        self.assertEqual(len(brief.risks), 1)
        self.assertEqual(len(run.runtime.calls), 1)
        call = run.runtime.calls[0]
        self.assertEqual(
            call.request.contract.allowed_capabilities,
            (SCRIPTED_CAPABILITY_ID,),
        )
        self.assertEqual(call.request.contract.allowed_tools, ())
        self.assertEqual(call.request.contract.limits.max_tool_calls, 0)
        self.assertEqual(
            call.request.contract.permissions,
            {"workspace": "read_only", "network": "denied"},
        )
        self.assertEqual(
            call.request.input,
            {
                "question": "Should the team automate this release check?",
                "constraints": (
                    "The check must be deterministic offline.",
                    "Live provider execution must remain explicit.",
                ),
            },
        )
        self.assertEqual(
            tuple(event.kind for event in run.events),
            (
                EventKind.PROVIDER_CALL_STARTED,
                EventKind.PROVIDER_CALL_COMPLETED,
            ),
        )
        serialized_events = "\n".join(event.to_json() for event in run.events)
        self.assertNotIn("automate this release check", serialized_events)
        self.assertNotIn("Automate the deterministic gate", serialized_events)

    def test_malformed_output_is_a_sanitized_schema_failure(self) -> None:
        configured = create_scripted_example(
            AgentExecutionResult.succeeded(
                {
                    "summary": "Missing the required fields.",
                }
            )
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
        self.assertNotIn(
            "Missing the required fields",
            run.result.failure.message,
        )

    def test_decoder_enforces_domain_rules_outside_provider_schema(self) -> None:
        invalid_outputs: tuple[JsonValue, ...] = (
            {
                "summary": "No risks were returned.",
                "risks": [],
                "recommendation": "proceed",
            },
            {
                "summary": "The recommendation is outside the vocabulary.",
                "risks": ["One bounded risk."],
                "recommendation": "wait",
            },
        )

        for invalid_output in invalid_outputs:
            with self.subTest(invalid_output=invalid_output):
                configured = create_scripted_example(
                    AgentExecutionResult.succeeded(invalid_output)
                )

                run = asyncio.run(run_scripted_example(configured))

                self.assertEqual(run.result.status, AgentStatus.FAILED)
                self.assertIsNotNone(run.result.failure)
                if run.result.failure is None:
                    raise AssertionError("failed result has no failure")
                self.assertEqual(
                    run.result.failure.code,
                    "agent.output_schema_mismatch",
                )

    def test_contract_rejects_invalid_runtime_capability_identity(self) -> None:
        from examples.agents.codex_structured_agent.workflow import (
            decision_contract,
        )

        with self.assertRaises(ValueError):
            decision_contract(runtime_capability_id=" provider ")


if __name__ == "__main__":
    unittest.main()
