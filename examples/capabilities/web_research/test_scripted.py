from __future__ import annotations

import asyncio
from collections.abc import Mapping
import unittest

from agentrig.agents import AgentExecutionResult
from agentrig.capabilities import SearchProvider
from agentrig.core import AgentRigError, EventKind, FailureKind, JsonValue

from examples.capabilities.web_research.scripted import (
    SCRIPTED_RUNTIME_CAPABILITY_ID,
    SCRIPTED_WEB_SEARCH_TOOL_ID,
    SOURCE_ONE,
    SOURCE_TWO,
    create_context,
    create_scripted_example,
    example_request,
    run_scripted_example,
)


class WebResearchExampleTest(unittest.TestCase):
    def test_same_search_provider_runs_against_the_scripted_runtime(self) -> None:
        configured = create_scripted_example()

        run = asyncio.run(run_scripted_example(configured))

        self.assertIsInstance(configured.provider, SearchProvider)
        self.assertEqual(
            tuple(hit.source_uri for hit in run.result.hits),
            (SOURCE_ONE, SOURCE_TWO),
        )
        self.assertEqual(run.result.metadata.total_available, 2)
        self.assertEqual(run.result.metadata.duration_seconds, 0.0)
        self.assertEqual(len(run.runtime.calls), 1)
        call = run.runtime.calls[0]
        self.assertEqual(
            call.request.contract.allowed_capabilities,
            (SCRIPTED_RUNTIME_CAPABILITY_ID,),
        )
        self.assertEqual(
            call.request.contract.allowed_tools,
            (SCRIPTED_WEB_SEARCH_TOOL_ID,),
        )
        self.assertEqual(call.request.contract.limits.max_turns, 1)
        self.assertEqual(call.request.contract.limits.max_tool_calls, 6)
        self.assertEqual(
            call.request.contract.permissions,
            {"workspace": "read_only", "network": "allowed"},
        )
        self.assertIsInstance(call.request.input, Mapping)
        if not isinstance(call.request.input, Mapping):
            raise AssertionError("encoded search request must be an object")
        self.assertEqual(call.request.input["max_results"], 2)
        self.assertEqual(
            tuple(event.kind for event in run.events),
            (
                EventKind.PROVIDER_CALL_STARTED,
                EventKind.PROVIDER_CALL_COMPLETED,
            ),
        )
        serialized_events = "\n".join(event.to_json() for event in run.events)
        self.assertNotIn("portable autonomous runtime", serialized_events)
        self.assertNotIn("application policy", serialized_events)

    def test_request_capacity_fails_before_runtime_execution(self) -> None:
        configured = create_scripted_example()
        context, _ = create_context()

        with self.assertRaises(ValueError):
            asyncio.run(
                configured.provider.search(
                    example_request(max_results=4),
                    context,
                )
            )

        self.assertEqual(configured.runtime.calls, ())

    def test_request_relative_result_bound_is_safely_rejected(self) -> None:
        configured = create_scripted_example()
        context, _ = create_context()

        with self.assertRaises(AgentRigError) as raised:
            asyncio.run(
                configured.provider.search(
                    example_request(max_results=1),
                    context,
                )
            )

        self.assertEqual(raised.exception.failure.kind, FailureKind.INVALID_INPUT)
        self.assertEqual(
            raised.exception.failure.code,
            "example.research_report_invalid",
        )
        self.assertNotIn(SOURCE_TWO, raised.exception.failure.message)

    def test_non_https_and_duplicate_citations_are_schema_failures(self) -> None:
        invalid_outputs: tuple[JsonValue, ...] = (
            {
                "hits": (
                    {
                        "source_uri": "file:///private/report",
                        "title": "Local file",
                        "summary": "This is not a web citation.",
                    },
                )
            },
            {
                "hits": (
                    {
                        "source_uri": SOURCE_ONE,
                        "title": "First",
                        "summary": "First summary.",
                    },
                    {
                        "source_uri": SOURCE_ONE,
                        "title": "Duplicate",
                        "summary": "Duplicate summary.",
                    },
                )
            },
        )

        for invalid_output in invalid_outputs:
            with self.subTest(invalid_output=invalid_output):
                configured = create_scripted_example(
                    AgentExecutionResult.succeeded(invalid_output)
                )
                context, _ = create_context()

                with self.assertRaises(AgentRigError) as raised:
                    asyncio.run(
                        configured.provider.search(example_request(), context)
                    )

                self.assertEqual(
                    raised.exception.failure.code,
                    "agent.output_schema_mismatch",
                )
                self.assertNotIn(
                    "file:///private/report",
                    raised.exception.failure.message,
                )


if __name__ == "__main__":
    unittest.main()
