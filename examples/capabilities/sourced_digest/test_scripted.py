from __future__ import annotations

import asyncio
import unittest

from agentrig.capabilities import SearchProvider, StructuredGenerator
from agentrig.core import EventKind, ExecutionStatus, FailureKind, RunId

from examples.capabilities.sourced_digest.scripted import (
    SOURCE_ONE,
    SOURCE_TWO,
    create_context,
    create_scripted_example,
    run_scripted_example,
)
from examples.capabilities.sourced_digest.workflow import DigestRequest


class SourcedDigestExampleTest(unittest.TestCase):
    def test_search_and_structured_generation_compose_without_a_provider(self) -> None:
        run = asyncio.run(run_scripted_example())

        self.assertEqual(run.outcome.status, ExecutionStatus.SUCCEEDED)
        result = run.outcome.unwrap()
        self.assertEqual(
            result.headline,
            "Portable, testable capability pipelines",
        )
        self.assertEqual(
            tuple(citation.source_uri for citation in result.citations),
            (SOURCE_ONE, SOURCE_TWO),
        )
        self.assertEqual(result.search_metadata.total_available, 2)
        self.assertEqual(result.generation_usage.total_tokens, 96)
        self.assertEqual(result.generation_model.provider, "scripted")
        self.assertEqual((run.search_calls, run.generation_calls), (1, 1))
        self.assertEqual(
            tuple(event.kind for event in run.events),
            (
                EventKind.STEP_STARTED,
                EventKind.STEP_COMPLETED,
                EventKind.STEP_STARTED,
                EventKind.STEP_COMPLETED,
            ),
        )
        self.assertEqual(
            tuple(event.run_id for event in run.events[::2]),
            (RunId("run-2"), RunId("run-3")),
        )
        self.assertEqual(
            tuple(event.parent_run_id for event in run.events[::2]),
            (RunId("run-1"), RunId("run-1")),
        )
        serialized_events = " ".join(event.to_json() for event in run.events)
        self.assertNotIn("provider-neutral AI capabilities", serialized_events)
        self.assertNotIn("Portable contracts separate", serialized_events)
        self.assertNotIn(result.summary, serialized_events)

    def test_capability_requirements_fail_before_provider_execution(self) -> None:
        configured = create_scripted_example(search_capacity=1)
        self.assertIsInstance(configured.search_provider, SearchProvider)
        self.assertIsInstance(configured.generator, StructuredGenerator)
        context, _ = create_context()

        outcome = asyncio.run(
            configured.workflow.execute(
                DigestRequest(topic="bounded search", max_sources=2),
                context,
            )
        )

        self.assertEqual(outcome.status, ExecutionStatus.FAILED)
        self.assertIsNotNone(outcome.failure)
        if outcome.failure is None:
            raise AssertionError("failed outcome has no failure")
        self.assertEqual(outcome.failure.kind, FailureKind.INVALID_INPUT)
        self.assertEqual(
            outcome.failure.code,
            "example.search_requirements_unmet",
        )
        self.assertEqual(configured.search_provider.calls, ())
        self.assertEqual(configured.generator.calls, ())

    def test_generated_citations_must_come_from_search_results(self) -> None:
        invented_source = "https://untrusted.test/invented"
        configured = create_scripted_example(
            encoded_output={
                "headline": "Untrusted",
                "summary": "This digest invented its evidence.",
                "source_uris": [invented_source],
            }
        )
        context, sink = create_context()

        outcome = asyncio.run(
            configured.workflow.execute(
                DigestRequest(topic="citation provenance"),
                context,
            )
        )

        self.assertEqual(outcome.status, ExecutionStatus.FAILED)
        self.assertIsNotNone(outcome.failure)
        if outcome.failure is None:
            raise AssertionError("failed outcome has no failure")
        self.assertEqual(outcome.failure.kind, FailureKind.INVALID_INPUT)
        self.assertEqual(
            outcome.failure.code,
            "example.citation_not_in_search_results",
        )
        self.assertEqual(len(configured.search_provider.calls), 1)
        self.assertEqual(len(configured.generator.calls), 1)
        self.assertNotIn(invented_source, outcome.failure.message)
        self.assertNotIn(
            invented_source,
            " ".join(event.to_json() for event in sink.events),
        )

    def test_malformed_structured_output_is_safely_normalized(self) -> None:
        configured = create_scripted_example(
            encoded_output={"headline": "Missing required fields"}
        )
        context, _ = create_context()

        outcome = asyncio.run(
            configured.workflow.execute(
                DigestRequest(topic="strict output"),
                context,
            )
        )

        self.assertEqual(outcome.status, ExecutionStatus.FAILED)
        self.assertIsNotNone(outcome.failure)
        if outcome.failure is None:
            raise AssertionError("failed outcome has no failure")
        self.assertEqual(outcome.failure.kind, FailureKind.INVALID_INPUT)
        self.assertEqual(
            outcome.failure.code,
            "example.digest_schema_mismatch",
        )
        self.assertEqual(len(configured.search_provider.calls), 1)
        self.assertEqual(len(configured.generator.calls), 1)


if __name__ == "__main__":
    unittest.main()
