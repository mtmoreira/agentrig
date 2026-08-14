from __future__ import annotations

import asyncio
import unittest

from agentrig.core import (
    CancellationSource,
    EffectProfile,
    EventKind,
    ExecutionStatus,
    FailureKind,
    RunId,
)

from examples.fundamentals.typed_sequence.scripted import (
    create_context,
    create_scripted_example,
    run_scripted_example,
)
from examples.fundamentals.typed_sequence.workflow import RawRequest


class TypedSequenceExampleTest(unittest.TestCase):
    def test_typed_sequence_retries_repeatable_transient_failure(self) -> None:
        run = asyncio.run(run_scripted_example())

        self.assertEqual(run.outcome.status, ExecutionStatus.SUCCEEDED)
        result = run.outcome.unwrap()
        self.assertEqual(result.message, "question: How do retries work?")
        self.assertEqual(result.category, "question")
        self.assertEqual(result.character_count, 20)
        self.assertEqual(len(run.classifier_calls), 2)
        self.assertIs(
            run.classifier_calls[0].context,
            run.classifier_calls[1].context,
        )

        started = tuple(
            event
            for event in run.events
            if event.kind is EventKind.STEP_STARTED
        )
        self.assertEqual(
            tuple(event.run_id for event in started),
            (RunId("run-2"), RunId("run-3"), RunId("run-3"), RunId("run-4")),
        )
        self.assertEqual(
            tuple(event.parent_run_id for event in started),
            (RunId("run-1"),) * 4,
        )
        self.assertEqual(
            tuple(event.attributes["attempt"] for event in started),
            (1, 1, 2, 1),
        )
        self.assertEqual(
            tuple(event.kind for event in run.events).count(
                EventKind.RETRY_SCHEDULED
            ),
            1,
        )
        serialized_events = " ".join(event.to_json() for event in run.events)
        self.assertNotIn("How do retries work?", serialized_events)

    def test_non_repeatable_failure_is_not_automatically_retried(self) -> None:
        run = asyncio.run(
            run_scripted_example(
                effect_profile=EffectProfile.NON_REPEATABLE,
            )
        )

        self.assertEqual(run.outcome.status, ExecutionStatus.FAILED)
        self.assertIsNotNone(run.outcome.failure)
        if run.outcome.failure is None:
            raise AssertionError("failed outcome has no failure")
        self.assertEqual(
            run.outcome.failure.kind,
            FailureKind.TRANSIENT_PROVIDER,
        )
        self.assertEqual(len(run.classifier_calls), 1)
        self.assertNotIn(
            EventKind.RETRY_SCHEDULED,
            tuple(event.kind for event in run.events),
        )

    def test_pre_cancelled_context_never_reaches_step_implementations(self) -> None:
        configured = create_scripted_example()
        source = CancellationSource()
        source.cancel("example caller stopped")
        context, sink = create_context(source)

        outcome = asyncio.run(
            configured.workflow.execute(
                RawRequest(text="How do retries work?"),
                context,
            )
        )

        self.assertEqual(outcome.status, ExecutionStatus.CANCELLED)
        self.assertEqual(configured.classifier.calls, [])
        self.assertEqual(
            tuple(event.kind for event in sink.events),
            (EventKind.STEP_STARTED, EventKind.STEP_COMPLETED),
        )
        self.assertEqual(sink.events[-1].attributes["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
