from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime

from agentrig.core import (
    AgentRigError,
    Cancellation,
    Deadline,
    DeadlineExceeded,
    Failure,
    FailureKind,
    RunCancelled,
    normalize_exception,
)


class FailureKindTest(unittest.TestCase):
    def test_vocabulary_has_stable_wire_values(self) -> None:
        self.assertEqual(
            {kind.value for kind in FailureKind},
            {
                "approval_denied",
                "approval_required",
                "budget_exhausted",
                "cancelled",
                "deadline_exceeded",
                "grader_failed",
                "invalid_input",
                "permanent_provider",
                "policy_refusal",
                "transient_provider",
                "unexpected",
                "workflow_blocked",
            },
        )


class FailureTest(unittest.TestCase):
    def test_copies_and_freezes_sanitized_metadata(self) -> None:
        metadata = {"provider_code": "rate_limit"}
        failure = Failure(
            kind=FailureKind.TRANSIENT_PROVIDER,
            message="provider temporarily unavailable",
            code="provider.rate_limit",
            metadata=metadata,
        )
        metadata["provider_code"] = "mutated"

        self.assertEqual(failure.metadata["provider_code"], "rate_limit")
        self.assertTrue(failure.is_retryable)
        with self.assertRaises(TypeError):
            failure.metadata["new"] = "value"  # type: ignore[index]

    def test_only_transient_provider_category_is_retryable(self) -> None:
        for kind in FailureKind:
            with self.subTest(kind=kind):
                failure = Failure(kind=kind, message="safe message")
                self.assertEqual(
                    failure.is_retryable,
                    kind is FailureKind.TRANSIENT_PROVIDER,
                )

    def test_requires_valid_kind_message_code_and_metadata(self) -> None:
        with self.assertRaises(TypeError):
            Failure(kind="unexpected", message="message")  # type: ignore[arg-type]
        for message in ("", " padded", "padded "):
            with self.subTest(message=message):
                with self.assertRaises(ValueError):
                    Failure(kind=FailureKind.UNEXPECTED, message=message)
        with self.assertRaises(ValueError):
            Failure(
                kind=FailureKind.UNEXPECTED,
                message="message",
                code=" padded",
            )
        with self.assertRaises(ValueError):
            Failure(
                kind=FailureKind.UNEXPECTED,
                message="message",
                metadata={"key": ""},
            )

    def test_agentrig_error_preserves_normalized_failure(self) -> None:
        failure = Failure(
            kind=FailureKind.POLICY_REFUSAL,
            message="request refused by policy",
        )
        error = AgentRigError(failure)

        self.assertIs(error.failure, failure)
        self.assertEqual(str(error), failure.message)
        self.assertIs(normalize_exception(error), failure)
        with self.assertRaises(TypeError):
            AgentRigError("message")  # type: ignore[arg-type]


class NormalizeExceptionTest(unittest.TestCase):
    def test_run_cancellation_preserves_reason(self) -> None:
        failure = normalize_exception(
            RunCancelled(Cancellation("caller requested stop"))
        )

        self.assertEqual(failure.kind, FailureKind.CANCELLED)
        self.assertEqual(failure.message, "caller requested stop")

    def test_generic_asyncio_cancellation_uses_safe_message(self) -> None:
        failure = normalize_exception(asyncio.CancelledError("private detail"))

        self.assertEqual(failure.kind, FailureKind.CANCELLED)
        self.assertEqual(failure.message, "execution cancelled")
        self.assertNotIn("private", failure.message)

    def test_deadline_failure_carries_only_structured_timestamp(self) -> None:
        deadline = Deadline(
            expires_at=datetime(2026, 8, 12, 20, 0, tzinfo=UTC),
            monotonic_deadline=100.0,
        )

        failure = normalize_exception(DeadlineExceeded(deadline))

        self.assertEqual(failure.kind, FailureKind.DEADLINE_EXCEEDED)
        self.assertEqual(failure.message, "execution deadline exceeded")
        self.assertEqual(
            failure.metadata,
            {"expires_at": "2026-08-12T20:00:00+00:00"},
        )

    def test_unexpected_exception_does_not_retain_message(self) -> None:
        error = RuntimeError("password=private-do-not-retain")

        failure = normalize_exception(error)

        self.assertEqual(failure.kind, FailureKind.UNEXPECTED)
        self.assertEqual(failure.message, "unexpected implementation failure")
        self.assertEqual(
            failure.metadata,
            {"exception_type": "builtins.RuntimeError"},
        )
        self.assertNotIn("private", repr(failure))


if __name__ == "__main__":
    unittest.main()
