from __future__ import annotations

import unittest
from datetime import UTC, datetime

from agentrig.core import (
    DEFAULT_REDACTION_POLICY,
    REDACTED_VALUE,
    Event,
    EventId,
    EventKind,
    NoOpRedactionPolicy,
    RunId,
    SafeRedactionPolicy,
)
from agentrig.core.events import JsonValue


def create_event(
    *,
    correlation: dict[str, str] | None = None,
    attributes: dict[str, object] | None = None,
) -> Event:
    return Event(
        event_id=EventId("event-1"),
        kind=EventKind.PROVIDER_CALL_STARTED,
        occurred_at=datetime(2026, 8, 12, 20, 0, tzinfo=UTC),
        run_id=RunId("run-1"),
        correlation=correlation or {},
        attributes=attributes or {},  # type: ignore[arg-type]
    )


class SafeRedactionPolicyTest(unittest.TestCase):
    def test_redacts_sensitive_keys_across_naming_styles(self) -> None:
        event = create_event(
            attributes={
                "api_key": "one",
                "api-key": "two",
                "providerApiKey": "three",
                "nested": {
                    "client_secret": "four",
                    "Authorization": "five",
                },
            }
        )

        redacted = SafeRedactionPolicy().redact(event)

        self.assertEqual(redacted.attributes["api_key"], REDACTED_VALUE)
        self.assertEqual(redacted.attributes["api-key"], REDACTED_VALUE)
        self.assertEqual(redacted.attributes["providerApiKey"], REDACTED_VALUE)
        self.assertEqual(
            redacted.attributes["nested"],
            {
                "client_secret": REDACTED_VALUE,
                "Authorization": REDACTED_VALUE,
            },
        )

    def test_redacts_correlation_values_by_sensitive_key(self) -> None:
        event = create_event(
            correlation={
                "request_id": "request-1",
                "provider_session_id": "private-session",
            }
        )

        redacted = SafeRedactionPolicy().redact(event)

        self.assertEqual(redacted.correlation["request_id"], "request-1")
        self.assertEqual(
            redacted.correlation["provider_session_id"],
            REDACTED_VALUE,
        )

    def test_redacts_credential_shaped_strings_under_benign_keys(self) -> None:
        event = create_event(
            attributes={
                "error": "request used Bearer abcdefghijklmnop",
                "endpoint": "https://user:password@example.com/path",
                "detail": "key sk-abcdefghijklmnop",
                "pem": "-----BEGIN PRIVATE KEY-----\nabc",
            }
        )

        redacted = SafeRedactionPolicy().redact(event)

        self.assertEqual(
            redacted.attributes,
            {
                "error": REDACTED_VALUE,
                "endpoint": REDACTED_VALUE,
                "detail": REDACTED_VALUE,
                "pem": REDACTED_VALUE,
            },
        )

    def test_preserves_benign_values_and_token_usage(self) -> None:
        event = create_event(
            attributes={
                "input_tokens": 120,
                "output_tokens": 30,
                "message": "draft completed",
                "model": "example-model",
            }
        )

        redacted = SafeRedactionPolicy().redact(event)

        self.assertEqual(redacted.attributes, event.attributes)

    def test_returns_new_event_without_mutating_envelope_or_input(self) -> None:
        attributes: dict[str, object] = {"password": "private"}
        event = create_event(attributes=attributes)

        redacted = SafeRedactionPolicy().redact(event)

        self.assertIsNot(redacted, event)
        self.assertEqual(redacted.event_id, event.event_id)
        self.assertEqual(redacted.kind, event.kind)
        self.assertEqual(redacted.occurred_at, event.occurred_at)
        self.assertEqual(redacted.run_id, event.run_id)
        self.assertEqual(event.attributes["password"], "private")
        self.assertEqual(attributes["password"], "private")

    def test_redacts_standalone_json_objects_without_mutating_input(self) -> None:
        value: dict[str, JsonValue] = {
            "environment": {
                "api_key": "private",
                "endpoint": "https://user:password@example.com/path",
            },
            "status": "complete",
        }

        redacted = SafeRedactionPolicy().redact_json_object(value)

        self.assertEqual(
            redacted,
            {
                "environment": {
                    "api_key": REDACTED_VALUE,
                    "endpoint": REDACTED_VALUE,
                },
                "status": "complete",
            },
        )
        environment = value["environment"]
        if not isinstance(environment, dict):
            raise AssertionError("environment was not a JSON object")
        self.assertEqual(environment["api_key"], "private")

    def test_additional_keys_and_replacement_are_configurable(self) -> None:
        policy = SafeRedactionPolicy(
            additional_sensitive_keys=frozenset({"internalCode"}),
            replacement="<hidden>",
        )
        event = create_event(attributes={"internal_code": "alpha"})

        redacted = policy.redact(event)

        self.assertEqual(redacted.attributes["internal_code"], "<hidden>")

    def test_configuration_rejects_empty_values(self) -> None:
        with self.assertRaises(ValueError):
            SafeRedactionPolicy(replacement="")
        with self.assertRaises(ValueError):
            SafeRedactionPolicy(
                additional_sensitive_keys=frozenset({"---"})
            )

    def test_noop_policy_must_be_explicit(self) -> None:
        event = create_event(attributes={"password": "private"})

        preserved = NoOpRedactionPolicy().redact(event)
        safely_redacted = DEFAULT_REDACTION_POLICY.redact(event)

        self.assertIs(preserved, event)
        self.assertEqual(safely_redacted.attributes["password"], REDACTED_VALUE)


if __name__ == "__main__":
    unittest.main()
