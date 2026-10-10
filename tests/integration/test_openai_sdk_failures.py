from __future__ import annotations

import asyncio
from types import SimpleNamespace
import unittest

from agentrig.core import AgentRigError, FailureKind
from agentrig.integrations.openai.images_sdk import (
    _request_failure as image_failure,
    _SdkImageClient,
)
from agentrig.integrations.openai.responses_sdk import (
    _request_failure as responses_failure,
    _SdkClient,
)
from agentrig.integrations.openai import (
    OpenAIImageOperation,
    OpenAIImageRequest,
    OpenAIResponsesMessage,
    OpenAIResponsesRequest,
)
from agentrig.capabilities import TextMessageRole


_NORMALIZERS = (image_failure, responses_failure)


def _error(**fields: object) -> Exception:
    error = RuntimeError("synthetic-private-message")
    for name, value in fields.items():
        setattr(error, name, value)
    return error


class OpenAISdkFailureTest(unittest.TestCase):
    def test_official_sdk_rate_limit_error_reaches_both_bridges_once(self) -> None:
        from openai import RateLimitError

        # Construct the official exception offline; the SDK extracts the safe
        # fields from this response and error body before either bridge sees it.
        error = RateLimitError(
            "synthetic-private-message",
            response=SimpleNamespace(
                status_code=429, request=object(),
                headers={"x-request-id": "req_fixture_123", "retry-after": "2.500"},
            ),
            body={"code": "rate_limit_exceeded", "type": "tokens",
                  "message": "synthetic-private-body", "param": "private-prompt"},
        )
        for bridge, request in (
            (_SdkImageClient, OpenAIImageRequest(
                operation=OpenAIImageOperation.GENERATE,
                model="fixture", prompt="synthetic", width=1024, height=1024,
                output_media_type="image/png",
            )),
            (_SdkClient, OpenAIResponsesRequest(
                model="fixture",
                messages=(OpenAIResponsesMessage(role=TextMessageRole.USER, text="synthetic"),),
                schema_name="fixture", output_schema={"type": "object"},
                max_output_tokens=20,
            )),
        ):
            with self.subTest(bridge=bridge.__name__):
                raw = _RawClient(error)
                client = bridge(raw)
                try:
                    with self.assertRaises(AgentRigError) as raised:
                        asyncio.run(client.create(request))
                finally:
                    asyncio.run(client.close())
                failure = raised.exception.failure
                self.assertEqual(failure.kind, FailureKind.TRANSIENT_PROVIDER)
                self.assertEqual(dict(failure.metadata), {
                    "status_code": "429", "provider_code": "rate_limit_exceeded",
                    "provider_type": "tokens", "provider_request_id": "req_fixture_123",
                    "retry_after_seconds": "2.5",
                })
                self.assertEqual(raw.calls, 1)
                self.assertTrue(raw.closed)
                expected_code = (
                    "openai.image.request_failed"
                    if bridge is _SdkImageClient else "openai.responses.request_failed"
                )
                self.assertEqual(failure.code, expected_code)
                self.assertNotIn("private", repr(failure))
                self.assertNotIn("private", str(raised.exception))

    def test_known_quota_is_permanent_and_ordinary_status_behavior_is_preserved(self) -> None:
        cases = [
            ({"status_code": 429, "code": code}, FailureKind.PERMANENT_PROVIDER)
            for code in (
                "insufficient_quota", "billing_hard_limit_reached", "billing_not_active",
                "credit_balance_exhausted", "organization_spend_limit_exceeded",
                "project_spend_limit_exceeded", "organization_usage_limit_exceeded",
            )
        ] + [
            ({"status_code": 429, "type": "insufficient_quota"}, FailureKind.PERMANENT_PROVIDER),
            ({"status_code": 429, "code": "rate_limit_exceeded"}, FailureKind.TRANSIENT_PROVIDER),
            ({"status_code": 429, "code": "slow_down"}, FailureKind.TRANSIENT_PROVIDER),
            ({"status_code": 429}, FailureKind.TRANSIENT_PROVIDER),
            ({"status_code": 503}, FailureKind.TRANSIENT_PROVIDER),
            ({"status_code": 400}, FailureKind.PERMANENT_PROVIDER),
            ({"status_code": 401}, FailureKind.PERMANENT_PROVIDER),
            ({"status_code": 100}, FailureKind.PERMANENT_PROVIDER),
            ({"status_code": 599}, FailureKind.TRANSIENT_PROVIDER),
            ({"status_code": 99}, FailureKind.TRANSIENT_PROVIDER),
            ({"status_code": 600}, FailureKind.TRANSIENT_PROVIDER),
            ({"status_code": -1}, FailureKind.TRANSIENT_PROVIDER),
            ({"status_code": 10 ** 50}, FailureKind.TRANSIENT_PROVIDER),
            ({"status_code": True}, FailureKind.TRANSIENT_PROVIDER),
            ({"status_code": "429"}, FailureKind.TRANSIENT_PROVIDER),
            ({}, FailureKind.TRANSIENT_PROVIDER),
        ]
        for normalize in _NORMALIZERS:
            for fields, expected in cases:
                with self.subTest(normalizer=normalize.__module__, fields=fields):
                    failure = normalize(_error(**fields))
                    self.assertEqual(failure.kind, expected)
                    self.assertEqual(failure.is_retryable, expected is FailureKind.TRANSIENT_PROVIDER)
                    status = fields.get("status_code")
                    self.assertEqual(
                        "status_code" in failure.metadata,
                        type(status) is int and 100 <= status <= 599,
                    )
                    if "code" in fields:
                        self.assertEqual(failure.metadata.get("provider_code"), fields["code"])

    def test_explicit_code_takes_precedence_over_conflicting_quota_type(self) -> None:
        for normalize in _NORMALIZERS:
            for code in ("rate_limit_exceeded", "slow_down", "server_is_overloaded"):
                with self.subTest(normalizer=normalize.__module__, code=code):
                    failure = normalize(_error(
                        status_code=429, code=code, type="insufficient_quota",
                    ))
                    self.assertEqual(failure.kind, FailureKind.TRANSIENT_PROVIDER)
                    self.assertEqual(failure.metadata["provider_code"], code)
                    self.assertEqual(failure.metadata["provider_type"], "insufficient_quota")

    def test_current_overload_diagnostics_are_retained(self) -> None:
        for normalize in _NORMALIZERS:
            failure = normalize(_error(
                status_code=503, code="server_is_overloaded", type="service_unavailable_error",
            ))
            self.assertEqual(failure.kind, FailureKind.TRANSIENT_PROVIDER)
            self.assertEqual(dict(failure.metadata), {
                "status_code": "503", "provider_code": "server_is_overloaded",
                "provider_type": "service_unavailable_error",
            })

    def test_unknown_and_malicious_diagnostics_do_not_leak_or_reclassify_429(self) -> None:
        for normalize in _NORMALIZERS:
            for value in (
                "private", "unknown_future_code", "insufficient_quota\nprivate",
                "rate_limit_exceeded private", {"code": "insufficient_quota"}, None,
            ):
                with self.subTest(normalizer=normalize.__module__, value=value):
                    failure = normalize(_error(
                        status_code=429, code=value, type=value, request_id=value,
                        body={"code": "insufficient_quota", "message": "private"},
                        response=SimpleNamespace(headers={
                            "authorization": "private", "x-request-id": "private",
                            "retry-after": "Wed, 21 Oct 2026 07:28:00 GMT",
                        }),
                    ))
                    self.assertEqual(dict(failure.metadata), {"status_code": "429"})
                    self.assertEqual(failure.kind, FailureKind.TRANSIENT_PROVIDER)
                    self.assertNotIn("private", repr(failure))

    def test_request_ids_require_a_bounded_known_shape(self) -> None:
        for normalize in _NORMALIZERS:
            for value, valid in (
                ("req_abc-123_DEF", True),
                ("01234567-89ab-cdef-0123-456789abcdef", True),
                ("01234567-89AB-CDEF-0123-456789ABCDEF", True),
                ("req_" + "a" * 100, True),
                ("req_" + "a" * 101, False),
                ("req_", False), ("req_private\n", False),
                ("req_private@example.test", False), ("private", False),
            ):
                with self.subTest(normalizer=normalize.__module__, value=value):
                    failure = normalize(_error(request_id=value))
                    expected = {"provider_request_id": value} if valid else {}
                    self.assertEqual(dict(failure.metadata), expected)

    def test_retry_headers_are_numeric_bounded_and_normalized(self) -> None:
        cases = [
            ({"Retry-After": "2.500000"}, "2.5"),
            ({"retry-after": "0"}, "0"),
            ({"retry-after": "3600"}, "3600"),
            ({"retry-after-ms": "1250", "retry-after": "99"}, "1.25"),
            ({"retry-after-ms": "0.001"}, "0.000001"),
            ({"retry-after-ms": "invalid", "retry-after": "2"}, "2"),
        ] + [({"retry-after": value}, None) for value in (
            "-1", "3600.1", "NaN", "Infinity", "1e2", " 2", "2\n",
            "0.0000001", "9" * 33, "private", 3, True, None,
        )] + [({"retry-after-ms": value}, None) for value in (
            "3600001", "0.0001", "-100", "NaN",
        )]
        for normalize in _NORMALIZERS:
            for headers, expected in cases:
                with self.subTest(normalizer=normalize.__module__, headers=headers):
                    failure = normalize(_error(response=SimpleNamespace(headers=headers)))
                    metadata = {"retry_after_seconds": expected} if expected is not None else {}
                    self.assertEqual(dict(failure.metadata), metadata)

    def test_throwing_provider_properties_and_header_access_are_ignored(self) -> None:
        class BrokenError(Exception):
            def __getattribute__(self, name: str) -> object:
                raise ValueError("private-property")

        class BrokenHeaders(dict[str, str]):
            def get(self, key: object, default: object = None) -> object:
                raise ValueError("private-header")

        for normalize in _NORMALIZERS:
            for error in (
                BrokenError(), _error(response=SimpleNamespace(headers=BrokenHeaders())),
            ):
                with self.subTest(normalizer=normalize.__module__):
                    failure = normalize(error)
                    self.assertEqual(dict(failure.metadata), {})
                    self.assertEqual(failure.kind, FailureKind.TRANSIENT_PROVIDER)


class _RawClient:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.images = self
        self.responses = self
        self.calls = 0
        self.closed = False

    async def create(self, **kwargs: object) -> object:
        self.calls += 1
        raise self.error

    async def generate(self, **kwargs: object) -> object:
        return await self.create(**kwargs)

    async def edit(self, **kwargs: object) -> object:
        return await self.create(**kwargs)

    async def close(self) -> None:
        self.closed = True


if __name__ == "__main__":
    unittest.main()
