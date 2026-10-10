"""Allowlisted diagnostics shared by the optional OpenAI SDK bridges."""

from __future__ import annotations

from collections.abc import Mapping
import re

from agentrig.core.errors import Failure, FailureKind


_QUOTA_CODES = frozenset({
    "insufficient_quota", "billing_hard_limit_reached", "billing_not_active",
    "credit_balance_exhausted", "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded", "organization_usage_limit_exceeded",
})
_PROVIDER_CODES = _QUOTA_CODES | {
    "rate_limit_exceeded", "slow_down", "invalid_api_key", "model_not_found",
    "content_policy_violation", "moderation_blocked", "server_error",
    "invalid_request_error", "unsupported_parameter", "server_is_overloaded",
}
_PROVIDER_TYPES = frozenset({
    "insufficient_quota", "rate_limit_error", "requests", "tokens",
    "invalid_request_error", "server_error", "api_error", "service_unavailable_error",
})
_REQUEST_ID = re.compile(
    r"(?:req_[A-Za-z0-9_-]{1,100}|"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)
_RETRY_NUMBER = re.compile(r"[0-9]+(?:\.[0-9]{1,6})?")


def request_failure(error: Exception, *, message: str, code: str) -> Failure:
    """Retain known diagnostics without reading messages or response bodies."""
    metadata: dict[str, str] = {}
    status = _attribute(error, "status_code")
    kind = FailureKind.TRANSIENT_PROVIDER
    if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599:
        metadata["status_code"] = str(status)
        if status != 429 and status < 500:
            kind = FailureKind.PERMANENT_PROVIDER

    provider_code = _attribute(error, "code")
    provider_type = _attribute(error, "type")
    if isinstance(provider_code, str) and provider_code in _PROVIDER_CODES:
        metadata["provider_code"] = provider_code
    if isinstance(provider_type, str) and provider_type in _PROVIDER_TYPES:
        metadata["provider_type"] = provider_type
    if (
        metadata.get("provider_code") in _QUOTA_CODES
        or (
            "provider_code" not in metadata
            and metadata.get("provider_type") == "insufficient_quota"
        )
    ):
        kind = FailureKind.PERMANENT_PROVIDER

    request_id = _attribute(error, "request_id")
    if isinstance(request_id, str) and _REQUEST_ID.fullmatch(request_id):
        metadata["provider_request_id"] = request_id
    headers = _attribute(_attribute(error, "response"), "headers")
    if isinstance(headers, Mapping):
        # The SDK gives retry-after-ms precedence; never copy the header map.
        retry = _retry_seconds(_header(headers, "retry-after-ms"), milliseconds=True)
        if retry is None:
            retry = _retry_seconds(_header(headers, "retry-after"), milliseconds=False)
        if retry is not None:
            metadata["retry_after_seconds"] = retry
    return Failure(kind=kind, message=message, code=code, metadata=metadata)


def _attribute(value: object, name: str) -> object:
    try:
        return getattr(value, name, None)
    except Exception:
        return None


def _header(headers: Mapping[object, object], name: str) -> object:
    try:
        # Real SDK headers are case insensitive. Support common plain mappings
        # at the injected seam without inspecting unrelated header fields.
        for key in (name, name.title(), name.upper()):
            value = headers.get(key)
            if value is not None:
                return value
    except Exception:
        pass
    return None


def _retry_seconds(value: object, *, milliseconds: bool) -> str | None:
    if (
        not isinstance(value, str)
        or len(value) > 32
        or _RETRY_NUMBER.fullmatch(value) is None
    ):
        return None
    whole, _, fraction = value.partition(".")
    microseconds = int(whole) * 1_000_000 + int(fraction.ljust(6, "0"))
    if milliseconds:
        if microseconds % 1000:
            return None
        microseconds //= 1000
    if microseconds > 3_600_000_000:
        return None
    seconds, remainder = divmod(microseconds, 1_000_000)
    return f"{seconds}.{remainder:06d}".rstrip("0").rstrip(".")
