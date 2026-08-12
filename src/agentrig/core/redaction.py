"""Event redaction policies with a security-oriented default."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

from agentrig.core._validation import require_trimmed_string
from agentrig.core.events import Event, JsonValue

REDACTED_VALUE = "[REDACTED]"

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_KEY_SEPARATOR = re.compile(r"[^a-z0-9]+")
_BEARER_CREDENTIAL = re.compile(r"(?i)\bbearer\s+[^\s,;]{8,}")
_TOKEN_CREDENTIAL = re.compile(
    r"(?:\bsk-[A-Za-z0-9_-]{12,}\b|\bgh[pousr]_[A-Za-z0-9]{16,}\b|"
    r"\bAKIA[0-9A-Z]{16}\b)"
)
_CREDENTIAL_URI = re.compile(
    r"[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@",
    re.IGNORECASE,
)

_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "auth_token",
        "authorization",
        "bearer_token",
        "client_secret",
        "cookie",
        "credential",
        "credentials",
        "id_token",
        "password",
        "passwd",
        "private_key",
        "proxy_authorization",
        "refresh_token",
        "secret",
        "session_id",
        "set_cookie",
        "token",
    }
)


class RedactionPolicy(Protocol):
    """Transform an event before it crosses an observability boundary."""

    def redact(self, event: Event) -> Event:
        """Return a safe event without mutating the input."""
        ...


@dataclass(frozen=True, slots=True)
class NoOpRedactionPolicy:
    """Explicitly preserve an event; useful only at trusted boundaries."""

    def redact(self, event: Event) -> Event:
        return event


@dataclass(frozen=True, slots=True)
class SafeRedactionPolicy:
    """Redact sensitive keys and unmistakable credential-shaped strings."""

    additional_sensitive_keys: frozenset[str] = frozenset()
    replacement: str = REDACTED_VALUE

    def __post_init__(self) -> None:
        require_trimmed_string("redaction replacement", self.replacement)
        normalized_keys: set[str] = set()
        for key in self.additional_sensitive_keys:
            normalized = _normalize_key(
                require_trimmed_string("additional sensitive key", key)
            )
            if not normalized:
                raise ValueError(
                    "additional sensitive keys must contain a letter or number"
                )
            normalized_keys.add(normalized)
        object.__setattr__(
            self,
            "additional_sensitive_keys",
            frozenset(normalized_keys),
        )

    def redact(self, event: Event) -> Event:
        sensitive_keys = _SENSITIVE_KEYS | self.additional_sensitive_keys
        correlation = _redact_mapping(
            event.correlation,
            sensitive_keys=sensitive_keys,
            replacement=self.replacement,
        )
        attributes = _redact_mapping(
            event.attributes,
            sensitive_keys=sensitive_keys,
            replacement=self.replacement,
        )
        return Event(
            event_id=event.event_id,
            kind=event.kind,
            occurred_at=event.occurred_at,
            run_id=event.run_id,
            parent_run_id=event.parent_run_id,
            correlation=cast(Mapping[str, str], correlation),
            attributes=attributes,
            schema_version=event.schema_version,
        )


DEFAULT_REDACTION_POLICY: RedactionPolicy = SafeRedactionPolicy()


def _redact_mapping(
    value: Mapping[str, JsonValue],
    *,
    sensitive_keys: frozenset[str],
    replacement: str,
) -> dict[str, JsonValue]:
    redacted: dict[str, JsonValue] = {}
    for key, child in value.items():
        if _is_sensitive_key(key, sensitive_keys):
            redacted[key] = replacement
        else:
            redacted[key] = _redact_value(
                child,
                sensitive_keys=sensitive_keys,
                replacement=replacement,
            )
    return redacted


def _redact_value(
    value: JsonValue,
    *,
    sensitive_keys: frozenset[str],
    replacement: str,
) -> JsonValue:
    if isinstance(value, str):
        return replacement if _looks_like_credential(value) else value
    if isinstance(value, Mapping):
        return _redact_mapping(
            value,
            sensitive_keys=sensitive_keys,
            replacement=replacement,
        )
    if isinstance(value, (list, tuple)):
        return [
            _redact_value(
                child,
                sensitive_keys=sensitive_keys,
                replacement=replacement,
            )
            for child in value
        ]
    return value


def _normalize_key(key: str) -> str:
    with_boundaries = _CAMEL_BOUNDARY.sub("_", key)
    return _KEY_SEPARATOR.sub("_", with_boundaries.casefold()).strip("_")


def _is_sensitive_key(key: str, sensitive_keys: frozenset[str]) -> bool:
    normalized = _normalize_key(key)
    return any(
        normalized == candidate or normalized.endswith(f"_{candidate}")
        for candidate in sensitive_keys
    )


def _looks_like_credential(value: str) -> bool:
    return (
        _BEARER_CREDENTIAL.search(value) is not None
        or _TOKEN_CREDENTIAL.search(value) is not None
        or _CREDENTIAL_URI.search(value) is not None
        or (
            "-----BEGIN " in value
            and "PRIVATE KEY-----" in value
        )
    )
