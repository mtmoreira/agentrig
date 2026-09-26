"""Explicit application-owned private capture, separate from safe events."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agentrig.core._json import JsonValue, freeze_json_object, thaw_json_value
from agentrig.core._validation import freeze_string_map, require_trimmed_string
from agentrig.core.errors import AgentRigError, Failure, FailureKind
from agentrig.core.identity import RunId
from agentrig.core.redaction import DEFAULT_REDACTION_POLICY, REDACTED_VALUE

if TYPE_CHECKING:
    from agentrig.core.context import RunContext

_KINDS = frozenset({"provider.request", "provider.response", "process.request", "process.response"})
_METADATA_KEYS = frozenset({"provider", "model", "output_schema", "prompt_version", "agent_id"})
_TEXT_INSPECTION_LIMIT = 4 * 1024 * 1024
_SECRET_NAME = (r"(?:api[-_]?key|(?:access|refresh|auth|bearer|session|id)[-_]?token|"
                r"token|password|passwd|pwd|passphrase|credentials?|client[-_]?secret|secret|"
                r"(?:proxy[-_])?authorization|(?:set[-_])?cookie|private[-_]?key)")
_SECRET_FIELD = r"(?:[A-Za-z][A-Za-z0-9]*[-_]){0,6}" + _SECRET_NAME
_SECRET_OPTION = re.compile(r"(?i)" + _SECRET_FIELD + r"\Z")
_QUOTED_OR_TOKEN = r'''(?:\[REDACTED\]|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,;&}\]]+)'''
_SECRET_ASSIGNMENT = re.compile(
    r'''(?i)(?P<prefix>(?<![A-Za-z0-9_])["']?''' + _SECRET_FIELD +
    r'''["']?\s*[:=]\s*)''' + _QUOTED_OR_TOKEN)
_SECRET_FLAG = re.compile(
    r"(?i)(?P<prefix>(?<!\w)--" + _SECRET_FIELD + r"(?:=|\s+))" + _QUOTED_OR_TOKEN)
_AUTH_VALUE = re.compile(
    r"(?i)(?P<prefix>\b(?:bearer|basic)\s+)[^\s,;\"'\\]+")
_COOKIE_VALUE = re.compile(
    r"(?i)(?P<prefix>\b(?:set[-_])?cookie\s*[:=]\s*)[^\r\n]*")
_PRIVATE_KEY = re.compile(
    r"(?is)-----BEGIN[ \t]+(?P<label>(?:[A-Z0-9]+[ \t]+){0,4}PRIVATE[ \t]+KEY)-----"
    r".*?(?:-----END[ \t]+(?P=label)-----|\Z)")


@dataclass(frozen=True, slots=True, kw_only=True)
class PrivateTraceRecord:
    """A bounded private JSON-text snapshot, never an ordinary event.

    ``content_bytes`` is the full redacted UTF-8 length before truncation.
    Truncated content is a UTF-8 prefix and need not be valid JSON.
    """

    record_id: str
    kind: str
    occurred_at: datetime
    run_id: RunId
    parent_run_id: RunId | None
    content: str = field(repr=False)
    content_bytes: int
    truncated: bool
    metadata: Mapping[str, str] = field(default_factory=dict, repr=False)
    correlation: Mapping[str, str] = field(default_factory=dict, repr=False)
    schema_version: int = 1

    def __post_init__(self) -> None:
        require_trimmed_string("private trace record ID", self.record_id)
        if self.kind not in _KINDS:
            raise ValueError("private trace kind is unsupported")
        if not isinstance(self.occurred_at, datetime) or self.occurred_at.utcoffset() is None:
            raise ValueError("private trace timestamp must be timezone-aware")
        if not isinstance(self.run_id, RunId) or (
            self.parent_run_id is not None and not isinstance(self.parent_run_id, RunId)
        ):
            raise TypeError("private trace lineage is invalid")
        if not isinstance(self.content, str):
            raise TypeError("private trace content must be text")
        size = len(self.content.encode("utf-8"))
        if type(self.content_bytes) is not int or self.content_bytes < size:
            raise ValueError("private trace content size is invalid")
        if type(self.truncated) is not bool or self.truncated != (self.content_bytes > size):
            raise ValueError("private trace truncation is inconsistent")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("private trace schema version is unsupported")
        if size > 4 * 1024 * 1024:
            raise ValueError("private trace exceeds the maximum byte bound")
        metadata = freeze_string_map("private trace metadata", self.metadata)
        if any(key not in _METADATA_KEYS or len(value) > 256 or any(ord(char) < 32 for char in value)
               for key, value in metadata.items()):
            raise ValueError("private trace metadata is invalid")
        object.__setattr__(self, "metadata", freeze_string_map("private trace metadata", {
            key: str(value) for key, value in redact_private_trace_content(metadata).items()
        }))
        correlation = freeze_string_map("private trace correlation", self.correlation)
        object.__setattr__(self, "correlation", freeze_string_map("private trace correlation", {
            key: str(value) for key, value in redact_private_trace_content(correlation).items()
        }))
        object.__setattr__(self, "occurred_at", self.occurred_at.astimezone(UTC))


@runtime_checkable
class PrivateTraceRecorder(Protocol):
    """Application-owned private storage; never forward to public logging."""

    def record(self, record: PrivateTraceRecord) -> None: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class PrivateTraceCapture:
    """Opt in to bounded private provider content; credential redaction remains on."""

    recorder: PrivateTraceRecorder = field(repr=False)
    max_content_bytes: int = 262144
    include_reasoning: bool = False
    include_process: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.recorder, PrivateTraceRecorder):
            raise TypeError("private trace recorder is invalid")
        if type(self.max_content_bytes) is not int or not 1 <= self.max_content_bytes <= 4 * 1024 * 1024:
            raise ValueError("private trace byte limit is invalid")
        if type(self.include_reasoning) is not bool:
            raise TypeError("private trace reasoning choice must be boolean")
        if type(self.include_process) is not bool:
            raise TypeError("private trace process choice must be boolean")

    def record(self, context: RunContext, *, kind: str,
               content: Mapping[str, JsonValue], metadata: Mapping[str, str]) -> None:
        """Freeze, redact and bound one snapshot before invoking private storage."""
        redacted = redact_private_trace_content(content)
        encoded = json.dumps(thaw_json_value(redacted), ensure_ascii=False,
                             allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        prefix = encoded[:self.max_content_bytes].decode("utf-8", errors="ignore")
        record = PrivateTraceRecord(
            record_id=str(context.event_id_generator.generate()), kind=kind,
            occurred_at=context.clock.now(), run_id=context.run_id,
            parent_run_id=context.parent_run_id, content=prefix,
            content_bytes=len(encoded), truncated=len(encoded) > self.max_content_bytes,
            metadata=metadata, correlation=context.correlation,
        )
        try:
            self.recorder.record(record)
        except Exception:
            raise AgentRigError(Failure(
                kind=FailureKind.UNEXPECTED, code="private_trace.recording_failed",
                message="Private execution trace could not be recorded",
            )) from None


def redact_private_trace_content(content: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    """Return a detached private-capture snapshot with embedded secrets scrubbed.

    This strengthens private capture only. Ordinary lifecycle-event redaction is
    unchanged. Large individual strings are omitted rather than partly inspected.
    """
    value = freeze_json_object("private trace content", content)
    scrubbed = {key: _scrub_private_value(item, key=key) for key, item in value.items()}
    return freeze_json_object("redacted private trace content",
                              DEFAULT_REDACTION_POLICY.redact_json_object(scrubbed))


def redact_private_trace_arguments(arguments: Sequence[str]) -> tuple[str, ...]:
    """Redact credential switches and inline assignments in a private argv view."""
    if isinstance(arguments, (str, bytes)) or any(not isinstance(value, str) for value in arguments):
        raise TypeError("private trace arguments must be a sequence of text")
    if len(arguments) > 4096 or sum(len(value) for value in arguments) > _TEXT_INSPECTION_LIMIT:
        return ("[REDACTED: argument list exceeds inspection bound]",)
    result: list[str] = []
    redact_next = False
    for argument in arguments:
        if redact_next:
            result.append(REDACTED_VALUE)
            redact_next = False
            continue
        if argument.startswith('-'):
            if len(argument) > 2 and argument[:2] in {"-p", "-u"} and not argument.startswith('--'):
                result.append(argument[:2] + REDACTED_VALUE)
                continue
            option, separator, value = argument.partition('=')
            key = option.lstrip('-')
            probe = DEFAULT_REDACTION_POLICY.redact_json_object({key: "value"})[key]
            if (probe != "value" or _SECRET_OPTION.fullmatch(key) is not None or
                key.casefold() in {"p", "u", "pass", "user", "proxy-user", "auth"}):
                result.append(option + '=' + REDACTED_VALUE if separator else option)
                redact_next = not separator
                continue
        result.append(_scrub_private_text(argument))
    return tuple(result)


def _scrub_private_value(value: JsonValue, *, key: str = "") -> JsonValue:
    if _SECRET_OPTION.fullmatch(key) is not None:
        return REDACTED_VALUE
    if isinstance(value, str):
        return _scrub_private_text(value)
    if isinstance(value, Mapping):
        return {name: _scrub_private_value(item, key=name) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        if key == "argv" and all(isinstance(item, str) for item in value):
            return redact_private_trace_arguments(tuple(str(item) for item in value))
        return tuple(_scrub_private_value(item) for item in value)
    return value


def _scrub_private_text(value: str) -> str:
    if len(value) > _TEXT_INSPECTION_LIMIT:
        return "[REDACTED: private text exceeds inspection bound]"
    result = _PRIVATE_KEY.sub(REDACTED_VALUE, value)
    result = _AUTH_VALUE.sub(lambda match: match.group("prefix") + REDACTED_VALUE, result)
    result = _COOKIE_VALUE.sub(lambda match: match.group("prefix") + REDACTED_VALUE, result)
    result = _SECRET_ASSIGNMENT.sub(lambda match: match.group("prefix") + REDACTED_VALUE, result)
    return _SECRET_FLAG.sub(lambda match: match.group("prefix") + REDACTED_VALUE, result)
