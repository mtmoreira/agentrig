"""Portable references to artifacts stored outside AgentRig core."""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit

from agentrig.core._validation import freeze_string_map, require_trimmed_string
from agentrig.core.identity import RunId

_KIND_PATTERN = re.compile(r"[a-z][a-z0-9._-]*")
_DIGEST_ALGORITHM_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*")


@dataclass(frozen=True, order=True, slots=True)
class ArtifactId:
    """Opaque, serializable identity for one artifact."""

    value: str

    def __post_init__(self) -> None:
        require_trimmed_string("artifact ID", self.value)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ContentDigest:
    """A content digest with an explicit algorithm."""

    algorithm: str
    value: str

    def __post_init__(self) -> None:
        require_trimmed_string("digest algorithm", self.algorithm)
        require_trimmed_string("digest value", self.value)
        if _DIGEST_ALGORITHM_PATTERN.fullmatch(self.algorithm) is None:
            raise ValueError("digest algorithm must be a lowercase identifier")
        if any(character.isspace() for character in self.value):
            raise ValueError("digest value must not contain whitespace")

    def __str__(self) -> str:
        return f"{self.algorithm}:{self.value}"


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactRef:
    """Storage-independent identity, location, lineage, and handling metadata."""

    artifact_id: ArtifactId
    kind: str
    media_type: str
    producer_run_id: RunId
    uri: str | None = None
    workspace_path: str | None = None
    content_digest: ContentDigest | None = None
    input_artifact_ids: tuple[ArtifactId, ...] = ()
    labels: Mapping[str, str] = field(default_factory=dict)
    provider_lineage: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_trimmed_string("artifact kind", self.kind)
        if _KIND_PATTERN.fullmatch(self.kind) is None:
            raise ValueError("artifact kind must be a lowercase identifier")

        require_trimmed_string("media type", self.media_type)
        media_type_essence = self.media_type.partition(";")[0]
        media_type_parts = media_type_essence.split("/")
        if (
            len(media_type_parts) != 2
            or not all(media_type_parts)
            or any(character.isspace() for character in media_type_essence)
        ):
            raise ValueError("media type must contain a valid type and subtype")

        if (self.uri is None) == (self.workspace_path is None):
            raise ValueError(
                "artifact must have exactly one URI or workspace-relative path"
            )
        if self.uri is not None:
            _validate_uri(self.uri)
        if self.workspace_path is not None:
            _validate_workspace_path(self.workspace_path)

        copied_inputs = tuple(self.input_artifact_ids)
        if len(copied_inputs) != len(set(copied_inputs)):
            raise ValueError("input artifact IDs must not contain duplicates")
        if self.artifact_id in copied_inputs:
            raise ValueError("artifact must not list itself as an input")

        object.__setattr__(self, "input_artifact_ids", copied_inputs)
        object.__setattr__(
            self,
            "labels",
            freeze_string_map("artifact labels", self.labels),
        )
        object.__setattr__(
            self,
            "provider_lineage",
            freeze_string_map("provider lineage", self.provider_lineage),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedArtifact:
    """Private artifact bytes detached from a storage implementation."""

    artifact: ArtifactRef
    content: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ArtifactRef):
            raise TypeError("resolved artifact must reference an ArtifactRef")
        if not isinstance(self.content, bytes) or not self.content:
            raise ValueError("resolved artifact content must be nonempty bytes")
        expected = self.artifact.content_digest
        if expected is not None:
            try:
                actual = hashlib.new(expected.algorithm, self.content).hexdigest()
            except ValueError:
                raise ValueError(
                    "resolved artifact digest algorithm is unsupported"
                ) from None
            if not hmac.compare_digest(actual, expected.value.lower()):
                raise ValueError("resolved artifact content digest does not match")


@runtime_checkable
class ArtifactResolver(Protocol):
    """Resolve one portable reference without exposing storage to providers."""

    async def resolve(self, artifact: ArtifactRef) -> ResolvedArtifact:
        """Return bytes for exactly the requested artifact reference."""
        ...


def _validate_uri(uri: str) -> None:
    require_trimmed_string("artifact URI", uri)
    if any(character.isspace() for character in uri):
        raise ValueError("artifact URI must not contain whitespace")
    if not urlsplit(uri).scheme:
        raise ValueError("artifact URI must include a scheme")


def _validate_workspace_path(workspace_path: str) -> None:
    require_trimmed_string("artifact workspace path", workspace_path)
    if "\\" in workspace_path or "\x00" in workspace_path:
        raise ValueError("artifact workspace path must use safe POSIX syntax")

    parsed = PurePosixPath(workspace_path)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError("artifact workspace path must remain inside the workspace")
    if parsed.as_posix() in ("", ".") or parsed.as_posix() != workspace_path:
        raise ValueError("artifact workspace path must be canonical")
