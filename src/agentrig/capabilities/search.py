"""Provider-independent bounded search contracts."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit

from agentrig.capabilities.base import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    CapabilityRequirements,
)
from agentrig.core._validation import require_trimmed_string
from agentrig.core.context import RunContext


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchRequest:
    """One private search query with a hard result-count bound."""

    query: str = field(repr=False)
    max_results: int = 10

    def __post_init__(self) -> None:
        _require_content_text("search query", self.query)
        _require_positive_integer("search max_results", self.max_results)

    @property
    def requirements(self) -> CapabilityRequirements:
        """Derive citation and result-capacity requirements."""
        return CapabilityRequirements(
            kind=CapabilityKind.SEARCH,
            features=frozenset({CapabilityFeature.CITATIONS}),
            minimum_limits={
                CapabilityLimit.MAX_RESULTS: self.max_results,
            },
        )

    def require_supported_by(self, descriptor: CapabilityDescriptor) -> None:
        """Fail before provider execution if this request is unsupported."""
        self.requirements.require(descriptor)


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchCitation:
    """Citation-ready identity for one returned source."""

    source_uri: str = field(repr=False)
    title: str = field(repr=False)

    def __post_init__(self) -> None:
        _require_source_uri("search citation source_uri", self.source_uri)
        _require_content_text("search citation title", self.title)


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchHit:
    """One source-bearing search result with excerpt or summary text."""

    source_uri: str = field(repr=False)
    title: str = field(repr=False)
    excerpt: str | None = field(default=None, repr=False)
    summary: str | None = field(default=None, repr=False)
    published_at: datetime | None = None

    def __post_init__(self) -> None:
        _require_source_uri("search hit source_uri", self.source_uri)
        _require_content_text("search hit title", self.title)
        if self.excerpt is not None:
            _require_content_text("search hit excerpt", self.excerpt)
        if self.summary is not None:
            _require_content_text("search hit summary", self.summary)
        if self.excerpt is None and self.summary is None:
            raise ValueError("search hit requires an excerpt or summary")
        if self.published_at is not None:
            if not isinstance(self.published_at, datetime):
                raise TypeError("search hit published_at must be a datetime or None")
            if (
                self.published_at.tzinfo is None
                or self.published_at.utcoffset() is None
            ):
                raise ValueError("search hit published_at must be timezone-aware")
            object.__setattr__(
                self,
                "published_at",
                self.published_at.astimezone(UTC),
            )

    @property
    def citation(self) -> SearchCitation:
        """Return the citation identity preserved by this hit."""
        return SearchCitation(source_uri=self.source_uri, title=self.title)


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchRetrievalMetadata:
    """Portable timing and result-availability metadata for one search."""

    retrieved_at: datetime
    duration_seconds: float | None = None
    total_available: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.retrieved_at, datetime):
            raise TypeError("search retrieved_at must be a datetime")
        if (
            self.retrieved_at.tzinfo is None
            or self.retrieved_at.utcoffset() is None
        ):
            raise ValueError("search retrieved_at must be timezone-aware")
        object.__setattr__(
            self,
            "retrieved_at",
            self.retrieved_at.astimezone(UTC),
        )
        if self.duration_seconds is not None:
            duration = _require_non_negative_number(
                "search duration_seconds",
                self.duration_seconds,
            )
            object.__setattr__(self, "duration_seconds", duration)
        if self.total_available is not None:
            _require_non_negative_integer(
                "search total_available",
                self.total_available,
            )


@dataclass(frozen=True, slots=True, init=False)
class SearchResult:
    """A bounded ordered result set with citations and retrieval metadata."""

    hits: tuple[SearchHit, ...] = field(repr=False)
    metadata: SearchRetrievalMetadata

    def __init__(
        self,
        *,
        request: SearchRequest,
        hits: Iterable[SearchHit],
        metadata: SearchRetrievalMetadata,
    ) -> None:
        if not isinstance(request, SearchRequest):
            raise TypeError("search result request must be a SearchRequest")
        copied_hits = tuple(hits)
        if any(not isinstance(hit, SearchHit) for hit in copied_hits):
            raise TypeError("search result hits must contain SearchHit values")
        if len(copied_hits) > request.max_results:
            raise ValueError("search result exceeds the request max_results")
        source_uris = tuple(hit.source_uri for hit in copied_hits)
        if len(source_uris) != len(set(source_uris)):
            raise ValueError("search result source URIs must be unique")
        if not isinstance(metadata, SearchRetrievalMetadata):
            raise TypeError(
                "search result metadata must be SearchRetrievalMetadata"
            )
        if (
            metadata.total_available is not None
            and metadata.total_available < len(copied_hits)
        ):
            raise ValueError(
                "search total_available must cover every returned hit"
            )
        object.__setattr__(self, "hits", copied_hits)
        object.__setattr__(self, "metadata", metadata)

    @property
    def citations(self) -> tuple[SearchCitation, ...]:
        """Return citations in the same rank order as the result hits."""
        return tuple(hit.citation for hit in self.hits)


@runtime_checkable
class SearchProvider(Protocol):
    """Search through one provider-independent implementation."""

    @property
    def descriptor(self) -> CapabilityDescriptor:
        """Return stable identity and supported optional features."""
        ...

    async def search(
        self,
        request: SearchRequest,
        context: RunContext,
    ) -> SearchResult:
        """Return one bounded result set or raise a normalized failure."""
        ...


def _require_source_uri(field_name: str, value: object) -> str:
    uri = require_trimmed_string(field_name, value)
    if any(character.isspace() for character in uri):
        raise ValueError(f"{field_name} must not contain whitespace")
    if not urlsplit(uri).scheme:
        raise ValueError(f"{field_name} must include a scheme")
    return uri


def _require_positive_integer(field_name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_non_negative_integer(field_name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_non_negative_number(field_name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return float(value)


def _require_content_text(field_name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must contain non-whitespace text")
    return value
