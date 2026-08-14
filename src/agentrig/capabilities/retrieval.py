"""Storage-independent typed retrieval contracts."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field as dataclass_field
from typing import Generic, Protocol, TypeVar, runtime_checkable

from agentrig.capabilities.base import (
    CapabilityDescriptor,
    CapabilityKind,
    CapabilityLimit,
    CapabilityRequirements,
)
from agentrig.core._json import JsonValue, freeze_json_object, freeze_json_value
from agentrig.core._validation import require_trimmed_string
from agentrig.core.context import RunContext

DocumentT = TypeVar("DocumentT")
ChunkT = TypeVar("ChunkT")


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalFilter:
    """One storage-independent document-metadata equality constraint."""

    field: str
    value: JsonValue = dataclass_field(repr=False)

    def __post_init__(self) -> None:
        require_trimmed_string("retrieval filter field", self.field)
        object.__setattr__(
            self,
            "value",
            freeze_json_value("retrieval filter value", self.value),
        )

    def matches(self, metadata: Mapping[str, JsonValue]) -> bool:
        """Whether metadata contains the exact frozen field value."""
        if not isinstance(metadata, Mapping):
            raise TypeError("retrieval filter metadata must be a mapping")
        frozen_metadata = freeze_json_object(
            "retrieval filter metadata",
            metadata,
        )
        return (
            self.field in frozen_metadata
            and frozen_metadata[self.field] == self.value
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalRequest:
    """One private query, metadata filters, and hard result bound."""

    query: str = dataclass_field(repr=False)
    filters: tuple[RetrievalFilter, ...] = dataclass_field(
        default=(),
        repr=False,
    )
    max_results: int = 10

    def __post_init__(self) -> None:
        _require_content_text("retrieval query", self.query)
        copied_filters = tuple(self.filters)
        if any(
            not isinstance(item, RetrievalFilter) for item in copied_filters
        ):
            raise TypeError(
                "retrieval filters must contain RetrievalFilter values"
            )
        fields = tuple(item.field for item in copied_filters)
        if len(fields) != len(set(fields)):
            raise ValueError("retrieval filter fields must be unique")
        object.__setattr__(self, "filters", copied_filters)
        _require_positive_integer("retrieval max_results", self.max_results)

    @property
    def requirements(self) -> CapabilityRequirements:
        """Require retrieval identity and sufficient result capacity."""
        return CapabilityRequirements(
            kind=CapabilityKind.RETRIEVAL,
            minimum_limits={
                CapabilityLimit.MAX_RESULTS: self.max_results,
            },
        )

    def require_supported_by(self, descriptor: CapabilityDescriptor) -> None:
        """Fail before provider execution if this request is unsupported."""
        self.requirements.require(descriptor)


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievedDocument(Generic[DocumentT]):
    """One typed document with immutable portable metadata."""

    document_id: str
    value: DocumentT = dataclass_field(repr=False)
    metadata: Mapping[str, JsonValue] = dataclass_field(
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        require_trimmed_string("retrieved document ID", self.document_id)
        object.__setattr__(
            self,
            "metadata",
            freeze_json_object(
                "retrieved document metadata",
                self.metadata,
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievedChunk(Generic[ChunkT]):
    """One typed chunk linked to its containing document identity."""

    chunk_id: str
    document_id: str
    value: ChunkT = dataclass_field(repr=False)
    metadata: Mapping[str, JsonValue] = dataclass_field(
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        require_trimmed_string("retrieved chunk ID", self.chunk_id)
        require_trimmed_string(
            "retrieved chunk document ID",
            self.document_id,
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_json_object(
                "retrieved chunk metadata",
                self.metadata,
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalScore:
    """Finite provider-declared score semantics without a vector assumption."""

    value: float
    metric: str
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, (int, float))
            or not math.isfinite(self.value)
        ):
            raise ValueError("retrieval score must be a finite number")
        object.__setattr__(self, "value", float(self.value))
        require_trimmed_string("retrieval score metric", self.metric)
        if not isinstance(self.higher_is_better, bool):
            raise TypeError("retrieval score higher_is_better must be a bool")


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalMatch(Generic[DocumentT, ChunkT]):
    """One ranked typed document or document-chunk match."""

    document: RetrievedDocument[DocumentT] = dataclass_field(repr=False)
    score: RetrievalScore
    chunk: RetrievedChunk[ChunkT] | None = dataclass_field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.document, RetrievedDocument):
            raise TypeError(
                "retrieval match document must be a RetrievedDocument"
            )
        if not isinstance(self.score, RetrievalScore):
            raise TypeError("retrieval match score must be a RetrievalScore")
        if self.chunk is not None:
            if not isinstance(self.chunk, RetrievedChunk):
                raise TypeError(
                    "retrieval match chunk must be a RetrievedChunk or None"
                )
            if self.chunk.document_id != self.document.document_id:
                raise ValueError(
                    "retrieval chunk must reference the matched document"
                )


@dataclass(frozen=True, slots=True, init=False)
class RetrievalResult(Generic[DocumentT, ChunkT]):
    """A bounded, consistently scored, ordered set of typed matches."""

    matches: tuple[RetrievalMatch[DocumentT, ChunkT], ...] = dataclass_field(
        repr=False,
    )

    def __init__(
        self,
        *,
        request: RetrievalRequest,
        matches: Iterable[RetrievalMatch[DocumentT, ChunkT]],
    ) -> None:
        if not isinstance(request, RetrievalRequest):
            raise TypeError(
                "retrieval result request must be a RetrievalRequest"
            )
        copied_matches = tuple(matches)
        if any(
            not isinstance(item, RetrievalMatch) for item in copied_matches
        ):
            raise TypeError(
                "retrieval result matches must contain RetrievalMatch values"
            )
        if len(copied_matches) > request.max_results:
            raise ValueError(
                "retrieval result exceeds the request max_results"
            )
        _require_unique_match_identities(copied_matches)
        _require_consistent_documents(copied_matches)
        _require_matching_filters(request, copied_matches)
        _require_ranked_scores(copied_matches)
        object.__setattr__(self, "matches", copied_matches)

    @property
    def documents(self) -> tuple[RetrievedDocument[DocumentT], ...]:
        """Return unique documents in first-match rank order."""
        documents: list[RetrievedDocument[DocumentT]] = []
        seen_ids: set[str] = set()
        for match in self.matches:
            if match.document.document_id not in seen_ids:
                seen_ids.add(match.document.document_id)
                documents.append(match.document)
        return tuple(documents)

    @property
    def chunks(self) -> tuple[RetrievedChunk[ChunkT], ...]:
        """Return matched chunks in result rank order."""
        return tuple(
            match.chunk for match in self.matches if match.chunk is not None
        )


@runtime_checkable
class Retriever(Protocol[DocumentT, ChunkT]):
    """Retrieve typed documents or chunks without prescribing storage."""

    @property
    def descriptor(self) -> CapabilityDescriptor:
        """Return stable identity and supported portable limits."""
        ...

    async def retrieve(
        self,
        request: RetrievalRequest,
        context: RunContext,
    ) -> RetrievalResult[DocumentT, ChunkT]:
        """Return one bounded typed result or raise a normalized failure."""
        ...


def _require_unique_match_identities(
    matches: tuple[RetrievalMatch[DocumentT, ChunkT], ...],
) -> None:
    identities = tuple(
        (
            match.document.document_id,
            match.chunk.chunk_id if match.chunk is not None else None,
        )
        for match in matches
    )
    if len(identities) != len(set(identities)):
        raise ValueError("retrieval result match identities must be unique")


def _require_consistent_documents(
    matches: tuple[RetrievalMatch[DocumentT, ChunkT], ...],
) -> None:
    documents: dict[str, RetrievedDocument[DocumentT]] = {}
    for match in matches:
        document_id = match.document.document_id
        if document_id in documents and documents[document_id] != match.document:
            raise ValueError(
                "retrieval result must preserve one value per document ID"
            )
        documents[document_id] = match.document


def _require_matching_filters(
    request: RetrievalRequest,
    matches: tuple[RetrievalMatch[DocumentT, ChunkT], ...],
) -> None:
    if any(
        not item.matches(match.document.metadata)
        for match in matches
        for item in request.filters
    ):
        raise ValueError("retrieval result contains a document outside its filters")


def _require_ranked_scores(
    matches: tuple[RetrievalMatch[DocumentT, ChunkT], ...],
) -> None:
    if not matches:
        return
    first_score = matches[0].score
    if any(
        match.score.metric != first_score.metric
        or match.score.higher_is_better != first_score.higher_is_better
        for match in matches[1:]
    ):
        raise ValueError(
            "retrieval result scores must use one metric and direction"
        )
    for previous, current in zip(matches, matches[1:]):
        if first_score.higher_is_better:
            out_of_order = previous.score.value < current.score.value
        else:
            out_of_order = previous.score.value > current.score.value
        if out_of_order:
            raise ValueError("retrieval result scores must follow rank order")


def _require_positive_integer(field_name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_content_text(field_name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must contain non-whitespace text")
    return value
