"""Scripted search and retrieval capabilities for deterministic tests."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityKind,
    RetrievalMatch,
    RetrievalRequest,
    RetrievalResult,
    SearchHit,
    SearchRequest,
    SearchResult,
    SearchRetrievalMetadata,
)
from agentrig.core.context import RunContext
from agentrig.core.errors import AgentRigError, Failure
from agentrig.testing._scripted_capabilities import (
    check_constraints,
    exhaustion_failure,
    require_context,
    require_descriptor_kind,
)
from agentrig.testing._scripted_outcomes import ScriptedOutcomes

DocumentT = TypeVar("DocumentT")
ChunkT = TypeVar("ChunkT")


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedSearchScenario:
    """Provider-neutral hits and metadata for one bounded search result."""

    hits: tuple[SearchHit, ...]
    metadata: SearchRetrievalMetadata

    def __post_init__(self) -> None:
        copied_hits = tuple(self.hits)
        if any(not isinstance(hit, SearchHit) for hit in copied_hits):
            raise TypeError(
                "scripted search hits must contain SearchHit values"
            )
        source_uris = tuple(hit.source_uri for hit in copied_hits)
        if len(source_uris) != len(set(source_uris)):
            raise ValueError(
                "scripted search source URIs must not contain duplicates"
            )
        if not isinstance(self.metadata, SearchRetrievalMetadata):
            raise TypeError(
                "scripted search metadata must be SearchRetrievalMetadata"
            )
        if (
            self.metadata.total_available is not None
            and self.metadata.total_available < len(copied_hits)
        ):
            raise ValueError(
                "scripted search total_available must cover every hit"
            )
        object.__setattr__(self, "hits", copied_hits)


ScriptedSearchOutcome: TypeAlias = ScriptedSearchScenario | Failure


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedSearchProviderCall:
    """One request and context presented to a scripted search provider."""

    index: int
    request: SearchRequest
    context: RunContext


class ScriptedSearchProvider:
    """Return predefined search scenarios or normalized failures in order."""

    def __init__(
        self,
        *,
        descriptor: CapabilityDescriptor,
        outcomes: Iterable[ScriptedSearchOutcome],
        repeat_last: bool = False,
    ) -> None:
        require_descriptor_kind(
            descriptor,
            CapabilityKind.SEARCH,
            "scripted search provider",
        )
        copied_outcomes = tuple(outcomes)
        if not copied_outcomes:
            raise ValueError(
                "scripted search provider requires at least one outcome"
            )
        if any(
            not isinstance(outcome, (ScriptedSearchScenario, Failure))
            for outcome in copied_outcomes
        ):
            raise TypeError(
                "scripted search outcomes must contain "
                "ScriptedSearchScenario or Failure values"
            )

        self._descriptor = descriptor
        self._script = ScriptedOutcomes[
            ScriptedSearchOutcome,
            ScriptedSearchProviderCall,
        ](outcomes=copied_outcomes, repeat_last=repeat_last)

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    @property
    def calls(self) -> tuple[ScriptedSearchProviderCall, ...]:
        """Return a stable snapshot of recorded search calls."""
        return self._script.calls

    @property
    def is_exhausted(self) -> bool:
        """Whether another call would raise the exhaustion failure."""
        return self._script.is_exhausted

    async def search(
        self,
        request: SearchRequest,
        context: RunContext,
    ) -> SearchResult:
        """Consume one scenario after portable preflight and constraints."""
        if not isinstance(request, SearchRequest):
            raise TypeError("scripted search request must be a SearchRequest")
        require_context(context, "scripted search provider")
        request.require_supported_by(self.descriptor)
        check_constraints(context)

        outcome = self._script.record_and_take(
            lambda index: ScriptedSearchProviderCall(
                index=index,
                request=request,
                context=context,
            )
        )
        if outcome is None:
            raise AgentRigError(
                exhaustion_failure(
                    self.descriptor,
                    code="scripted_search_provider.exhausted",
                    message=(
                        "scripted search provider has no remaining outcomes"
                    ),
                )
            )
        if isinstance(outcome, Failure):
            raise AgentRigError(outcome)
        return SearchResult(
            request=request,
            hits=outcome.hits,
            metadata=outcome.metadata,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedRetrievalScenario(Generic[DocumentT, ChunkT]):
    """Provider-neutral typed matches for one bounded retrieval result."""

    matches: tuple[RetrievalMatch[DocumentT, ChunkT], ...]

    def __post_init__(self) -> None:
        copied_matches = tuple(self.matches)
        if any(
            not isinstance(match, RetrievalMatch)
            for match in copied_matches
        ):
            raise TypeError(
                "scripted retrieval matches must contain RetrievalMatch values"
            )
        object.__setattr__(self, "matches", copied_matches)


ScriptedRetrievalOutcome: TypeAlias = (
    ScriptedRetrievalScenario[DocumentT, ChunkT] | Failure
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedRetrieverCall:
    """One request and context presented to a scripted retriever."""

    index: int
    request: RetrievalRequest
    context: RunContext


class ScriptedRetriever(Generic[DocumentT, ChunkT]):
    """Return predefined typed retrieval matches in call order."""

    def __init__(
        self,
        *,
        descriptor: CapabilityDescriptor,
        outcomes: Iterable[ScriptedRetrievalOutcome[DocumentT, ChunkT]],
        repeat_last: bool = False,
    ) -> None:
        require_descriptor_kind(
            descriptor,
            CapabilityKind.RETRIEVAL,
            "scripted retriever",
        )
        copied_outcomes = tuple(outcomes)
        if not copied_outcomes:
            raise ValueError("scripted retriever requires at least one outcome")
        if any(
            not isinstance(outcome, (ScriptedRetrievalScenario, Failure))
            for outcome in copied_outcomes
        ):
            raise TypeError(
                "scripted retrieval outcomes must contain "
                "ScriptedRetrievalScenario or Failure values"
            )

        self._descriptor = descriptor
        self._script = ScriptedOutcomes[
            ScriptedRetrievalOutcome[DocumentT, ChunkT],
            ScriptedRetrieverCall,
        ](outcomes=copied_outcomes, repeat_last=repeat_last)

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    @property
    def calls(self) -> tuple[ScriptedRetrieverCall, ...]:
        """Return a stable snapshot of recorded retrieval calls."""
        return self._script.calls

    @property
    def is_exhausted(self) -> bool:
        """Whether another call would raise the exhaustion failure."""
        return self._script.is_exhausted

    async def retrieve(
        self,
        request: RetrievalRequest,
        context: RunContext,
    ) -> RetrievalResult[DocumentT, ChunkT]:
        """Consume one scenario after portable preflight and constraints."""
        if not isinstance(request, RetrievalRequest):
            raise TypeError(
                "scripted retrieval request must be a RetrievalRequest"
            )
        require_context(context, "scripted retriever")
        request.require_supported_by(self.descriptor)
        check_constraints(context)

        outcome = self._script.record_and_take(
            lambda index: ScriptedRetrieverCall(
                index=index,
                request=request,
                context=context,
            )
        )
        if outcome is None:
            raise AgentRigError(
                exhaustion_failure(
                    self.descriptor,
                    code="scripted_retriever.exhausted",
                    message="scripted retriever has no remaining outcomes",
                )
            )
        if isinstance(outcome, Failure):
            raise AgentRigError(outcome)
        return RetrievalResult(
            request=request,
            matches=outcome.matches,
        )
