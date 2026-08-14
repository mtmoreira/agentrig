"""Reusable contract probes for search and retrieval implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from agentrig.capabilities import (
    CapabilityKind,
    RetrievalRequest,
    RetrievalResult,
    Retriever,
    SearchProvider,
    SearchRequest,
    SearchResult,
)
from agentrig.core.context import RunContext
from agentrig.testing._capability_contracts import (
    InvocationCount,
    validate_contract_suite,
    verify_cancellation_does_not_invoke,
    verify_preflight_does_not_invoke,
)

DocumentT = TypeVar("DocumentT")
ChunkT = TypeVar("ChunkT")


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchProviderContractSuite:
    """Portable checks shared by search fakes and provider integrations."""

    provider: SearchProvider = field(repr=False, compare=False)
    supported_request: SearchRequest = field(repr=False)
    unsupported_request: SearchRequest = field(repr=False)
    context: RunContext = field(repr=False)
    cancelled_context: RunContext = field(repr=False)
    invocation_count: InvocationCount = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.provider, SearchProvider):
            raise TypeError(
                "search contract provider must satisfy SearchProvider"
            )
        if not isinstance(self.supported_request, SearchRequest):
            raise TypeError(
                "search contract supported_request must be a SearchRequest"
            )
        if not isinstance(self.unsupported_request, SearchRequest):
            raise TypeError(
                "search contract unsupported_request must be a SearchRequest"
            )
        validate_contract_suite(
            label="search",
            descriptor=self.provider.descriptor,
            expected_kind=CapabilityKind.SEARCH,
            supported_requirements=self.supported_request.requirements,
            unsupported_requirements=self.unsupported_request.requirements,
            context=self.context,
            cancelled_context=self.cancelled_context,
            invocation_count=self.invocation_count,
        )

    async def verify(self) -> SearchResult:
        """Run shared result, citation, preflight, and cancellation checks."""
        result = await self.provider.search(
            self.supported_request,
            self.context,
        )
        if not isinstance(result, SearchResult):
            raise AssertionError(
                "search provider returned a non-SearchResult value"
            )
        if len(result.hits) > self.supported_request.max_results:
            raise AssertionError(
                "search result exceeds the requested result bound"
            )
        if tuple(hit.citation for hit in result.hits) != result.citations:
            raise AssertionError(
                "search result citations do not preserve hit rank order"
            )

        await verify_preflight_does_not_invoke(
            label="search",
            operation=lambda: self.provider.search(
                self.unsupported_request,
                self.context,
            ),
            invocation_count=self.invocation_count,
        )
        await verify_cancellation_does_not_invoke(
            label="search",
            operation=lambda: self.provider.search(
                self.supported_request,
                self.cancelled_context,
            ),
            invocation_count=self.invocation_count,
        )
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrieverContractSuite(Generic[DocumentT, ChunkT]):
    """Portable checks shared by typed retrieval implementations."""

    retriever: Retriever[DocumentT, ChunkT] = field(
        repr=False,
        compare=False,
    )
    supported_request: RetrievalRequest = field(repr=False)
    unsupported_request: RetrievalRequest = field(repr=False)
    context: RunContext = field(repr=False)
    cancelled_context: RunContext = field(repr=False)
    invocation_count: InvocationCount = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.retriever, Retriever):
            raise TypeError(
                "retrieval contract retriever must satisfy Retriever"
            )
        if not isinstance(self.supported_request, RetrievalRequest):
            raise TypeError(
                "retrieval contract supported_request must be a "
                "RetrievalRequest"
            )
        if not isinstance(self.unsupported_request, RetrievalRequest):
            raise TypeError(
                "retrieval contract unsupported_request must be a "
                "RetrievalRequest"
            )
        validate_contract_suite(
            label="retrieval",
            descriptor=self.retriever.descriptor,
            expected_kind=CapabilityKind.RETRIEVAL,
            supported_requirements=self.supported_request.requirements,
            unsupported_requirements=self.unsupported_request.requirements,
            context=self.context,
            cancelled_context=self.cancelled_context,
            invocation_count=self.invocation_count,
        )

    async def verify(self) -> RetrievalResult[DocumentT, ChunkT]:
        """Run shared result, bound, preflight, and cancellation checks."""
        result = await self.retriever.retrieve(
            self.supported_request,
            self.context,
        )
        if not isinstance(result, RetrievalResult):
            raise AssertionError(
                "retriever returned a non-RetrievalResult value"
            )
        if len(result.matches) > self.supported_request.max_results:
            raise AssertionError(
                "retrieval result exceeds the requested result bound"
            )

        await verify_preflight_does_not_invoke(
            label="retrieval",
            operation=lambda: self.retriever.retrieve(
                self.unsupported_request,
                self.context,
            ),
            invocation_count=self.invocation_count,
        )
        await verify_cancellation_does_not_invoke(
            label="retrieval",
            operation=lambda: self.retriever.retrieve(
                self.supported_request,
                self.cancelled_context,
            ),
            invocation_count=self.invocation_count,
        )
        return result
