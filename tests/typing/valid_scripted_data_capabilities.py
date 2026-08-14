"""Positive fixture for scripted search and retrieval contract suites."""

from dataclasses import dataclass
from datetime import UTC, datetime

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    RetrievalMatch,
    RetrievalRequest,
    RetrievalScore,
    RetrievedDocument,
    Retriever,
    SearchHit,
    SearchProvider,
    SearchRequest,
    SearchRetrievalMetadata,
)
from agentrig.core import RunContext
from agentrig.testing import (
    RetrieverContractSuite,
    ScriptedRetrievalScenario,
    ScriptedRetriever,
    ScriptedSearchProvider,
    ScriptedSearchScenario,
    SearchProviderContractSuite,
)


@dataclass(frozen=True, slots=True)
class Document:
    title: str


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str


search_fake = ScriptedSearchProvider(
    descriptor=CapabilityDescriptor(
        capability_id="scripted.search",
        version="1",
        kind=CapabilityKind.SEARCH,
        features=frozenset({CapabilityFeature.CITATIONS}),
        limits={CapabilityLimit.MAX_RESULTS: 2},
    ),
    outcomes=(
        ScriptedSearchScenario(
            hits=(
                SearchHit(
                    source_uri="https://example.com/source",
                    title="Source",
                    excerpt="Excerpt",
                ),
            ),
            metadata=SearchRetrievalMetadata(
                retrieved_at=datetime(2026, 8, 15, tzinfo=UTC),
            ),
        ),
    ),
)
search: SearchProvider = search_fake

document = RetrievedDocument(
    document_id="document-1",
    value=Document(title="Example"),
)
retrieval_fake = ScriptedRetriever[Document, Chunk](
    descriptor=CapabilityDescriptor(
        capability_id="scripted.retrieval",
        version="1",
        kind=CapabilityKind.RETRIEVAL,
        limits={CapabilityLimit.MAX_RESULTS: 2},
    ),
    outcomes=(
        ScriptedRetrievalScenario(
            matches=(
                RetrievalMatch(
                    document=document,
                    score=RetrievalScore(
                        value=1,
                        metric="provider_relevance",
                    ),
                ),
            ),
        ),
    ),
)
retriever: Retriever[Document, Chunk] = retrieval_fake


def search_suite(
    supported_request: SearchRequest,
    unsupported_request: SearchRequest,
    context: RunContext,
    cancelled_context: RunContext,
) -> SearchProviderContractSuite:
    return SearchProviderContractSuite(
        provider=search,
        supported_request=supported_request,
        unsupported_request=unsupported_request,
        context=context,
        cancelled_context=cancelled_context,
        invocation_count=lambda: len(search_fake.calls),
    )


def retrieval_suite(
    supported_request: RetrievalRequest,
    unsupported_request: RetrievalRequest,
    context: RunContext,
    cancelled_context: RunContext,
) -> RetrieverContractSuite[Document, Chunk]:
    return RetrieverContractSuite(
        retriever=retriever,
        supported_request=supported_request,
        unsupported_request=unsupported_request,
        context=context,
        cancelled_context=cancelled_context,
        invocation_count=lambda: len(retrieval_fake.calls),
    )
