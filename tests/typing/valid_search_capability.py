"""Positive fixture for typed bounded search execution."""

from dataclasses import dataclass

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    SearchHit,
    SearchProvider,
    SearchRequest,
    SearchResult,
    SearchRetrievalMetadata,
)
from agentrig.core import RunContext


@dataclass(frozen=True)
class DeterministicSearchProvider:
    descriptor: CapabilityDescriptor

    async def search(
        self,
        request: SearchRequest,
        context: RunContext,
    ) -> SearchResult:
        request.require_supported_by(self.descriptor)
        return SearchResult(
            request=request,
            hits=(
                SearchHit(
                    source_uri="https://example.com/source",
                    title="Example source",
                    excerpt="A bounded search excerpt.",
                ),
            ),
            metadata=SearchRetrievalMetadata(
                retrieved_at=context.clock.now(),
                total_available=1,
            ),
        )


provider: SearchProvider = DeterministicSearchProvider(
    descriptor=CapabilityDescriptor(
        capability_id="example.search",
        version="1",
        kind=CapabilityKind.SEARCH,
        features=frozenset({CapabilityFeature.CITATIONS}),
        limits={CapabilityLimit.MAX_RESULTS: 10},
    )
)


async def search(
    request: SearchRequest,
    context: RunContext,
) -> tuple[SearchHit, ...]:
    result = await provider.search(request, context)
    return result.hits
