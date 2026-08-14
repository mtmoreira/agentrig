"""Positive fixture for storage-independent typed retrieval."""

from dataclasses import dataclass

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityKind,
    CapabilityLimit,
    RetrievalMatch,
    RetrievalRequest,
    RetrievalResult,
    RetrievalScore,
    RetrievedChunk,
    RetrievedDocument,
    Retriever,
)
from agentrig.core import RunContext


@dataclass(frozen=True)
class StoryDocument:
    title: str


@dataclass(frozen=True)
class StoryChunk:
    text: str


@dataclass(frozen=True)
class DeterministicRetriever:
    descriptor: CapabilityDescriptor

    async def retrieve(
        self,
        request: RetrievalRequest,
        context: RunContext,
    ) -> RetrievalResult[StoryDocument, StoryChunk]:
        del context
        request.require_supported_by(self.descriptor)
        document = RetrievedDocument(
            document_id="story-1",
            value=StoryDocument(title="Arrival"),
        )
        return RetrievalResult(
            request=request,
            matches=(
                RetrievalMatch(
                    document=document,
                    chunk=RetrievedChunk(
                        chunk_id="story-1:opening",
                        document_id=document.document_id,
                        value=StoryChunk(text="The traveler arrived."),
                    ),
                    score=RetrievalScore(
                        value=1.0,
                        metric="provider_relevance",
                    ),
                ),
            ),
        )


retriever: Retriever[StoryDocument, StoryChunk] = DeterministicRetriever(
    descriptor=CapabilityDescriptor(
        capability_id="example.retrieval",
        version="1",
        kind=CapabilityKind.RETRIEVAL,
        limits={CapabilityLimit.MAX_RESULTS: 10},
    )
)


async def retrieve(
    request: RetrievalRequest,
    context: RunContext,
) -> tuple[RetrievedDocument[StoryDocument], ...]:
    result = await retriever.retrieve(request, context)
    return result.documents
