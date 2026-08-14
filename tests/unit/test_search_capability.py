from __future__ import annotations

import asyncio
import math
import unittest
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    DataRetention,
    SearchCitation,
    SearchHit,
    SearchProvider,
    SearchRequest,
    SearchResult,
    SearchRetrievalMetadata,
)
from agentrig.core import (
    CancellationSource,
    RunCancelled,
    RunContext,
    RunId,
)


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 14, 6, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


@dataclass
class SequentialRunIdGenerator:
    next_value: int = 1

    def generate(self) -> RunId:
        run_id = RunId(f"run-{self.next_value}")
        self.next_value += 1
        return run_id


def create_context(
    source: CancellationSource | None = None,
) -> RunContext:
    owned_source = source if source is not None else CancellationSource()
    return RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialRunIdGenerator(),
        cancellation=owned_source.token,
    )


def create_descriptor(
    *,
    features: frozenset[CapabilityFeature] | None = None,
    max_results: int | None = 5,
) -> CapabilityDescriptor:
    limits = (
        {CapabilityLimit.MAX_RESULTS: max_results}
        if max_results is not None
        else {}
    )
    return CapabilityDescriptor(
        capability_id="example.search",
        version="1",
        kind=CapabilityKind.SEARCH,
        features=(
            features
            if features is not None
            else frozenset({CapabilityFeature.CITATIONS})
        ),
        limits=limits,
        data_retention=DataRetention.NOT_RETAINED,
    )


def create_hit(
    suffix: str = "1",
    *,
    source_uri: str | None = None,
) -> SearchHit:
    return SearchHit(
        source_uri=(
            source_uri
            if source_uri is not None
            else f"https://example.com/sources/{suffix}"
        ),
        title=f"  Source {suffix}\n",
        excerpt=f"  Excerpt {suffix}\n",
    )


def create_metadata(
    *,
    total_available: int | None = 2,
) -> SearchRetrievalMetadata:
    return SearchRetrievalMetadata(
        retrieved_at=datetime(2026, 8, 14, 6, 0, tzinfo=UTC),
        duration_seconds=0.25,
        total_available=total_available,
    )


@dataclass
class ScriptedSearchProvider:
    descriptor: CapabilityDescriptor
    batches: tuple[tuple[SearchHit, ...], ...]
    calls: list[tuple[SearchRequest, RunContext]] = field(default_factory=list)

    async def search(
        self,
        request: SearchRequest,
        context: RunContext,
    ) -> SearchResult:
        request.require_supported_by(self.descriptor)
        context.cancellation.raise_if_cancelled()
        if context.deadline is not None:
            context.deadline.raise_if_expired(context.clock)
        index = len(self.calls)
        self.calls.append((request, context))
        hits = self.batches[min(index, len(self.batches) - 1)]
        return SearchResult(
            request=request,
            hits=hits,
            metadata=SearchRetrievalMetadata(
                retrieved_at=context.clock.now(),
                duration_seconds=0,
                total_available=len(hits),
            ),
        )


async def search_typed(
    provider: SearchProvider,
    request: SearchRequest,
    context: RunContext,
) -> SearchResult:
    return await provider.search(request, context)


class SearchRequestTest(unittest.TestCase):
    def test_preserves_private_query_and_derives_bounded_requirements(self) -> None:
        request = SearchRequest(
            query="  reliable lunar tide sources\n",
            max_results=3,
        )

        self.assertEqual(request.query, "  reliable lunar tide sources\n")
        self.assertNotIn("lunar tide", repr(request))
        self.assertEqual(
            request.requirements.features,
            frozenset({CapabilityFeature.CITATIONS}),
        )
        self.assertEqual(
            request.requirements.minimum_limits,
            {CapabilityLimit.MAX_RESULTS: 3},
        )
        request.require_supported_by(create_descriptor(max_results=3))

    def test_rejects_invalid_request_and_unsupported_capacity(self) -> None:
        for query in ("", " \n", object()):
            with self.subTest(query=query):
                with self.assertRaises(ValueError):
                    SearchRequest(query=query)  # type: ignore[arg-type]
        for max_results in (0, -1, True):
            with self.subTest(max_results=max_results):
                with self.assertRaises(ValueError):
                    SearchRequest(
                        query="query",
                        max_results=max_results,  # type: ignore[arg-type]
                    )
        request = SearchRequest(query="query", max_results=3)
        with self.assertRaisesRegex(ValueError, "feature:citations"):
            request.require_supported_by(
                create_descriptor(features=frozenset(), max_results=3)
            )
        with self.assertRaisesRegex(ValueError, "max_results>=3"):
            request.require_supported_by(create_descriptor(max_results=2))


class SearchHitTest(unittest.TestCase):
    def test_preserves_citation_content_and_normalizes_publication_time(self) -> None:
        eastern = timezone(timedelta(hours=-4))
        hit = SearchHit(
            source_uri="https://example.com/tides",
            title="  Lunar tides\n",
            summary="  A sourced overview.\n",
            published_at=datetime(2026, 8, 14, 1, 0, tzinfo=eastern),
        )

        self.assertEqual(hit.title, "  Lunar tides\n")
        self.assertEqual(hit.summary, "  A sourced overview.\n")
        self.assertEqual(
            hit.published_at,
            datetime(2026, 8, 14, 5, 0, tzinfo=UTC),
        )
        self.assertEqual(
            hit.citation,
            SearchCitation(
                source_uri="https://example.com/tides",
                title="  Lunar tides\n",
            ),
        )
        self.assertNotIn("Lunar tides", repr(hit))
        self.assertNotIn("example.com", repr(hit.citation))

    def test_rejects_invalid_sources_content_or_publication_time(self) -> None:
        for source_uri in (
            "example.com/source",
            " https://example.com/source",
            "https://example.com/private path",
        ):
            with self.subTest(source_uri=source_uri):
                with self.assertRaises(ValueError):
                    create_hit(source_uri=source_uri)

        with self.assertRaises(ValueError):
            SearchHit(
                source_uri="https://example.com",
                title=" \n",
                excerpt="content",
            )
        with self.assertRaises(ValueError):
            SearchHit(
                source_uri="https://example.com",
                title="Source",
            )
        with self.assertRaises(ValueError):
            SearchHit(
                source_uri="https://example.com",
                title="Source",
                excerpt="content",
                published_at=datetime(2026, 8, 14),
            )
        with self.assertRaises(TypeError):
            SearchHit(
                source_uri="https://example.com",
                title="Source",
                excerpt="content",
                published_at="2026-08-14",  # type: ignore[arg-type]
            )


class SearchRetrievalMetadataTest(unittest.TestCase):
    def test_normalizes_timestamp_duration_and_availability(self) -> None:
        eastern = timezone(timedelta(hours=-4))
        metadata = SearchRetrievalMetadata(
            retrieved_at=datetime(2026, 8, 14, 2, 0, tzinfo=eastern),
            duration_seconds=2,
            total_available=8,
        )

        self.assertEqual(
            metadata.retrieved_at,
            datetime(2026, 8, 14, 6, 0, tzinfo=UTC),
        )
        self.assertEqual(metadata.duration_seconds, 2.0)
        self.assertEqual(metadata.total_available, 8)

    def test_rejects_invalid_retrieval_metadata(self) -> None:
        invalid_timestamps = ("2026-08-14", datetime(2026, 8, 14))
        for retrieved_at in invalid_timestamps:
            with self.subTest(retrieved_at=retrieved_at):
                with self.assertRaises((TypeError, ValueError)):
                    SearchRetrievalMetadata(
                        retrieved_at=retrieved_at,  # type: ignore[arg-type]
                    )
        for duration in (-1, math.inf, math.nan, True):
            with self.subTest(duration=duration):
                with self.assertRaises(ValueError):
                    SearchRetrievalMetadata(
                        retrieved_at=datetime(2026, 8, 14, tzinfo=UTC),
                        duration_seconds=duration,  # type: ignore[arg-type]
                    )
        for total_available in (-1, True):
            with self.subTest(total_available=total_available):
                with self.assertRaises(ValueError):
                    SearchRetrievalMetadata(
                        retrieved_at=datetime(2026, 8, 14, tzinfo=UTC),
                        total_available=total_available,  # type: ignore[arg-type]
                    )


class SearchResultTest(unittest.TestCase):
    def test_copies_ranked_hits_and_derives_citations(self) -> None:
        hits = [create_hit("1"), create_hit("2")]
        result = SearchResult(
            request=SearchRequest(query="query", max_results=2),
            hits=hits,
            metadata=create_metadata(),
        )
        hits.clear()

        self.assertEqual(
            tuple(citation.source_uri for citation in result.citations),
            (
                "https://example.com/sources/1",
                "https://example.com/sources/2",
            ),
        )
        self.assertEqual(result.metadata.total_available, 2)
        self.assertNotIn("Excerpt", repr(result))

    def test_rejects_invalid_unbounded_or_inconsistent_results(self) -> None:
        request = SearchRequest(query="query", max_results=1)
        with self.assertRaises(TypeError):
            SearchResult(
                request="invalid",  # type: ignore[arg-type]
                hits=(),
                metadata=create_metadata(total_available=0),
            )
        with self.assertRaises(TypeError):
            SearchResult(
                request=request,
                hits=("invalid",),  # type: ignore[arg-type]
                metadata=create_metadata(),
            )
        with self.assertRaises(ValueError):
            SearchResult(
                request=request,
                hits=(create_hit("1"), create_hit("2")),
                metadata=create_metadata(),
            )
        duplicate = create_hit("1")
        with self.assertRaises(ValueError):
            SearchResult(
                request=SearchRequest(query="query", max_results=2),
                hits=(duplicate, duplicate),
                metadata=create_metadata(),
            )
        with self.assertRaises(TypeError):
            SearchResult(
                request=request,
                hits=(),
                metadata="invalid",  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            SearchResult(
                request=SearchRequest(query="query", max_results=2),
                hits=(create_hit("1"), create_hit("2")),
                metadata=create_metadata(total_available=1),
            )


class SearchProviderContractTest(unittest.TestCase):
    def test_protocol_supports_a_provider_neutral_search(self) -> None:
        provider = ScriptedSearchProvider(
            descriptor=create_descriptor(),
            batches=((create_hit("1"), create_hit("2")),),
        )
        request = SearchRequest(query="query", max_results=2)
        context = create_context()

        result = asyncio.run(search_typed(provider, request, context))

        self.assertIsInstance(provider, SearchProvider)
        self.assertEqual(len(result.hits), 2)
        self.assertEqual(result.metadata.retrieved_at, context.clock.now())
        self.assertEqual(provider.calls, [(request, context)])

    def test_preflight_and_constraints_run_before_consuming_a_batch(self) -> None:
        unsupported = ScriptedSearchProvider(
            descriptor=create_descriptor(features=frozenset(), max_results=1),
            batches=((create_hit(),),),
        )
        request = SearchRequest(query="query", max_results=1)

        with self.assertRaises(ValueError):
            asyncio.run(unsupported.search(request, create_context()))
        self.assertEqual(unsupported.calls, [])

        source = CancellationSource()
        source.cancel("caller stopped")
        cancelled = ScriptedSearchProvider(
            descriptor=create_descriptor(max_results=1),
            batches=((create_hit(),),),
        )
        with self.assertRaises(RunCancelled):
            asyncio.run(cancelled.search(request, create_context(source)))
        self.assertEqual(cancelled.calls, [])


if __name__ == "__main__":
    unittest.main()
