from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import unittest

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    DataRetention,
    RetrievalFilter,
    RetrievalMatch,
    RetrievalRequest,
    RetrievalScore,
    RetrievedChunk,
    RetrievedDocument,
    Retriever,
    SearchHit,
    SearchProvider,
    SearchRequest,
    SearchRetrievalMetadata,
)
from agentrig.core import (
    AgentRigError,
    CancellationSource,
    Deadline,
    DeadlineExceeded,
    Failure,
    FailureKind,
    RunCancelled,
    RunContext,
    RunId,
)
from agentrig.testing import (
    RetrieverContractSuite,
    ScriptedRetrievalScenario,
    ScriptedRetriever,
    ScriptedSearchProvider,
    ScriptedSearchScenario,
    SearchProviderContractSuite,
)


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 15, 8, 0, tzinfo=UTC)

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
    *,
    deadline: Deadline | None = None,
) -> RunContext:
    owned_source = source if source is not None else CancellationSource()
    return RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialRunIdGenerator(),
        cancellation=owned_source.token,
        deadline=deadline,
    )


def cancelled_context() -> RunContext:
    source = CancellationSource()
    source.cancel("contract cancellation")
    return create_context(source)


def search_descriptor(max_results: int = 2) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id="scripted.search",
        version="1",
        kind=CapabilityKind.SEARCH,
        features=frozenset({CapabilityFeature.CITATIONS}),
        limits={CapabilityLimit.MAX_RESULTS: max_results},
        data_retention=DataRetention.NOT_RETAINED,
    )


def search_hit(suffix: str = "1") -> SearchHit:
    return SearchHit(
        source_uri=f"https://example.com/sources/{suffix}",
        title=f"Source {suffix}",
        excerpt=f"Excerpt {suffix}",
    )


def search_scenario(
    hits: tuple[SearchHit, ...] | None = None,
) -> ScriptedSearchScenario:
    owned_hits = hits if hits is not None else (search_hit(),)
    return ScriptedSearchScenario(
        hits=owned_hits,
        metadata=SearchRetrievalMetadata(
            retrieved_at=datetime(2026, 8, 15, 8, 0, tzinfo=UTC),
            duration_seconds=0.25,
            total_available=len(owned_hits),
        ),
    )


@dataclass(frozen=True, slots=True)
class StoryDocument:
    title: str


@dataclass(frozen=True, slots=True)
class StoryChunk:
    text: str


def retrieval_descriptor(max_results: int = 2) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id="scripted.retrieval",
        version="1",
        kind=CapabilityKind.RETRIEVAL,
        limits={CapabilityLimit.MAX_RESULTS: max_results},
        data_retention=DataRetention.NOT_RETAINED,
    )


def retrieval_match(
    suffix: str = "1",
    *,
    score: float = 1.0,
) -> RetrievalMatch[StoryDocument, StoryChunk]:
    document = RetrievedDocument(
        document_id=f"story-{suffix}",
        value=StoryDocument(title=f"Story {suffix}"),
        metadata={"world": "harbor"},
    )
    return RetrievalMatch(
        document=document,
        chunk=RetrievedChunk(
            chunk_id=f"story-{suffix}:opening",
            document_id=document.document_id,
            value=StoryChunk(text=f"Passage {suffix}"),
        ),
        score=RetrievalScore(
            value=score,
            metric="provider_relevance",
        ),
    )


def retrieval_scenario(
    matches: tuple[RetrievalMatch[StoryDocument, StoryChunk], ...] | None = None,
) -> ScriptedRetrievalScenario[StoryDocument, StoryChunk]:
    return ScriptedRetrievalScenario(
        matches=(retrieval_match(),) if matches is None else matches
    )


def retrieval_request(max_results: int = 1) -> RetrievalRequest:
    return RetrievalRequest(
        query="Find harbor stories.",
        filters=(RetrievalFilter(field="world", value="harbor"),),
        max_results=max_results,
    )


def provider_failure() -> Failure:
    return Failure(
        kind=FailureKind.TRANSIENT_PROVIDER,
        message="scripted provider is temporarily unavailable",
        code="provider.busy",
    )


class ScriptedSearchProviderTest(unittest.TestCase):
    def test_returns_scenarios_and_failures_in_order_with_stable_calls(
        self,
    ) -> None:
        scenario = search_scenario()
        failure = provider_failure()
        provider = ScriptedSearchProvider(
            descriptor=search_descriptor(),
            outcomes=(scenario, failure),
        )
        request = SearchRequest(query="sources", max_results=1)
        context = create_context()

        first = asyncio.run(provider.search(request, context))
        snapshot = provider.calls
        with self.assertRaises(AgentRigError) as raised:
            asyncio.run(provider.search(request, context))

        self.assertIsInstance(provider, SearchProvider)
        self.assertEqual(first.hits, scenario.hits)
        self.assertEqual(first.metadata, scenario.metadata)
        self.assertIs(raised.exception.failure, failure)
        self.assertEqual(tuple(call.index for call in snapshot), (0,))
        self.assertEqual(tuple(call.index for call in provider.calls), (0, 1))
        self.assertTrue(provider.is_exhausted)

    def test_request_bound_preflight_constraints_and_result_limit(self) -> None:
        provider = ScriptedSearchProvider(
            descriptor=search_descriptor(),
            outcomes=(
                search_scenario((search_hit("1"), search_hit("2"))),
            ),
        )
        with self.assertRaises(ValueError):
            asyncio.run(
                provider.search(
                    SearchRequest(query="sources", max_results=3),
                    create_context(),
                )
            )
        with self.assertRaises(RunCancelled):
            asyncio.run(
                provider.search(
                    SearchRequest(query="sources", max_results=1),
                    cancelled_context(),
                )
            )
        expired = Deadline(
            expires_at=datetime(2026, 8, 15, 8, 0, tzinfo=UTC),
            monotonic_deadline=100.0,
        )
        with self.assertRaises(DeadlineExceeded):
            asyncio.run(
                provider.search(
                    SearchRequest(query="sources", max_results=1),
                    create_context(deadline=expired),
                )
            )
        self.assertEqual(provider.calls, ())
        self.assertFalse(provider.is_exhausted)

        with self.assertRaisesRegex(ValueError, "request max_results"):
            asyncio.run(
                provider.search(
                    SearchRequest(query="sources", max_results=1),
                    create_context(),
                )
            )

    def test_exhaustion_is_sanitized_and_repeat_last_is_unbounded(self) -> None:
        exhausted = ScriptedSearchProvider(
            descriptor=search_descriptor(),
            outcomes=(search_scenario(),),
        )
        request = SearchRequest(query="sources", max_results=1)
        asyncio.run(exhausted.search(request, create_context()))

        with self.assertRaises(AgentRigError) as raised:
            asyncio.run(exhausted.search(request, create_context()))
        self.assertEqual(
            raised.exception.failure.code,
            "scripted_search_provider.exhausted",
        )

        repeating = ScriptedSearchProvider(
            descriptor=search_descriptor(),
            outcomes=(search_scenario(),),
            repeat_last=True,
        )
        for _ in range(3):
            self.assertEqual(
                len(asyncio.run(repeating.search(request, create_context())).hits),
                1,
            )
        self.assertFalse(repeating.is_exhausted)


class ScriptedRetrieverTest(unittest.TestCase):
    def test_returns_typed_scenarios_and_failures_in_order(self) -> None:
        scenario = retrieval_scenario()
        failure = provider_failure()
        retriever = ScriptedRetriever[StoryDocument, StoryChunk](
            descriptor=retrieval_descriptor(),
            outcomes=(scenario, failure),
        )
        request = retrieval_request()
        context = create_context()

        first = asyncio.run(retriever.retrieve(request, context))
        with self.assertRaises(AgentRigError) as raised:
            asyncio.run(retriever.retrieve(request, context))

        self.assertIsInstance(retriever, Retriever)
        self.assertEqual(first.matches, scenario.matches)
        self.assertIs(raised.exception.failure, failure)
        self.assertEqual(tuple(call.index for call in retriever.calls), (0, 1))
        self.assertTrue(retriever.is_exhausted)

    def test_preflight_constraints_filters_and_bounds_are_request_relative(
        self,
    ) -> None:
        scenario = retrieval_scenario(
            (retrieval_match("1", score=1), retrieval_match("2", score=0.5))
        )
        retriever = ScriptedRetriever[StoryDocument, StoryChunk](
            descriptor=retrieval_descriptor(),
            outcomes=(scenario,),
        )

        with self.assertRaises(ValueError):
            asyncio.run(retriever.retrieve(retrieval_request(3), create_context()))
        with self.assertRaises(RunCancelled):
            asyncio.run(
                retriever.retrieve(retrieval_request(), cancelled_context())
            )
        self.assertEqual(retriever.calls, ())

        with self.assertRaisesRegex(ValueError, "request max_results"):
            asyncio.run(
                retriever.retrieve(retrieval_request(1), create_context())
            )

        filtered = ScriptedRetriever[StoryDocument, StoryChunk](
            descriptor=retrieval_descriptor(),
            outcomes=(retrieval_scenario(),),
        )
        outside_filter = RetrievalRequest(
            query="Find mountain stories.",
            filters=(RetrievalFilter(field="world", value="mountain"),),
            max_results=1,
        )
        with self.assertRaisesRegex(ValueError, "outside its filters"):
            asyncio.run(filtered.retrieve(outside_filter, create_context()))

    def test_exhaustion_is_sanitized_and_repeat_last_is_unbounded(self) -> None:
        exhausted = ScriptedRetriever[StoryDocument, StoryChunk](
            descriptor=retrieval_descriptor(),
            outcomes=(retrieval_scenario(),),
        )
        asyncio.run(exhausted.retrieve(retrieval_request(), create_context()))
        with self.assertRaises(AgentRigError) as raised:
            asyncio.run(
                exhausted.retrieve(retrieval_request(), create_context())
            )
        self.assertEqual(
            raised.exception.failure.code,
            "scripted_retriever.exhausted",
        )

        repeating = ScriptedRetriever[StoryDocument, StoryChunk](
            descriptor=retrieval_descriptor(),
            outcomes=(retrieval_scenario(),),
            repeat_last=True,
        )
        for _ in range(3):
            result = asyncio.run(
                repeating.retrieve(retrieval_request(), create_context())
            )
            self.assertEqual(len(result.matches), 1)
        self.assertFalse(repeating.is_exhausted)


class DataContractSuiteTest(unittest.TestCase):
    def test_search_suite_verifies_shared_portable_semantics(self) -> None:
        provider = ScriptedSearchProvider(
            descriptor=search_descriptor(),
            outcomes=(search_scenario(),),
        )
        suite = SearchProviderContractSuite(
            provider=provider,
            supported_request=SearchRequest(query="sources", max_results=1),
            unsupported_request=SearchRequest(query="sources", max_results=3),
            context=create_context(),
            cancelled_context=cancelled_context(),
            invocation_count=lambda: len(provider.calls),
        )

        result = asyncio.run(suite.verify())

        self.assertEqual(len(result.hits), 1)
        self.assertEqual(len(provider.calls), 1)

    def test_retrieval_suite_verifies_shared_portable_semantics(self) -> None:
        retriever = ScriptedRetriever[StoryDocument, StoryChunk](
            descriptor=retrieval_descriptor(),
            outcomes=(retrieval_scenario(),),
        )
        suite = RetrieverContractSuite[StoryDocument, StoryChunk](
            retriever=retriever,
            supported_request=retrieval_request(),
            unsupported_request=retrieval_request(3),
            context=create_context(),
            cancelled_context=cancelled_context(),
            invocation_count=lambda: len(retriever.calls),
        )

        result = asyncio.run(suite.verify())

        self.assertEqual(len(result.matches), 1)
        self.assertEqual(len(retriever.calls), 1)


class ScriptedDataValidationTest(unittest.TestCase):
    def test_rejects_invalid_descriptors_outcomes_and_scenarios(self) -> None:
        with self.assertRaises(ValueError):
            ScriptedSearchProvider(
                descriptor=retrieval_descriptor(),
                outcomes=(search_scenario(),),
            )
        with self.assertRaises(ValueError):
            ScriptedSearchProvider(descriptor=search_descriptor(), outcomes=())
        with self.assertRaises(TypeError):
            ScriptedSearchProvider(
                descriptor=search_descriptor(),
                outcomes=("invalid",),  # type: ignore[arg-type]
            )
        duplicate = search_hit()
        with self.assertRaises(ValueError):
            search_scenario((duplicate, duplicate))
        with self.assertRaises(TypeError):
            ScriptedSearchScenario(
                hits=("invalid",),  # type: ignore[arg-type]
                metadata=SearchRetrievalMetadata(
                    retrieved_at=datetime(2026, 8, 15, tzinfo=UTC)
                ),
            )
        with self.assertRaises(ValueError):
            ScriptedRetriever[StoryDocument, StoryChunk](
                descriptor=search_descriptor(),
                outcomes=(retrieval_scenario(),),
            )
        with self.assertRaises(TypeError):
            ScriptedRetriever[StoryDocument, StoryChunk](
                descriptor=retrieval_descriptor(),
                outcomes=("invalid",),  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            ScriptedRetrievalScenario[StoryDocument, StoryChunk](
                matches=("invalid",),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
