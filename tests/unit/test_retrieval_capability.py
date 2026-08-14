from __future__ import annotations

import asyncio
import math
import unittest
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityKind,
    CapabilityLimit,
    DataRetention,
    RetrievalFilter,
    RetrievalMatch,
    RetrievalRequest,
    RetrievalResult,
    RetrievalScore,
    RetrievedChunk,
    RetrievedDocument,
    Retriever,
)
from agentrig.core import (
    CancellationSource,
    JsonValue,
    RunCancelled,
    RunContext,
    RunId,
)


@dataclass(frozen=True)
class StoryDocument:
    title: str


@dataclass(frozen=True)
class StoryChunk:
    text: str


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 14, 7, 0, tzinfo=UTC)

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


def create_descriptor(*, max_results: int | None = 5) -> CapabilityDescriptor:
    limits = (
        {CapabilityLimit.MAX_RESULTS: max_results}
        if max_results is not None
        else {}
    )
    return CapabilityDescriptor(
        capability_id="example.retrieval",
        version="1",
        kind=CapabilityKind.RETRIEVAL,
        limits=limits,
        data_retention=DataRetention.NOT_RETAINED,
    )


def create_document(
    suffix: str = "1",
    *,
    title: str | None = None,
    world: str = "harbor",
) -> RetrievedDocument[StoryDocument]:
    return RetrievedDocument(
        document_id=f"story-{suffix}",
        value=StoryDocument(
            title=title if title is not None else f"Story {suffix}"
        ),
        metadata={"world": world, "tags": ["night", "arrival"]},
    )


def create_match(
    suffix: str = "1",
    *,
    score: float = 1.0,
    metric: str = "provider_relevance",
    higher_is_better: bool = True,
    document: RetrievedDocument[StoryDocument] | None = None,
    include_chunk: bool = True,
) -> RetrievalMatch[StoryDocument, StoryChunk]:
    owned_document = (
        document if document is not None else create_document(suffix)
    )
    chunk = (
        RetrievedChunk(
            chunk_id=f"chunk-{suffix}",
            document_id=owned_document.document_id,
            value=StoryChunk(text=f"Passage {suffix}"),
            metadata={"section": suffix},
        )
        if include_chunk
        else None
    )
    return RetrievalMatch(
        document=owned_document,
        chunk=chunk,
        score=RetrievalScore(
            value=score,
            metric=metric,
            higher_is_better=higher_is_better,
        ),
    )


@dataclass
class ScriptedRetriever:
    descriptor: CapabilityDescriptor
    batches: tuple[
        tuple[RetrievalMatch[StoryDocument, StoryChunk], ...], ...
    ]
    calls: list[tuple[RetrievalRequest, RunContext]] = field(
        default_factory=list
    )

    async def retrieve(
        self,
        request: RetrievalRequest,
        context: RunContext,
    ) -> RetrievalResult[StoryDocument, StoryChunk]:
        request.require_supported_by(self.descriptor)
        context.cancellation.raise_if_cancelled()
        if context.deadline is not None:
            context.deadline.raise_if_expired(context.clock)
        index = len(self.calls)
        self.calls.append((request, context))
        return RetrievalResult(
            request=request,
            matches=self.batches[min(index, len(self.batches) - 1)],
        )


async def retrieve_typed(
    retriever: Retriever[StoryDocument, StoryChunk],
    request: RetrievalRequest,
    context: RunContext,
) -> RetrievalResult[StoryDocument, StoryChunk]:
    return await retriever.retrieve(request, context)


class RetrievalFilterTest(unittest.TestCase):
    def test_freezes_exact_metadata_value_and_matches_documents(self) -> None:
        value: dict[str, JsonValue] = {
            "name": "harbor",
            "tags": ["night", "arrival"],
        }
        item = RetrievalFilter(field="world", value=value)
        value["name"] = "changed"

        self.assertTrue(
            item.matches(
                {
                    "world": {
                        "name": "harbor",
                        "tags": ["night", "arrival"],
                    }
                }
            )
        )
        self.assertFalse(item.matches({"world": {"name": "other"}}))
        self.assertNotIn("harbor", repr(item))
        with self.assertRaises(TypeError):
            item.matches("invalid")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            item.value["name"] = "other"  # type: ignore[index]

    def test_rejects_invalid_filter_identity_or_json(self) -> None:
        with self.assertRaises(ValueError):
            RetrievalFilter(field=" padded ", value="value")
        with self.assertRaises(ValueError):
            RetrievalFilter(
                field="world",
                value=object(),  # type: ignore[arg-type]
            )


class RetrievalRequestTest(unittest.TestCase):
    def test_copies_filters_preserves_query_and_derives_capacity(self) -> None:
        filters = [RetrievalFilter(field="world", value="harbor")]
        request = RetrievalRequest(
            query="  opening scene near water\n",
            filters=filters,  # type: ignore[arg-type]
            max_results=3,
        )
        filters.clear()

        self.assertEqual(request.query, "  opening scene near water\n")
        self.assertEqual(len(request.filters), 1)
        self.assertNotIn("opening scene", repr(request))
        self.assertNotIn("harbor", repr(request))
        self.assertEqual(
            request.requirements.minimum_limits,
            {CapabilityLimit.MAX_RESULTS: 3},
        )
        request.require_supported_by(create_descriptor(max_results=3))

    def test_rejects_invalid_request_and_unsupported_capacity(self) -> None:
        for query in ("", " \n", object()):
            with self.subTest(query=query):
                with self.assertRaises(ValueError):
                    RetrievalRequest(query=query)  # type: ignore[arg-type]
        for max_results in (0, -1, True):
            with self.subTest(max_results=max_results):
                with self.assertRaises(ValueError):
                    RetrievalRequest(
                        query="query",
                        max_results=max_results,  # type: ignore[arg-type]
                    )
        with self.assertRaises(TypeError):
            RetrievalRequest(
                query="query",
                filters=("invalid",),  # type: ignore[arg-type]
            )
        duplicate = RetrievalFilter(field="world", value="harbor")
        with self.assertRaises(ValueError):
            RetrievalRequest(query="query", filters=(duplicate, duplicate))

        request = RetrievalRequest(query="query", max_results=3)
        with self.assertRaisesRegex(ValueError, "max_results>=3"):
            request.require_supported_by(create_descriptor(max_results=2))
        with self.assertRaisesRegex(ValueError, "kind:retrieval"):
            request.require_supported_by(
                CapabilityDescriptor(
                    capability_id="search",
                    version="1",
                    kind=CapabilityKind.SEARCH,
                    limits={CapabilityLimit.MAX_RESULTS: 3},
                )
            )


class RetrievedValueTest(unittest.TestCase):
    def test_freezes_document_and_chunk_metadata_without_exposing_values(self) -> None:
        document_metadata: dict[str, JsonValue] = {
            "world": "harbor",
            "tags": ["night"],
        }
        chunk_metadata: dict[str, JsonValue] = {"section": "opening"}
        document = RetrievedDocument(
            document_id="story-1",
            value=StoryDocument(title="Private title"),
            metadata=document_metadata,
        )
        chunk = RetrievedChunk(
            chunk_id="chunk-1",
            document_id=document.document_id,
            value=StoryChunk(text="Private passage"),
            metadata=chunk_metadata,
        )
        document_metadata["world"] = "changed"
        chunk_metadata["section"] = "changed"

        self.assertEqual(document.metadata["world"], "harbor")
        self.assertEqual(chunk.metadata["section"], "opening")
        self.assertNotIn("Private title", repr(document))
        self.assertNotIn("Private passage", repr(chunk))
        with self.assertRaises(TypeError):
            document.metadata["world"] = "other"  # type: ignore[index]

    def test_rejects_invalid_identity_or_metadata(self) -> None:
        with self.assertRaises(ValueError):
            RetrievedDocument(document_id=" padded ", value="document")
        with self.assertRaises(ValueError):
            RetrievedChunk(
                chunk_id="chunk",
                document_id="",
                value="content",
            )
        with self.assertRaises(ValueError):
            RetrievedDocument(
                document_id="document",
                value="content",
                metadata={"invalid": object()},  # type: ignore[dict-item]
            )


class RetrievalScoreAndMatchTest(unittest.TestCase):
    def test_accepts_explicit_non_vector_score_semantics(self) -> None:
        score = RetrievalScore(
            value=-12,
            metric="keyword_rank",
            higher_is_better=False,
        )

        self.assertEqual(score.value, -12.0)
        self.assertEqual(score.metric, "keyword_rank")
        self.assertFalse(score.higher_is_better)

    def test_rejects_invalid_scores_and_mismatched_chunks(self) -> None:
        for value in (math.inf, math.nan, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    RetrievalScore(
                        value=value,  # type: ignore[arg-type]
                        metric="rank",
                    )
        with self.assertRaises(ValueError):
            RetrievalScore(value=1, metric=" padded ")
        with self.assertRaises(TypeError):
            RetrievalScore(
                value=1,
                metric="rank",
                higher_is_better=1,  # type: ignore[arg-type]
            )

        document = create_document()
        with self.assertRaises(ValueError):
            RetrievalMatch(
                document=document,
                chunk=RetrievedChunk(
                    chunk_id="chunk",
                    document_id="other-document",
                    value=StoryChunk(text="content"),
                ),
                score=RetrievalScore(value=1, metric="rank"),
            )
        with self.assertRaises(TypeError):
            RetrievalMatch(
                document="invalid",  # type: ignore[arg-type]
                score=RetrievalScore(value=1, metric="rank"),
            )


class RetrievalResultTest(unittest.TestCase):
    def test_preserves_ranked_typed_documents_chunks_and_filters(self) -> None:
        document = create_document("1")
        matches = [
            create_match("1a", score=1.0, document=document),
            create_match("1b", score=0.8, document=document),
            create_match("2", score=0.6),
        ]
        request = RetrievalRequest(
            query="query",
            filters=(RetrievalFilter(field="world", value="harbor"),),
            max_results=3,
        )
        result = RetrievalResult(request=request, matches=matches)
        matches.clear()

        self.assertEqual(
            tuple(item.document_id for item in result.documents),
            ("story-1", "story-2"),
        )
        self.assertEqual(
            tuple(item.chunk_id for item in result.chunks),
            ("chunk-1a", "chunk-1b", "chunk-2"),
        )
        self.assertNotIn("Private", repr(result))

        lower_is_better = RetrievalResult(
            request=RetrievalRequest(query="query", max_results=2),
            matches=(
                create_match("a", score=1, higher_is_better=False),
                create_match("b", score=2, higher_is_better=False),
            ),
        )
        self.assertEqual(len(lower_is_better.matches), 2)

    def test_rejects_invalid_bounds_identity_filters_and_scores(self) -> None:
        with self.assertRaises(TypeError):
            RetrievalResult(
                request="invalid",  # type: ignore[arg-type]
                matches=(),
            )
        with self.assertRaises(TypeError):
            RetrievalResult(
                request=RetrievalRequest(query="query"),
                matches=("invalid",),  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            RetrievalResult(
                request=RetrievalRequest(query="query", max_results=1),
                matches=(create_match("1"), create_match("2")),
            )
        duplicate = create_match("1")
        with self.assertRaises(ValueError):
            RetrievalResult(
                request=RetrievalRequest(query="query", max_results=2),
                matches=(duplicate, duplicate),
            )

        document = create_document("1")
        conflicting = create_document("1", title="Changed")
        with self.assertRaises(ValueError):
            RetrievalResult(
                request=RetrievalRequest(query="query", max_results=2),
                matches=(
                    create_match("a", document=document),
                    create_match("b", document=conflicting, score=0.5),
                ),
            )
        with self.assertRaises(ValueError):
            RetrievalResult(
                request=RetrievalRequest(
                    query="query",
                    filters=(
                        RetrievalFilter(field="world", value="forest"),
                    ),
                ),
                matches=(create_match(),),
            )
        with self.assertRaises(ValueError):
            RetrievalResult(
                request=RetrievalRequest(query="query", max_results=2),
                matches=(
                    create_match("1", metric="keyword", score=1),
                    create_match("2", metric="semantic", score=0.5),
                ),
            )
        with self.assertRaises(ValueError):
            RetrievalResult(
                request=RetrievalRequest(query="query", max_results=2),
                matches=(
                    create_match("1", score=0.5),
                    create_match("2", score=1),
                ),
            )


class RetrieverContractTest(unittest.TestCase):
    def test_protocol_supports_storage_independent_typed_retrieval(self) -> None:
        retriever = ScriptedRetriever(
            descriptor=create_descriptor(max_results=2),
            batches=((create_match("1"), create_match("2", score=0.5)),),
        )
        request = RetrievalRequest(query="query", max_results=2)
        context = create_context()

        result = asyncio.run(retrieve_typed(retriever, request, context))

        self.assertIsInstance(retriever, Retriever)
        self.assertEqual(len(result.matches), 2)
        self.assertEqual(retriever.calls, [(request, context)])

    def test_preflight_and_constraints_run_before_consuming_a_batch(self) -> None:
        unsupported = ScriptedRetriever(
            descriptor=create_descriptor(max_results=1),
            batches=((create_match(),),),
        )
        request = RetrievalRequest(query="query", max_results=2)

        with self.assertRaises(ValueError):
            asyncio.run(unsupported.retrieve(request, create_context()))
        self.assertEqual(unsupported.calls, [])

        source = CancellationSource()
        source.cancel("caller stopped")
        cancelled = ScriptedRetriever(
            descriptor=create_descriptor(max_results=2),
            batches=((create_match(),),),
        )
        with self.assertRaises(RunCancelled):
            asyncio.run(cancelled.retrieve(request, create_context(source)))
        self.assertEqual(cancelled.calls, [])


if __name__ == "__main__":
    unittest.main()
