"""Deterministic executable for the sourced-digest capability pipeline."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    DataRetention,
    GenerationUsage,
    ModelMetadata,
    SearchHit,
    SearchRetrievalMetadata,
    TextGenerationFinishReason,
)
from agentrig.core import (
    CancellationSource,
    Event,
    EventId,
    ExecutionOutcome,
    InMemoryEventSink,
    JsonValue,
    RunContext,
    RunId,
)
from agentrig.testing import (
    ScriptedSearchProvider,
    ScriptedSearchScenario,
    ScriptedStructuredGeneration,
    ScriptedStructuredGenerator,
)
from agentrig.workflow import Workflow

from examples.capabilities.sourced_digest.workflow import (
    DigestRequest,
    GeneratedDigest,
    SourcedDigest,
    build_sourced_digest_workflow,
)

SOURCE_ONE = "https://example.test/portable-contracts"
SOURCE_TWO = "https://example.test/deterministic-tests"


@dataclass(frozen=True, slots=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 14, 16, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


@dataclass(slots=True)
class SequentialRunIdGenerator:
    next_value: int = 1

    def generate(self) -> RunId:
        run_id = RunId(f"run-{self.next_value}")
        self.next_value += 1
        return run_id


@dataclass(slots=True)
class SequentialEventIdGenerator:
    next_value: int = 1

    def generate(self) -> EventId:
        event_id = EventId(f"event-{self.next_value}")
        self.next_value += 1
        return event_id


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedExample:
    workflow: Workflow[DigestRequest, SourcedDigest]
    search_provider: ScriptedSearchProvider
    generator: ScriptedStructuredGenerator[GeneratedDigest]


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedRun:
    outcome: ExecutionOutcome[SourcedDigest]
    events: tuple[Event, ...]
    search_calls: int
    generation_calls: int


def create_scripted_example(
    *,
    search_capacity: int = 2,
    encoded_output: JsonValue | None = None,
) -> ScriptedExample:
    search_provider = ScriptedSearchProvider(
        descriptor=CapabilityDescriptor(
            capability_id="scripted.search",
            version="1",
            kind=CapabilityKind.SEARCH,
            features=frozenset({CapabilityFeature.CITATIONS}),
            limits={CapabilityLimit.MAX_RESULTS: search_capacity},
            data_retention=DataRetention.NOT_RETAINED,
        ),
        outcomes=(
            ScriptedSearchScenario(
                hits=(
                    SearchHit(
                        source_uri=SOURCE_ONE,
                        title="Portable contracts",
                        summary=(
                            "Portable contracts separate application policy "
                            "from provider implementations."
                        ),
                    ),
                    SearchHit(
                        source_uri=SOURCE_TWO,
                        title="Deterministic capability tests",
                        excerpt=(
                            "Injected scripted capabilities make workflow "
                            "behavior reproducible."
                        ),
                    ),
                ),
                metadata=SearchRetrievalMetadata(
                    retrieved_at=FixedClock().now(),
                    duration_seconds=0.02,
                    total_available=2,
                ),
            ),
        ),
    )
    generator = ScriptedStructuredGenerator[GeneratedDigest](
        descriptor=CapabilityDescriptor(
            capability_id="scripted.structured-generation",
            version="1",
            kind=CapabilityKind.STRUCTURED_GENERATION,
            features=frozenset({CapabilityFeature.STRUCTURED_OUTPUT}),
            limits={CapabilityLimit.MAX_OUTPUT_TOKENS: 256},
            data_retention=DataRetention.NOT_RETAINED,
        ),
        outcomes=(
            ScriptedStructuredGeneration(
                encoded_output=(
                    encoded_output
                    if encoded_output is not None
                    else {
                        "headline": "Portable, testable capability pipelines",
                        "summary": (
                            "Bounded search and strict generation can be "
                            "composed without selecting a provider."
                        ),
                        "source_uris": [SOURCE_ONE, SOURCE_TWO],
                    }
                ),
                usage=GenerationUsage(input_tokens=72, output_tokens=24),
                model=ModelMetadata(
                    provider="scripted",
                    model_id="structured-1",
                ),
                finish_reason=TextGenerationFinishReason.COMPLETED,
            ),
        ),
    )
    return ScriptedExample(
        workflow=build_sourced_digest_workflow(
            search_provider=search_provider,
            generator=generator,
        ),
        search_provider=search_provider,
        generator=generator,
    )


def create_context(
    source: CancellationSource | None = None,
) -> tuple[RunContext, InMemoryEventSink]:
    owned_source = source if source is not None else CancellationSource()
    sink = InMemoryEventSink()
    context = RunContext.create_root(
        clock=FixedClock(),
        id_generator=SequentialRunIdGenerator(),
        cancellation=owned_source.token,
        event_sink=sink,
        event_id_generator=SequentialEventIdGenerator(),
        correlation={"example": "sourced-digest"},
    )
    return context, sink


async def run_scripted_example() -> ScriptedRun:
    configured = create_scripted_example()
    context, sink = create_context()
    outcome = await configured.workflow.execute(
        DigestRequest(topic="provider-neutral AI capabilities"),
        context,
    )
    return ScriptedRun(
        outcome=outcome,
        events=sink.events,
        search_calls=len(configured.search_provider.calls),
        generation_calls=len(configured.generator.calls),
    )


def main() -> None:
    run = asyncio.run(run_scripted_example())
    result = run.outcome.unwrap()
    summary = {
        "citations": [
            {
                "source_uri": citation.source_uri,
                "title": citation.title,
            }
            for citation in result.citations
        ],
        "event_kinds": [event.kind.value for event in run.events],
        "generation_calls": run.generation_calls,
        "generation_model": result.generation_model.model_id,
        "generation_tokens": result.generation_usage.total_tokens,
        "headline": result.headline,
        "search_calls": run.search_calls,
        "summary": result.summary,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
