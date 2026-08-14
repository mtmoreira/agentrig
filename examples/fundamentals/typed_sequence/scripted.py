"""Deterministic executable for typed sequences and effect-aware retries."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agentrig.core import (
    AgentRigError,
    CancellationSource,
    EffectProfile,
    Event,
    EventId,
    EventKind,
    ExecutionOutcome,
    Failure,
    FailureKind,
    InMemoryEventSink,
    RunContext,
    RunId,
)
from agentrig.workflow import StepDescriptor, Workflow

from examples.fundamentals.typed_sequence.workflow import (
    ClassifiedRequest,
    NormalizedRequest,
    RawRequest,
    RequestSummary,
    build_typed_sequence,
)


@dataclass(frozen=True, slots=True)
class FixedClock:
    """Keep event timestamps stable across runs."""

    def now(self) -> datetime:
        return datetime(2026, 8, 14, 12, 0, tzinfo=UTC)

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
class ClassifierCall:
    input: NormalizedRequest
    context: RunContext


@dataclass(slots=True)
class ScriptedClassifier:
    """Fail transiently once, then classify the normalized request."""

    effect_profile: EffectProfile = EffectProfile.IDEMPOTENT
    failures_before_success: int = 1
    calls: list[ClassifierCall] = field(default_factory=list)
    descriptor: StepDescriptor = field(init=False)

    def __post_init__(self) -> None:
        self.descriptor = StepDescriptor(
            step_id="request.classify",
            version="1",
            effect_profile=self.effect_profile,
        )

    async def run(
        self,
        input: NormalizedRequest,
        context: RunContext,
    ) -> ClassifiedRequest:
        self.calls.append(ClassifierCall(input=input, context=context))
        if len(self.calls) <= self.failures_before_success:
            raise AgentRigError(
                Failure(
                    kind=FailureKind.TRANSIENT_PROVIDER,
                    message="classifier temporarily unavailable",
                    code="example.classifier_busy",
                )
            )
        category = "question" if input.text.endswith("?") else "statement"
        return ClassifiedRequest(text=input.text, category=category)


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedExample:
    workflow: Workflow[RawRequest, RequestSummary]
    classifier: ScriptedClassifier


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedRun:
    outcome: ExecutionOutcome[RequestSummary]
    events: tuple[Event, ...]
    classifier_calls: tuple[ClassifierCall, ...]


def create_scripted_example(
    *,
    effect_profile: EffectProfile = EffectProfile.IDEMPOTENT,
) -> ScriptedExample:
    classifier = ScriptedClassifier(effect_profile=effect_profile)
    return ScriptedExample(
        workflow=build_typed_sequence(classifier=classifier, max_attempts=2),
        classifier=classifier,
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
        correlation={"example": "typed-sequence"},
    )
    return context, sink


async def run_scripted_example(
    *,
    effect_profile: EffectProfile = EffectProfile.IDEMPOTENT,
) -> ScriptedRun:
    configured = create_scripted_example(effect_profile=effect_profile)
    context, sink = create_context()
    outcome = await configured.workflow.execute(
        RawRequest(text="  How   do retries work?  "),
        context,
    )
    return ScriptedRun(
        outcome=outcome,
        events=sink.events,
        classifier_calls=tuple(configured.classifier.calls),
    )


def main() -> None:
    run = asyncio.run(run_scripted_example())
    result = run.outcome.unwrap()
    started_events = tuple(
        event for event in run.events if event.kind is EventKind.STEP_STARTED
    )
    summary = {
        "category": result.category,
        "character_count": result.character_count,
        "classifier_attempts": len(run.classifier_calls),
        "event_kinds": [event.kind.value for event in run.events],
        "message": result.message,
        "step_parent_run_ids": [
            str(event.parent_run_id) for event in started_events
        ],
        "step_run_ids": [str(event.run_id) for event in started_events],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
