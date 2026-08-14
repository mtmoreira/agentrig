"""Deterministic executable for the offline eval regression gate."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType

from agentrig.core import (
    CancellationSource,
    Event,
    EventId,
    ExecutionOutcome,
    Failure,
    FailureKind,
    InMemoryEventSink,
    RunContext,
    RunId,
)
from agentrig.evals import (
    EvalBaseline,
    EvalReport,
    EvalRunResult,
    EvalTargetDescriptor,
    EvalTargetKind,
)

from examples.evals.regression_gate.suite import (
    RELEASE_DATASET,
    RegressionGateAssessment,
    ReleaseRequest,
    ReleaseSummary,
    approve_baseline,
    assess_candidate,
    create_eval_runner,
    create_private_report,
)

APPROVED_OUTPUTS: Mapping[str, str] = MappingProxyType(
    {
        "alpha": "Typed workflow composition preserves portable contracts.",
        "beta": "A baseline makes every regression gate deterministic.",
    }
)
REGRESSED_OUTPUTS: Mapping[str, str] = MappingProxyType(
    {
        "alpha": APPROVED_OUTPUTS["alpha"],
        "beta": "Release checks remain deterministic.",
    }
)


@dataclass(slots=True)
class DeterministicClock:
    monotonic_time: float = 100.0
    increment: float = 0.25

    def now(self) -> datetime:
        return datetime(2026, 8, 14, 17, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        current = self.monotonic_time
        self.monotonic_time += self.increment
        return current


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
class ScriptedTargetCall:
    input: ReleaseRequest = field(repr=False)
    context: RunContext = field(repr=False)


@dataclass(slots=True, kw_only=True)
class ScriptedReleaseTarget:
    """Return configured outputs or blocked outcomes for dataset cases."""

    version: str
    outputs: Mapping[str, str] = field(repr=False)
    blocked_release_ids: frozenset[str] = frozenset()
    calls: list[ScriptedTargetCall] = field(default_factory=list, init=False)
    descriptor: EvalTargetDescriptor = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("scripted target version must not be empty")
        if not isinstance(self.outputs, Mapping):
            raise TypeError("scripted target outputs must be a mapping")
        copied_outputs: dict[str, str] = {}
        for release_id, output in self.outputs.items():
            if (
                not isinstance(release_id, str)
                or not release_id.strip()
                or not isinstance(output, str)
                or not output.strip()
            ):
                raise ValueError("scripted target outputs must contain text")
            copied_outputs[release_id] = output
        self.outputs = MappingProxyType(copied_outputs)
        blocked_release_ids = frozenset(self.blocked_release_ids)
        if any(
            not isinstance(release_id, str) or not release_id.strip()
            for release_id in blocked_release_ids
        ):
            raise ValueError("blocked release IDs must contain text")
        self.blocked_release_ids = blocked_release_ids
        self.descriptor = EvalTargetDescriptor(
            target_id="release-note.workflow",
            version=self.version,
            kind=EvalTargetKind.WORKFLOW,
        )

    async def run(
        self,
        input: ReleaseRequest,
        context: RunContext,
    ) -> ExecutionOutcome[ReleaseSummary]:
        if not isinstance(input, ReleaseRequest):
            raise TypeError("scripted release target requires a ReleaseRequest")
        if not isinstance(context, RunContext):
            raise TypeError("scripted release target context must be a RunContext")
        context.cancellation.raise_if_cancelled()
        if context.deadline is not None:
            context.deadline.raise_if_expired(context.clock)
        self.calls.append(ScriptedTargetCall(input=input, context=context))

        if input.release_id in self.blocked_release_ids:
            return ExecutionOutcome.from_failure(
                Failure(
                    kind=FailureKind.WORKFLOW_BLOCKED,
                    message="release target dependency is unavailable",
                    code="example.release_target_blocked",
                )
            )
        output = self.outputs.get(input.release_id)
        if output is None:
            return ExecutionOutcome.from_failure(
                Failure(
                    kind=FailureKind.INVALID_INPUT,
                    message="release target has no configured case output",
                    code="example.release_output_missing",
                )
            )
        return ExecutionOutcome.succeeded(ReleaseSummary(text=output))


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedEvalRun:
    run: EvalRunResult[ReleaseSummary]
    report: EvalReport
    target: ScriptedReleaseTarget
    events: tuple[Event, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedGateRun:
    baseline: EvalBaseline
    approved: ScriptedEvalRun
    candidate: ScriptedEvalRun
    assessment: RegressionGateAssessment


def create_context() -> tuple[RunContext, InMemoryEventSink]:
    sink = InMemoryEventSink()
    context = RunContext.create_root(
        clock=DeterministicClock(),
        id_generator=SequentialRunIdGenerator(),
        cancellation=CancellationSource().token,
        event_sink=sink,
        event_id_generator=SequentialEventIdGenerator(),
        correlation={"example": "eval-regression-gate"},
    )
    return context, sink


async def run_scripted_report(
    *,
    version: str,
    outputs: Mapping[str, str],
    blocked_release_ids: frozenset[str] = frozenset(),
) -> ScriptedEvalRun:
    target = ScriptedReleaseTarget(
        version=version,
        outputs=outputs,
        blocked_release_ids=blocked_release_ids,
    )
    runner = create_eval_runner(target)
    context, sink = create_context()
    run = await runner.run(RELEASE_DATASET, context)
    return ScriptedEvalRun(
        run=run,
        report=create_private_report(run),
        target=target,
        events=sink.events,
    )


async def run_regression_gate() -> ScriptedGateRun:
    approved = await run_scripted_report(
        version="1",
        outputs=APPROVED_OUTPUTS,
    )
    candidate = await run_scripted_report(
        version="2",
        outputs=REGRESSED_OUTPUTS,
    )
    baseline = approve_baseline(approved.report)
    return ScriptedGateRun(
        baseline=baseline,
        approved=approved,
        candidate=candidate,
        assessment=assess_candidate(
            baseline=baseline,
            candidate=candidate.report,
        ),
    )


def main() -> None:
    gate = asyncio.run(run_regression_gate())
    summary = {
        "baseline_case_count": gate.baseline.report.summary.case_count,
        "baseline_payloads_retained": (
            gate.baseline.report.retention.outputs
            or gate.baseline.report.retention.artifacts
            or gate.baseline.report.retention.grade_details
        ),
        "candidate_target_version": (
            gate.assessment.comparison.candidate_target_version
        ),
        "decision": gate.assessment.decision.value,
        "regression_metrics": sorted(
            change.metric.value
            for change in gate.assessment.comparison.regressions
        ),
        "report_round_trip": (
            EvalReport.from_json(gate.candidate.report.to_json())
            == gate.candidate.report
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
