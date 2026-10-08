"""Compose normal EvalRunner execution, retained review files, and preferences."""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from agentrig.capabilities import (
    CapabilityDescriptor, CapabilityKind, GenerationUsage, ModelMetadata,
    TextGenerationFinishReason, TextGenerationRequest, TextGenerationResult,
    TextGenerator,
)
from agentrig.core import (
    CancellationSource, EventId, ExecutionOutcome, Grade, GradeClassification,
    GradeStatus, GraderDescriptor, RunContext, RunId,
)
from agentrig.evals import (
    ComparisonCandidate, ComparisonCase, ComparisonCell, ComparisonCost,
    ComparisonManifest, EvalCase, EvalDataset, EvalReport, EvalRunner, EvalSubject,
    EvalTargetDescriptor, EvalTargetKind, HumanRankingBallot, RankingLedger,
    RankingSummary, ReviewAsset, ReviewProjection, aggregate_rankings,
)
from agentrig.selection import JsonPreferenceStore, ModelSet, select_model_set
from agentrig.testing import ScriptedGrader, ScriptedTextGenerator


@dataclass(frozen=True, slots=True)
class Brief:
    topic: str


@dataclass(frozen=True, slots=True)
class Draft:
    text: str


@dataclass(frozen=True, slots=True)
class DraftTarget:
    generator: TextGenerator
    descriptor: EvalTargetDescriptor

    async def run(self, input: Brief, context: RunContext) -> ExecutionOutcome[Draft]:
        result = await self.generator.generate(TextGenerationRequest(prompt=input.topic), context)
        return ExecutionOutcome.succeeded(Draft(result.text))


@dataclass(slots=True)
class FixedClock:
    tick: float = 0.0

    def now(self) -> datetime:
        return datetime(2026, 1, 1, tzinfo=UTC)

    def monotonic(self) -> float:
        self.tick += 0.1
        return self.tick


@dataclass(slots=True)
class RunIds:
    next_value: int = 0

    def generate(self) -> RunId:
        self.next_value += 1
        return RunId(f"run-{self.next_value}")


@dataclass(slots=True)
class EventIds:
    next_value: int = 0

    def generate(self) -> EventId:
        self.next_value += 1
        return EventId(f"event-{self.next_value}")


def _write(root: Path, name: str, content: str) -> ReviewAsset:
    path = root / name
    payload = content.encode("utf-8")
    path.write_bytes(payload)
    path.chmod(0o600)
    return ReviewAsset(asset_id=name, kind="text", label=name, path=name,
                       sha256=hashlib.sha256(payload).hexdigest(), media_type="text/plain")


async def run_scripted(root: Path) -> tuple[ComparisonManifest, RankingLedger, RankingSummary]:
    """Write only synthetic private artifacts under the explicitly supplied root."""
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    dataset = EvalDataset(dataset_id="draft-comparison", version="1", cases=(
        EvalCase(case_id="garden", version="1", input=Brief("A robot grows a garden.")),
        EvalCase(case_id="moon", version="1", input=Brief("A kite reaches the moon.")),
    ))
    context = RunContext.create_root(clock=FixedClock(), id_generator=RunIds(),
                                     event_id_generator=EventIds(), cancellation=CancellationSource().token)
    cases = tuple(ComparisonCase.from_eval_case(case, review=ReviewProjection(assets=(
        _write(root, f"input-{case.case_id}.txt", case.input.topic),
    ))) for case in dataset.cases)
    candidates: list[ComparisonCandidate] = []
    cells: list[ComparisonCell] = []
    smoke = GraderDescriptor(grader_id="synthetic-smoke", version="1")
    for name in ("concise", "lyrical"):
        candidate = ComparisonCandidate(candidate_id=name, version="1", label=name.title(),
            model_set=ModelSet(application_id="example.drafts", model_set_id=name, version="1",
                               models={"writer": f"scripted-{name}"}, settings={"profile_version": "1"}))
        candidates.append(candidate)
        generator = ScriptedTextGenerator(
            descriptor=CapabilityDescriptor(capability_id=f"scripted-{name}", version="1", kind=CapabilityKind.TEXT_GENERATION),
            outcomes=tuple(TextGenerationResult(text=f"{name}: {case.input.topic}", usage=GenerationUsage(),
                model=ModelMetadata(provider="scripted", model_id=name), finish_reason=TextGenerationFinishReason.COMPLETED)
                for case in dataset.cases),
        )
        target = DraftTarget(generator, EvalTargetDescriptor(target_id="draft-writer", version=name, kind=EvalTargetKind.CAPABILITY))
        grader = ScriptedGrader[EvalSubject[Brief, Draft]](descriptor=smoke, outcomes=(
            Grade(grader=smoke, metric="renderable", status=GradeStatus.PASS,
                  classification=GradeClassification.HARD, explanation="Synthetic fixture is renderable"),
        ), repeat_last=True)
        result = await EvalRunner[Brief, Draft](target=target, graders=(grader,)).run(dataset, context)
        _write(root, f"report-{name}.json", EvalReport.from_run(result).to_json())
        for case_result in result.cases:
            output = _write(root, f"output-{case_result.case_id}-{name}.txt", case_result.outcome.unwrap().text)
            cells.append(ComparisonCell.from_eval_result(case_result, candidate_id=name,
                review=ReviewProjection(assets=(output,)), cost=ComparisonCost()))
    manifest = ComparisonManifest(comparison_id="synthetic-comparison", version="1", application_id="example.drafts",
                                  cases=cases, candidates=tuple(candidates), cells=tuple(cells))
    # A scripted stand-in for explicit human decisions, including a case-level tie.
    ledger = RankingLedger(manifest_digest=manifest.digest)
    for case, ranks in (("garden", (("lyrical",), ("concise",))), ("moon", (("concise", "lyrical"),))):
        ledger = ledger.record(manifest, HumanRankingBallot(ballot_id=f"synthetic-{case}", manifest_digest=manifest.digest,
            case_id=case, case_version="1", candidate_ids=manifest.candidate_ids, ranks=ranks))
    summary = aggregate_rankings(manifest, ledger)
    _write(root, "manifest.json", json.dumps(manifest.to_data(), sort_keys=True))
    _write(root, "rankings.json", json.dumps(ledger.to_data(), sort_keys=True))
    store = JsonPreferenceStore(path=root / "preference.json")
    store.write(summary.to_preference(manifest))
    selected = select_model_set(application_id=manifest.application_id, default=candidates[0].model_set,
                                preference=store.read(manifest.application_id))
    assert selected == candidates[1].model_set
    return manifest, ledger, summary


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="agentrig-comparison-") as directory:
        manifest, _, summary = asyncio.run(run_scripted(Path(directory)))
        print(json.dumps({"case_count": len(manifest.cases), "cell_count": len(manifest.cells),
                          "winner": summary.winner_candidate_id, "cost_status": "unknown"}, sort_keys=True))


if __name__ == "__main__":
    main()
