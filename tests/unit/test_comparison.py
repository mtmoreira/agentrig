from __future__ import annotations

import json
import unittest
from dataclasses import replace

from agentrig.core import ExecutionOutcome, Grade, GradeClassification, GradeStatus, GraderDescriptor, RunId
from agentrig.evals import (
    ComparisonCandidate, ComparisonCase, ComparisonCell, ComparisonCost,
    ComparisonManifest, EvalCase, EvalCaseResult, HumanRankingBallot,
    RankingLedger, RankingSummary, ReviewAsset, ReviewProjection, aggregate_rankings,
)
from agentrig.selection import ModelSet


def matrix(*, case_count: int = 2) -> ComparisonManifest:
    return ComparisonManifest(
        comparison_id="comparison-1", version="1", application_id="sample.books",
        cases=tuple(ComparisonCase(case_id=f"case-{index}", version="1") for index in range(case_count)),
        candidates=tuple(ComparisonCandidate(
            candidate_id=name, version="1", label=f"Candidate {name}",
            model_set=ModelSet(application_id="sample.books", model_set_id=name, version="1",
                               models={"image": f"scripted-{name}"}, settings={"quality": "standard"}),
        ) for name in ("a", "b", "c")),
        cells=tuple(ComparisonCell(case_id=f"case-{index}", candidate_id=name, status="succeeded")
                    for index in range(case_count) for name in ("a", "b", "c")),
    )


def ballot(manifest: ComparisonManifest, case: int, ranks: tuple[tuple[str, ...], ...]) -> HumanRankingBallot:
    return HumanRankingBallot(ballot_id=f"ballot-{case}", manifest_digest=manifest.digest,
                              case_id=f"case-{case}", case_version="1",
                              candidate_ids=manifest.candidate_ids, ranks=ranks)


class ComparisonTests(unittest.TestCase):
    def test_nine_effort_candidates_roundtrip_rank_and_bind_settings_to_ballots(self) -> None:
        candidates = tuple(ComparisonCandidate(
            candidate_id=f"model-{model}-{effort}", version="1",
            model_set=ModelSet(
                application_id="sample.books", model_set_id=f"model-{model}-{effort}", version="1",
                models={"text": f"scripted-{model}"}, settings={"reasoning_effort": effort}),
        ) for model in range(3) for effort in ("low", "medium", "high"))
        manifest = replace(matrix(case_count=1), candidates=candidates,
            cells=tuple(ComparisonCell(case_id="case-0", candidate_id=item.candidate_id,
                                      status="succeeded") for item in candidates))
        restored = ComparisonManifest.from_data(json.loads(json.dumps(manifest.to_data())))
        self.assertEqual(restored, manifest)
        self.assertTrue(restored.complete)
        ranking = ballot(manifest, 0, tuple((candidate,) for candidate in manifest.candidate_ids))
        ledger = RankingLedger(manifest_digest=manifest.digest).record(manifest, ranking)
        summary = aggregate_rankings(manifest, ledger)
        self.assertEqual(summary.winner_candidate_id, candidates[0].candidate_id)
        self.assertEqual(tuple(summary.scores.values()),
                         (1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0))
        changed = replace(candidates[0], model_set=replace(candidates[0].model_set,
                                                          settings={"reasoning_effort": "high"}))
        different = replace(manifest, candidates=(changed, *candidates[1:]))
        self.assertNotEqual(different.digest, manifest.digest)
        with self.assertRaises(ValueError):
            ranking.validate_for(different)

    def test_roundtrip_identity_detached_and_ordered_matrix(self) -> None:
        original = matrix()
        restored = ComparisonManifest.from_data(json.loads(json.dumps(original.to_data())))
        self.assertEqual(original, restored)
        self.assertEqual(original.digest, restored.digest)
        self.assertEqual(replace(original, cells=tuple(reversed(original.cells))).digest, original.digest)
        self.assertNotEqual(replace(original, version="2").digest, original.digest)
        changed = replace(original.candidates[0].model_set, settings={"quality": "high"})
        different = replace(original, candidates=(replace(original.candidates[0], model_set=changed), *original.candidates[1:]))
        self.assertNotEqual(different.digest, original.digest)
        serialized = json.loads(json.dumps(original.to_data()))
        serialized["candidates"][0]["model_set"]["models"]["image"] = "changed"
        self.assertEqual(original.candidates[0].model_set.models["image"], "scripted-a")

    def test_case_and_result_compose_with_existing_eval_contracts(self) -> None:
        case = EvalCase(case_id="case-0", version="1", input={"private": "original"})
        result = EvalCaseResult(case_id=case.case_id, case_version=case.version,
                                run_id=RunId("run-1"), outcome=ExecutionOutcome.succeeded("private result"),
                                duration_seconds=1.0)
        self.assertEqual(ComparisonCase.from_eval_case(case, review=ReviewProjection()).case_id, "case-0")
        cell = ComparisonCell.from_eval_result(result, candidate_id="a", review=ReviewProjection())
        self.assertEqual(cell.status, "succeeded")
        self.assertEqual(cell.run_id, "run-1")
        self.assertEqual(cell.cost.status, "unknown")
        self.assertNotIn("private result", repr(cell))

    def test_hard_grade_failure_blocks_but_soft_visual_grade_remains_advisory(self) -> None:
        result = EvalCaseResult(case_id="case-0", case_version="1", run_id=RunId("run-1"),
                                outcome=ExecutionOutcome.succeeded("synthetic output"), duration_seconds=1.0)
        grade = Grade(grader=GraderDescriptor(grader_id="smoke", version="1"), metric="renderable",
                      status=GradeStatus.FAILURE, classification=GradeClassification.HARD,
                      explanation="Synthetic hard requirement failed")
        failed = ComparisonCell.from_eval_result(replace(result, grades=(grade,)), candidate_id="a", review=ReviewProjection())
        self.assertEqual(failed.status, "failed")
        soft = replace(grade, classification=GradeClassification.SOFT)
        advisory = ComparisonCell.from_eval_result(replace(result, grades=(soft,)), candidate_id="a", review=ReviewProjection())
        self.assertEqual(advisory.status, "succeeded")

    def test_projection_collection_has_private_paths_and_portable_hashes(self) -> None:
        assets = [ReviewAsset(asset_id="photo", kind="image", label="PRIVATE LABEL",
                              path="private/photo.png", sha256="a" * 64, media_type="image/png")]
        projection = ReviewProjection(assets=tuple(assets))
        assets.clear()
        self.assertEqual(len(projection.assets), 1)
        self.assertEqual(ReviewProjection.from_data(projection.to_data()), projection)
        self.assertNotIn("private/photo", repr(projection.assets[0]))
        self.assertNotIn("PRIVATE LABEL", repr(projection.assets[0]))
        for path in ("/outside.png", "../outside.png", "a/../outside.png", "a//b", "a/./b", "C:\\private", "."):
            with self.subTest(path=path), self.assertRaises(ValueError):
                replace(projection.assets[0], path=path)
        with self.assertRaises(ValueError):
            replace(projection.assets[0], sha256="a" * 63)
        with self.assertRaises(ValueError):
            ReviewProjection(assets=(projection.assets[0], projection.assets[0]))

    def test_cost_unknown_estimated_observed_are_distinct(self) -> None:
        unknown = ComparisonCost()
        self.assertIsNone(unknown.amount)
        self.assertIsNone(unknown.currency)
        estimated = ComparisonCost(status="estimated", amount=0, currency="USD")
        observed = ComparisonCost(status="observed", amount=0, currency="USD")
        for cost in (unknown, estimated, observed):
            self.assertEqual(cost, ComparisonCost.from_data(cost.to_data()))
        self.assertNotEqual(unknown, estimated)
        self.assertNotEqual(estimated, observed)
        for value in (-1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ComparisonCost(status="observed", amount=value, currency="USD")
        with self.assertRaises(ValueError):
            ComparisonCost(amount=0)
        with self.assertRaises(ValueError):
            ComparisonCost(status="observed", amount=0, currency="usd")

    def test_matrix_rejects_duplicates_foreign_cells_and_wrong_scope(self) -> None:
        manifest = matrix()
        for cells in ((*manifest.cells, manifest.cells[0]), (replace(manifest.cells[0], case_id="foreign"),)):
            with self.assertRaises(ValueError):
                replace(manifest, cells=cells)
        with self.assertRaises(ValueError):
            replace(manifest, cases=())
        with self.assertRaises(ValueError):
            replace(manifest, candidates=(manifest.candidates[0],))
        with self.assertRaises(ValueError):
            replace(manifest, application_id="another.app")
        for field_name, value in (("schema_version", True), ("schema_version", 2), ("extra", "ignored")):
            data = manifest.to_data()
            data[field_name] = value
            with self.assertRaises(ValueError):
                ComparisonManifest.from_data(data)

    def test_pairwise_ties_case_weighting_and_unique_winner(self) -> None:
        manifest = matrix()
        ledger = RankingLedger(manifest_digest=manifest.digest)
        ledger = ledger.record(manifest, ballot(manifest, 0, (("a", "b"), ("c",))))
        ledger = ledger.record(manifest, ballot(manifest, 1, (("a",), ("c",), ("b",))))
        summary = aggregate_rankings(manifest, ledger)
        self.assertEqual(summary.scores, {"a": .875, "b": .375, "c": .25})
        self.assertEqual(summary.winner_candidate_id, "a")
        self.assertEqual(summary.reason, "complete")
        self.assertEqual(summary, RankingSummary.from_data(summary.to_data()))
        preference = summary.to_preference(manifest)
        self.assertEqual(preference.model_set, manifest.candidates[0].model_set)
        self.assertEqual(preference.manifest_digest, manifest.digest)
        self.assertNotIn("ballots", preference.to_data())
        self.assertNotIn("ranks", preference.to_data())
        restored = RankingLedger.from_data(json.loads(json.dumps(ledger.to_data())))
        self.assertEqual(restored, ledger)
        self.assertEqual(aggregate_rankings(manifest, restored), summary)
        reverse = replace(ledger, ballots=tuple(reversed(ledger.ballots)))
        self.assertEqual(aggregate_rankings(manifest, reverse), summary)

    def test_exact_aggregate_tie_never_uses_candidate_order_or_cost(self) -> None:
        manifest = matrix()
        ledger = RankingLedger(manifest_digest=manifest.digest, ballots=(
            ballot(manifest, 0, (("a",), ("b",), ("c",))),
            ballot(manifest, 1, (("c",), ("b",), ("a",))),
        ))
        summary = aggregate_rankings(manifest, ledger)
        self.assertEqual(summary.scores, {"a": .5, "b": .5, "c": .5})
        self.assertEqual(summary.reason, "tie")
        self.assertIsNone(summary.winner_candidate_id)
        with self.assertRaises(ValueError):
            summary.to_preference(manifest)

    def test_partial_ranking_and_missing_failed_or_pending_cell_never_select(self) -> None:
        manifest = matrix()
        ledger = RankingLedger(manifest_digest=manifest.digest).record(
            manifest, ballot(manifest, 0, (("a",), ("b",), ("c",))))
        summary = aggregate_rankings(manifest, ledger)
        self.assertEqual(summary.reason, "incomplete_rankings")
        with self.assertRaises(ValueError):
            summary.to_preference(manifest)
        variants = (manifest.cells[:-1],
                    (*manifest.cells[:-1], replace(manifest.cells[-1], status="failed")),
                    (*manifest.cells[:-1], replace(manifest.cells[-1], status="pending")))
        for cells in variants:
            incomplete = replace(manifest, cells=cells)
            partial = RankingLedger(manifest_digest=incomplete.digest).record(
                incomplete, ballot(incomplete, 0, (("a",), ("b",), ("c",))))
            result = aggregate_rankings(incomplete, partial)
            self.assertEqual(result.reason, "incomplete_matrix")
            self.assertIsNone(result.winner_candidate_id)
            with self.assertRaises(ValueError):
                result.to_preference(incomplete)
            with self.assertRaises(ValueError):
                ballot(incomplete, 1, (("a",), ("b",), ("c",))).validate_for(incomplete)

    def test_ballot_tampering_duplicates_and_staleness_fail_privately(self) -> None:
        manifest = matrix()
        original = ballot(manifest, 0, (("a",), ("b", "c")))
        for ranks in ((("a",), ("b",)), (("a",), ("b", "a")), (("a",), (), ("b", "c"))):
            with self.assertRaises(ValueError):
                replace(original, ranks=ranks)
        for changed in (replace(original, case_version="2"),
                        replace(original, manifest_digest="f" * 64),
                        replace(original, candidate_ids=("b", "a", "c"))):
            with self.assertRaises(ValueError):
                changed.validate_for(manifest)
        ledger = RankingLedger(manifest_digest=manifest.digest).record(manifest, original)
        self.assertIs(ledger.record(manifest, original), ledger)
        for changed in (replace(original, ranks=(("c",), ("a", "b"))),
                        replace(original, ballot_id="another-private-ballot")):
            with self.assertRaises(ValueError):
                ledger.record(manifest, changed)
        with self.assertRaises(ValueError):
            RankingLedger(manifest_digest=manifest.digest, ballots=(original, original))
        with self.assertRaises(ValueError):
            aggregate_rankings(replace(manifest, version="2"), ledger)
        self.assertEqual(repr(original), "HumanRankingBallot(<private>)")
        self.assertEqual(repr(ledger), "RankingLedger(<private>)")


if __name__ == "__main__":
    unittest.main()
