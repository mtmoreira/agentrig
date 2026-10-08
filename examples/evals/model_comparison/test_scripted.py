from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agentrig.evals import ComparisonManifest, RankingLedger, aggregate_rankings
from agentrig.selection import JsonPreferenceStore
from examples.evals.model_comparison.scripted import run_scripted


class ModelComparisonExampleTest(unittest.TestCase):
    def test_executes_same_cases_and_persists_only_explicit_review_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, ledger, summary = asyncio.run(run_scripted(root))
            self.assertEqual(len(manifest.cells), 4)
            self.assertTrue(manifest.complete)
            self.assertEqual(summary.scores, {"concise": .25, "lyrical": .75})
            self.assertEqual(summary.winner_candidate_id, "lyrical")
            for review in (*(case.review for case in manifest.cases), *(cell.review for cell in manifest.cells)):
                for asset in review.assets:
                    self.assertEqual(hashlib.sha256((root / asset.path).read_bytes()).hexdigest(), asset.sha256)
            restored = ComparisonManifest.from_data(json.loads((root / "manifest.json").read_text()))
            stored_votes = RankingLedger.from_data(json.loads((root / "rankings.json").read_text()))
            self.assertEqual(restored, manifest)
            self.assertEqual(stored_votes, ledger)
            self.assertEqual(aggregate_rankings(restored, stored_votes), summary)
            preferred = JsonPreferenceStore(path=root / "preference.json").read("example.drafts")
            self.assertIsNotNone(preferred)
            assert preferred is not None
            self.assertEqual(preferred.model_set.model_set_id, "lyrical")
            self.assertNotIn("ranks", preferred.to_data())
            self.assertTrue(all(cell.cost.status == "unknown" for cell in manifest.cells))
            self.assertNotIn("A robot grows", (root / "report-concise.json").read_text())


if __name__ == "__main__":
    unittest.main()
