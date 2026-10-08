"""Provider-free review, artifact integrity, and preference activation tests."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agentrig.evals.comparison import (
    ComparisonCandidate, ComparisonCase, ComparisonCell, ComparisonManifest,
    ReviewAsset, ReviewProjection,
)
from agentrig.eval_review import ReviewApplication
from agentrig.selection import JsonPreferenceStore, ModelSet


class ReviewApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.root.chmod(0o700)
        self.token = "test-session-token-not-a-provider-secret"
        content = b"Synthetic prompt: compare two short answers."
        (self.root / "prompt.txt").write_bytes(content)
        self.asset = ReviewAsset(asset_id="prompt", kind="text", label="Original prompt",
            path="prompt.txt", sha256=hashlib.sha256(content).hexdigest(), media_type="text/plain")
        cases = (ComparisonCase(case_id="one", version="1", review=ReviewProjection(assets=(self.asset,))),
                 ComparisonCase(case_id="two", version="1"))
        candidates = tuple(ComparisonCandidate(candidate_id=name, version="1",
            model_set=ModelSet(application_id="synthetic", model_set_id=name, version="1", models={"text": name}))
            for name in ("a", "b"))
        cells = tuple(ComparisonCell(case_id=case.case_id, candidate_id=candidate.candidate_id, status="succeeded")
                      for case in cases for candidate in candidates)
        self.manifest = ComparisonManifest(comparison_id="synthetic-review", version="1",
            application_id="synthetic", cases=cases, candidates=candidates, cells=cells)
        self.store = JsonPreferenceStore(path=self.root / "preference.json")
        self.app = ReviewApplication(self.root, self.manifest, review_token=self.token, preference_store=self.store)

    def payload(self) -> dict[str, object]:
        return {"manifest_digest": self.manifest.digest, "activate_preference": True,
                "ballots": [{"case_id": case.case_id, "ranks": [["a"], ["b"]]} for case in self.manifest.cases]}

    def test_explicit_review_assets_only_and_paths_not_exposed(self) -> None:
        view = self.app.view()
        asset = view["cases"][0]["review"]["assets"][0]
        self.assertNotIn("path", asset)
        self.assertIn("Synthetic prompt", asset["text"])
        self.assertEqual(self.app.dispatch("GET", asset["url"]).status, 200)
        self.assertEqual(self.app.dispatch("GET", "/asset/../../prompt.txt").status, 404)
        self.assertEqual(self.app.dispatch("GET", "/prompt.txt").status, 404)

    def test_complete_finalization_saves_votes_and_preference_idempotently(self) -> None:
        result = self.app.finalize(self.payload())
        self.assertEqual(result["preference_status"], "updated")
        selected = self.store.read("synthetic")
        assert selected is not None
        self.assertEqual(selected.model_set.model_set_id, "a")
        self.assertEqual(self.app.finalize(self.payload()), result)
        reopened = ReviewApplication(self.root, self.manifest, review_token=self.token, preference_store=self.store)
        self.assertEqual(len(reopened.view()["saved_ballots"]), 2)

    def test_stale_artifact_cannot_receive_a_ballot(self) -> None:
        (self.root / "prompt.txt").write_bytes(b"Changed after review")
        with self.assertRaises(ValueError):
            self.app.finalize(self.payload())
        self.assertFalse((self.root / "rankings.json").exists())
        self.assertIsNone(self.store.read("synthetic"))

    def test_equivalent_case_and_tie_order_resubmission_is_idempotent(self) -> None:
        payload = self.payload()
        payload["ballots"] = [{"case_id": case.case_id, "ranks": [["b", "a"]]} for case in reversed(self.manifest.cases)]
        result = self.app.finalize(payload)
        payload["ballots"] = [{"case_id": case.case_id, "ranks": [["a", "b"]]} for case in self.manifest.cases]
        self.assertEqual(self.app.finalize(payload), result)

    def test_partial_or_failed_comparison_cannot_change_preference(self) -> None:
        incomplete = replace(self.manifest, cells=self.manifest.cells[:-1])
        app = ReviewApplication(self.root, incomplete, review_token=self.token, preference_store=self.store)
        payload = self.payload()
        payload["manifest_digest"] = incomplete.digest
        with self.assertRaises(ValueError):
            app.finalize(payload)
        self.assertIsNone(self.store.read("synthetic"))

    def test_ties_preserve_current_preference(self) -> None:
        payload = self.payload()
        payload["ballots"] = [{"case_id": case.case_id, "ranks": [["a", "b"]]} for case in self.manifest.cases]
        result = self.app.finalize(payload)
        self.assertEqual(result["preference_status"], "tied")
        self.assertIsNone(self.store.read("synthetic"))
        self.assertTrue((self.root / "rankings.json").exists())

    def test_csrf_token_required_for_mutation(self) -> None:
        body = json.dumps(self.payload()).encode()
        self.assertEqual(self.app.dispatch("POST", "/api/finalize", body=body).status, 403)
        self.assertFalse((self.root / "rankings.json").exists())
        self.assertEqual(self.app.dispatch("POST", "/api/finalize", body=body, review_token=self.token).status, 200)

    def test_symlink_asset_rejected(self) -> None:
        (self.root / "link.txt").symlink_to(self.root / "prompt.txt")
        asset = replace(self.asset, path="link.txt")
        manifest = replace(self.manifest, cases=(replace(self.manifest.cases[0], review=ReviewProjection(assets=(asset,))), self.manifest.cases[1]))
        with self.assertRaises(ValueError):
            ReviewApplication(self.root, manifest, review_token=self.token)

    def test_request_size_and_raw_error_redaction(self) -> None:
        self.assertEqual(self.app.dispatch("POST", "/api/finalize", body=b"x" * 200000, review_token=self.token).status, 413)
        reply = self.app.dispatch("POST", "/api/finalize", body=b'{"sensitive-field":"do-not-echo"}', review_token=self.token)
        self.assertEqual(reply.status, 409)
        self.assertNotIn(b"do-not-echo", reply.content)

    def test_deeply_nested_json_is_a_safe_request_error(self) -> None:
        body = b'{"x":' + b'[' * 2000 + b'0' + b']' * 2000 + b'}'
        self.assertEqual(self.app.dispatch("POST", "/api/finalize", body=body, review_token=self.token).status, 409)

    def test_oversized_ledger_is_rejected_before_publishing(self) -> None:
        with patch("agentrig.eval_review.server._MAX_LEDGER", 32):
            with self.assertRaises(ValueError):
                self.app.finalize(self.payload())
        self.assertFalse((self.root / "rankings.json").exists())
        self.assertIsNone(self.store.read("synthetic"))


if __name__ == "__main__":
    unittest.main()
