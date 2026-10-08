import test from "node:test";
import assert from "node:assert/strict";
import {ranksToGroups, costLabel, imageGroups, completeCaseCount} from "../../src/agentrig/eval_review/static/ranking.js";

test("inherited choices and invalid saved drafts never become votes", () => {
  assert.equal(ranksToGroups(["a", "b"], Object.create({a: 1, b: 2})), null);
  assert.equal(ranksToGroups(["a", "b"], null), null);
  assert.equal(ranksToGroups(["a", "b"], {a: "1", b: 2}), null);
});

test("unreviewed candidates never become implicit last place", () => {
  assert.equal(ranksToGroups(["a", "b", "c"], {a: 1, b: 2}), null);
  assert.equal(ranksToGroups(["a", "b"], {a: 1, b: null}), null);
  assert.equal(ranksToGroups(["a", "b"], {a: 1, b: 0}), null);
});

test("ties preserve roster ordering without forced tie breaking", () => {
  assert.deepEqual(ranksToGroups(["a", "b", "c"], {a: 2, b: 1, c: 1}), [["b", "c"], ["a"]]);
  assert.deepEqual(ranksToGroups(["a", "b"], {a: 1, b: 1}), [["a", "b"]]);
});

test("unknown cost never appears as a free run", () => {
  assert.equal(costLabel({status: "unknown", amount: null}), "Cost unavailable");
  assert.equal(costLabel({status: "estimated", amount: null, currency: "USD"}), "Cost unavailable");
  assert.equal(costLabel({status: "estimated", amount: 0.045, currency: "USD"}), "USD 0.045 · estimated");
  assert.equal(costLabel({status: "observed", amount: 0, currency: "USD"}), "USD 0.00 · reported");
});

test("synchronized views align by semantic label even when a candidate lacks an image", () => {
  const cells = [
    {review: {assets: [{kind: "image", label: "Cover"}, {kind: "image", label: "Page 2"}]}},
    {review: {assets: [{kind: "image", label: "Cover"}, {kind: "image", label: "Page 1"}, {kind: "artifact", label: "PDF"}]}}
  ];
  assert.deepEqual(imageGroups(cells), ["Cover", "Page 2", "Page 1"]);
});

test("partial case reviews remain pending", () => {
  const cases = [{case_id: "one"}, {case_id: "two"}, {case_id: "three"}];
  assert.equal(completeCaseCount(cases, ["a", "b"], {one: {a: 1, b: 2}, two: {a: 1}}), 1);
});
