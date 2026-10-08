import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import * as helpers from "../../src/agentrig/eval_review/static/ranking.js";

// Small DOM adapter lets the actual browser controller run without dependencies
// or sockets. This verifies interaction/state; it is not visual browser QA.
class Element {
  constructor(tag = "div") {
    this.tagName = tag; this.children = []; this.events = {}; this.attributes = {};
    this.textContent = ""; this.value = ""; this.disabled = false; this.checked = false;
    this.classList = {toggle() {}};
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this.attributes[name] = value; }
  addEventListener(name, handler) { this.events[name] = handler; }
  showModal() { this.open = true; }
  close() { this.open = false; }
}

function all(element, predicate) {
  return [element, ...element.children.flatMap(child => all(child, () => true))].filter(predicate);
}

function fixture() {
  const candidates = ["a", "b", "c"].map(id => ({candidate_id: id, label: id, model_set: {models: {text: id}}}));
  const cases = ["one", "two", "three"].map(case_id => ({case_id, review: {assets: [{kind: "text", label: "Prompt", text: "<img src=x> is quoted input, not markup."}]}}));
  return {comparison_id: "Synthetic comparison", manifest_digest: "digest", review_token: "test-review-token",
    candidates, cases, preference_enabled: true, saved_ballots: [], summary: null,
    cells: cases.flatMap(item => candidates.map(candidate => ({case_id: item.case_id, candidate_id: candidate.candidate_id,
      status: "succeeded", cost: {status: "unknown", amount: null}, review: {assets: ["Cover", "Page 1"].map(label => ({kind: "image", label, url: "/synthetic.png"}))}})))};
}

async function page(state, preferenceStatus = "updated") {
  const elements = new Map();
  const get = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
  const requests = [];
  const source = fs.readFileSync(new URL("../../src/agentrig/eval_review/static/review.js", import.meta.url), "utf8").replace(/^import.*\n/, "");
  const sandbox = { ...helpers,
    document: {getElementById: get, createElement: tag => new Element(tag)},
    localStorage: {getItem: () => null, setItem() {}},
    fetch: async (url, options) => {
      requests.push({url, options});
      return {ok: true, json: async () => url === "/api/comparison" ? state : {
        summary: {scores: {a: 1, b: 0.5, c: 0}, winner_candidate_id: "a"}, preference_status: preferenceStatus}};
    },
  };
  vm.runInNewContext(source, sandbox);
  await new Promise(setImmediate);
  return {get, requests, selects: () => all(get("candidates"), node => node.tagName === "select")};
}

test("actual controller starts unranked, aligns images and renders input as text", async () => {
  const view = await page(fixture());
  assert.equal(view.get("candidates").children.length, 3);
  assert.ok(view.selects().every(select => select.value === ""));
  assert.equal(view.get("submit").disabled, true);
  assert.equal(view.get("progress").textContent, "0 of 3 scenarios ranked");
  assert.equal(all(view.get("inputs"), node => node.tagName === "img").length, 0);
  assert.ok(all(view.get("inputs"), node => node.textContent.includes("<img src=x>")).length);
  view.get("next-asset").events.click();
  assert.equal(view.get("asset-label").textContent, "Page 1");
  assert.ok(all(view.get("candidates"), node => node.tagName === "img").every(image => image.alt === "Page 1"));
  assert.equal(view.requests.length, 1);
});

test("all scenarios produce digest-bound ballots with ties and correctly scaled scores", async () => {
  const view = await page(fixture());
  for (let index = 0; index < 3; index++) {
    view.get("cases").children[index].events.click();
    view.selects().forEach((select, candidate) => { select.value = candidate < 2 ? "1" : "3"; select.events.change(); });
    assert.equal(view.get("submit").disabled, index < 2);
  }
  await view.get("submit").events.click();
  const request = view.requests.find(item => item.url === "/api/finalize");
  const body = JSON.parse(request.options.body);
  assert.equal(body.manifest_digest, "digest");
  assert.equal(body.activate_preference, true);
  assert.equal(request.options.headers["X-Review-Token"], "test-review-token");
  assert.deepEqual(body.ballots.map(item => item.ranks), Array.from({length: 3}, () => [["a", "b"], ["c"]]));
  assert.ok(all(view.get("scores"), node => node.textContent === "100.0 / 100").length);
  assert.ok(view.selects().every(select => select.disabled));
  assert.equal(view.get("submit").disabled, true);
});

test("finalized votes are locked on reload while preference activation can be retried", async () => {
  const state = fixture();
  state.saved_ballots = state.cases.map(item => ({case_id: item.case_id, ranks: [["a"], ["b"], ["c"]]}));
  state.summary = {scores: {a: 1, b: 0.5, c: 0}, winner_candidate_id: "a"};
  const view = await page(state, "failed");
  assert.ok(view.selects().every(select => select.disabled));
  assert.equal(view.get("submit").textContent, "Apply saved winner");
  assert.equal(view.get("submit").disabled, false);
  await view.get("submit").events.click();
  assert.ok(view.selects().every(select => select.disabled));
  assert.equal(view.get("submit").disabled, false);
  assert.match(view.get("message").textContent, /rankings are preserved/);
});

test("image captions follow the selected result without rendering markup", async () => {
  const state = fixture();
  for (const cell of state.cells) {
    cell.review.assets[1].asset_id = "scene";
    cell.review.assets.push({asset_id: "scene-caption", kind: "text", media_type: "text/plain", label: "Scene text", text: "We found a quiet beach <together>."});
  }
  const view = await page(state);
  assert.equal(all(view.get("candidates"), node => node.className === "image-caption").length, 0);
  view.get("next-asset").events.click();
  const captions = all(view.get("candidates"), node => node.className === "image-caption");
  assert.equal(captions.length, 3);
  assert.ok(captions.every(caption => caption.textContent === "We found a quiet beach <together>."));
});
