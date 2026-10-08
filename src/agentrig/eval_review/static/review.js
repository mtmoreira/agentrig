import {ranksToGroups, costLabel, imageGroups, completeCaseCount} from "/static/ranking.js";

const $ = id => document.getElementById(id);
let state;
let selectedCase = 0;
let selectedImage = 0;
let ranks = Object.create(null);
let submitted = false;
let ranksLocked = false;

function node(tag, className, text) {
  const result = document.createElement(tag);
  if (className) result.className = className;
  if (text !== undefined) result.textContent = text;
  return result;
}

function setMessage(text, error = false) {
  $("message").textContent = text;
  $("message").classList.toggle("error", error);
}

function draftKey() { return `agentrig-ranking:${state.manifest_digest}`; }

function persistDraft() {
  try { localStorage.setItem(draftKey(), JSON.stringify(ranks)); } catch { /* Optional convenience only. */ }
}

function zoom(asset) {
  $("zoom-image").src = asset.url;
  $("zoom-image").alt = asset.label;
  $("zoom").showModal();
}

function renderInput(asset) {
  if (asset.kind === "image") {
    const figure = node("figure", "reference");
    const img = node("img");
    img.src = asset.url; img.alt = asset.label;
    img.tabIndex = 0;
    img.addEventListener("click", () => zoom(asset));
    img.addEventListener("keydown", event => { if (event.key === "Enter") zoom(asset); });
    figure.append(img, node("figcaption", "", asset.label));
    return figure;
  }
  const panel = node("div", "input-text");
  panel.append(node("h3", "", asset.label));
  if (asset.kind === "text") panel.append(node("pre", "", asset.text || ""));
  else { const link = node("a", "", "Open input artifact"); link.href = asset.url; link.target = "_blank"; link.rel = "noopener"; panel.append(link); }
  return panel;
}

function updateProgress() {
  const ids = state.candidates.map(item => item.candidate_id);
  const count = completeCaseCount(state.cases, ids, ranks);
  $("progress").textContent = `${count} of ${state.cases.length} scenarios ranked`;
  const matrixComplete = state.cells.length === state.cases.length * ids.length && state.cells.every(cell => cell.status === "succeeded");
  $("submit").disabled = submitted || count !== state.cases.length || !matrixComplete || (ranksLocked && !$("activate-preference").checked);
  if (!matrixComplete) setMessage("Some results are failed or pending. You can inspect them, but this comparison cannot update the preferred configuration.");
}

function renderCard(candidate, cell, label) {
  const card = node("article", "candidate-card");
  const header = node("div", "candidate-header");
  header.append(node("h3", "", candidate.label || candidate.candidate_id));
  const metrics = node("div", "metric-line");
  metrics.append(node("span", `cost ${cell.cost.status === "unknown" ? "unknown" : ""}`, costLabel(cell.cost)));
  metrics.append(node("span", `status ${cell.status === "failed" ? "failed" : ""}`, cell.status));
  header.append(metrics); card.append(header);
  const image = cell.review.assets.find(asset => asset.kind === "image" && asset.label === label);
  if (image) {
    const img = node("img", "result-image"); img.src = image.url; img.alt = image.label; img.tabIndex = 0;
    img.addEventListener("click", () => zoom(image));
    img.addEventListener("keydown", event => { if (event.key === "Enter") zoom(image); });
    card.append(img);
  } else if (cell.review.assets.some(asset => asset.kind === "text") && !label) {
    const result = node("div", "candidate-body");
    result.append(node("pre", "evidence-text", cell.review.assets.filter(asset => asset.kind === "text").map(asset => asset.text).join("\n\n")));
    card.append(result);
  } else card.append(node("div", "empty-result", cell.status === "succeeded" ? "No image for this view. Inspect the artifacts below." : "This result is unavailable. Failure evidence remains available below."));
  const body = node("div", "candidate-body");
  // Optional projection convention works for any image result, not just books.
  const caption = image && cell.review.assets.find(asset => asset.kind === "text" && asset.media_type === "text/plain" && asset.asset_id === `${image.asset_id}-caption`);
  if (caption) body.append(node("p", "image-caption", caption.text || ""));
  const rankLabel = node("label", "rank-control", "Your preference");
  const select = node("select"); select.setAttribute("aria-label", `Rank ${candidate.label || candidate.candidate_id}`);
  const initial = node("option", "", "Unranked"); initial.value = ""; select.append(initial);
  for (let number = 1; number <= state.candidates.length; number++) {
    const option = node("option", "", `${number}${number === 1 ? " · favorite" : ""}`); option.value = String(number); select.append(option);
  }
  const current = state.cases[selectedCase].case_id;
  select.value = ranks[current]?.[candidate.candidate_id] ? String(ranks[current][candidate.candidate_id]) : "";
  select.disabled = ranksLocked || submitted || cell.status !== "succeeded";
  select.addEventListener("change", () => {
    if (!Object.hasOwn(ranks, current)) ranks[current] = Object.create(null);
    ranks[current][candidate.candidate_id] = select.value ? Number(select.value) : null;
    persistDraft(); updateProgress();
  });
  rankLabel.append(select); body.append(rankLabel);
  const downloads = node("div", "downloads");
  for (const asset of cell.review.assets.filter(item => item.kind === "artifact")) {
    const link = node("a", "", asset.label); link.href = asset.url; link.target = "_blank"; link.rel = "noopener"; downloads.append(link);
  }
  body.append(downloads);
  const details = node("details", "candidate-detail"); details.append(node("summary", "", "Models, settings and evidence"));
  details.append(node("pre", "", JSON.stringify(candidate.model_set, null, 2)));
  for (const asset of cell.review.assets.filter(item => item.kind === "text" && item !== caption)) {
    details.append(node("strong", "muted", asset.label), node("pre", "", asset.text || ""));
  }
  body.append(details); card.append(body); return card;
}

function renderCase() {
  const current = state.cases[selectedCase];
  $("case-title").textContent = current.case_id.replaceAll("_", " ").replaceAll("-", " ");
  $("case-counter").textContent = `Scenario ${selectedCase + 1} / ${state.cases.length}`;
  $("inputs").replaceChildren(...current.review.assets.map(renderInput));
  $("cases").replaceChildren(...state.cases.map((item, index) => {
    const button = node("button"); button.type = "button"; button.setAttribute("aria-current", String(index === selectedCase));
    button.append(node("span", "case-number", String(index + 1)), node("span", "case-name", item.case_id.replaceAll("_", " ").replaceAll("-", " ")));
    button.addEventListener("click", () => { selectedCase = index; selectedImage = 0; renderCase(); }); return button;
  }));
  const cells = state.cells.filter(cell => cell.case_id === current.case_id);
  const labels = imageGroups(cells);
  selectedImage = Math.min(selectedImage, Math.max(labels.length - 1, 0));
  const label = labels[selectedImage] || "";
  $("asset-label").textContent = label || "Text results";
  $("previous-asset").disabled = selectedImage <= 0;
  $("next-asset").disabled = selectedImage >= labels.length - 1;
  $("candidates").replaceChildren(...state.candidates.map(candidate => {
    const cell = cells.find(item => item.candidate_id === candidate.candidate_id) || {status: "pending", review: {assets: []}, cost: {status: "unknown"}};
    return renderCard(candidate, cell, label);
  }));
  updateProgress();
}

function renderSummary(summary, preferenceStatus) {
  if (!summary) return;
  const panel = $("scores"); panel.hidden = false;
  panel.replaceChildren(node("h3", "", "Your comparison scores"));
  for (const candidate of state.candidates) {
    const row = node("div", "score-row"); const score = summary.scores?.[candidate.candidate_id];
    row.append(node("span", "", candidate.label || candidate.candidate_id), node("strong", "", Number.isFinite(score) ? `${(score * 100).toFixed(1)} / 100` : "Pending")); panel.append(row);
  }
  const messages = {updated: "Preferred configuration saved for future runs.", unchanged: "Rankings saved. Your current preference is unchanged.", tied: "The top score is tied. Your current preference is unchanged.", unavailable: "Rankings saved. No preference storage was configured.", failed: "Rankings saved, but the preferred configuration could not be saved. Check the configured preference path and submit again."};
  panel.append(node("p", "muted", messages[preferenceStatus] || "Rankings saved."));
}

async function submit() {
  const candidateIds = state.candidates.map(item => item.candidate_id);
  const ballots = state.cases.map(item => ({case_id: item.case_id, ranks: ranksToGroups(candidateIds, ranks[item.case_id] || {})}));
  if (ballots.some(item => item.ranks === null)) return;
  $("submit").disabled = true; setMessage("Saving your rankings…");
  try {
    const response = await fetch("/api/finalize", {method: "POST", headers: {"Content-Type": "application/json", "X-Review-Token": state.review_token}, body: JSON.stringify({manifest_digest: state.manifest_digest, ballots, activate_preference: $("activate-preference").checked})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Ranking could not be saved.");
    ranksLocked = true;
    submitted = result.preference_status !== "failed";
    renderSummary(result.summary, result.preference_status); renderCase();
    setMessage(submitted ? "Ranking complete. No new generation calls were made." : "Preference update failed; your rankings are preserved. You can retry saving the same preference.", !submitted);
  } catch (error) { setMessage(error instanceof Error ? error.message : "Ranking could not be saved.", true); updateProgress(); }
}

async function start() {
  try {
    const response = await fetch("/api/comparison", {cache: "no-store"});
    if (!response.ok) throw new Error("Could not read this comparison. Check the manifest and artifact hashes.");
    state = await response.json();
    $("experiment-title").textContent = state.comparison_id;
    try { const draft = JSON.parse(localStorage.getItem(draftKey()) || "{}"); if (draft && typeof draft === "object" && !Array.isArray(draft)) ranks = Object.assign(Object.create(null), draft); } catch { ranks = Object.create(null); }
    if (state.saved_ballots?.length) {
      ranksLocked = true;
      ranks = Object.create(null);
      for (const ballot of state.saved_ballots) {
        ranks[ballot.case_id] = Object.create(null);
        ballot.ranks.forEach((group, index) => group.forEach(id => { ranks[ballot.case_id][id] = index + 1; }));
      }
    }
    $("activate-preference").disabled = !state.preference_enabled;
    $("activate-preference").checked = state.preference_enabled;
    if (ranksLocked) {
      submitted = !state.preference_enabled || !state.summary?.winner_candidate_id;
      $("submit").textContent = "Apply saved winner";
    }
    $("preference-help").textContent = state.preference_enabled ? "Explicit model choices at launch still take precedence." : "Start review with a preference path to enable this.";
    renderCase();
    if (state.summary) renderSummary(state.summary, "unchanged");
    if (ranksLocked) setMessage("These rankings are finalized. You can apply the saved winner without changing the votes or generating new results.");
  } catch (error) { setMessage(error instanceof Error ? error.message : "Could not load comparison.", true); }
}

$("previous-asset").addEventListener("click", () => { selectedImage--; renderCase(); });
$("next-asset").addEventListener("click", () => { selectedImage++; renderCase(); });
$("close-zoom").addEventListener("click", () => $("zoom").close());
$("submit").addEventListener("click", submit);
$("activate-preference").addEventListener("change", () => { if (state) updateProgress(); });
start();
