/** Pure presentation helpers. Server-side contracts remain authoritative. */
export function ranksToGroups(candidateIds, ranks) {
  if (!ranks || typeof ranks !== "object" || !candidateIds.length || candidateIds.some(id => !Object.hasOwn(ranks, id) || !Number.isInteger(ranks[id]) || ranks[id] < 1 || ranks[id] > candidateIds.length)) return null;
  const groups = new Map();
  for (const id of candidateIds) {
    const rank = ranks[id];
    if (!groups.has(rank)) groups.set(rank, []);
    groups.get(rank).push(id);
  }
  return [...groups.entries()].sort(([a], [b]) => a - b).map(([, ids]) => ids);
}

export function costLabel(cost) {
  if (!cost || cost.status === "unknown" || cost.amount === null || !Number.isFinite(cost.amount)) return "Cost unavailable";
  const amount = new Intl.NumberFormat("en-US", {maximumFractionDigits: 4, minimumFractionDigits: 2}).format(cost.amount);
  return `${cost.currency || ""} ${amount} · ${cost.status === "estimated" ? "estimated" : "reported"}`.trim();
}

export function imageGroups(cells) {
  const labels = [];
  for (const cell of cells) {
    for (const asset of cell.review.assets) {
      if (asset.kind === "image" && !labels.includes(asset.label)) labels.push(asset.label);
    }
  }
  return labels;
}

export function completeCaseCount(cases, candidateIds, ranks) {
  return cases.filter(item => ranksToGroups(candidateIds, ranks[item.case_id] || {}) !== null).length;
}
