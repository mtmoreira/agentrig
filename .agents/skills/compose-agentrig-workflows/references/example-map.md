# Example map

Read the closest example's README, workflow module, scripted fixture, and tests. Reuse its structure and boundaries; do not import example code into an application.

| Goal | Example |
| --- | --- |
| Learn typed steps, child contexts, and safe retry | `examples/fundamentals/typed_sequence/` |
| Substitute a configured agent inside a workflow | `examples/agents/configured_workflow/` |
| Run one strict agent against scripted and Codex runtimes | `examples/agents/codex_structured_agent/` |
| Compose portable search and structured generation with citation checks | `examples/capabilities/sourced_digest/` |
| Back research with Codex web tools while preserving typed citations | `examples/capabilities/web_research/` |
| Authorize a Codex coding task and validate its workspace changes | `examples/capabilities/bounded_coding/` |
| Grade, repair, request approval, and publish | `examples/workflows/review_repair_approve/` |
| Compare candidate behavior to an offline baseline | `examples/evals/regression_gate/` |

Use these examples to answer four design questions before coding:

1. Which module owns domain types and invariants?
2. Which boundary is scripted in tests and live in production?
3. Where are authority, cancellation, deadlines, and retry limits enforced?
4. Which events, failures, artifacts, and eval reports may be retained safely?
