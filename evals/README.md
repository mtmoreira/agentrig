# AgentRig repository evals

This directory contains AgentRig's own evaluation assets. Reusable execution,
reporting, and promotion contracts remain under `src/agentrig/evals/`.

- `datasets/` owns immutable, versioned case selections.
- `graders/` owns deterministic or explicitly agentic grading implementations.
- `baselines/` owns reviewed baseline factories or sanitized baseline JSON.
- `suites/` composes targets, datasets, graders, reports, and promotion policy.
- `reports/` is reserved for generated output and is ignored by default.

Offline suites must use deterministic fakes and must not access the network or
child processes. Live suites require the Buck `live = True` policy and
`AGENTRIG_RUN_LIVE=1`; unavailable credentials must produce blocked or
inconclusive evidence rather than a passing result.

Run the deterministic suite with:

```sh
./buck2 test //evals:scripted_agent_eval
```
