# AgentRig test guidance

- Declare every Buck Python test through `agentrig_python_test` with exactly one
  scope: `unit`, `contract`, `integration`, or `eval`.
- Keep unit tests offline. The unit entry point installs a process-wide guard
  before importing tests; do not bypass its network or child-process failures.
- Use injected clocks, IDs, contexts, clients, and scripted outcomes. Avoid wall
  clock, random, network, subprocess, and provider dependencies in unit tests.
- Test both successful values and constructor rejection of impossible states.
- Assert that cancellation, deadlines, failed preflight, and disallowed tools do
  not call implementations or consume scripted outcomes.
- Treat privacy as behavior: verify reprs, events, reports, and normalized
  failures omit raw content, credentials, arguments, and unexpected messages.
- Put SDK packaging and real offline lifecycle checks in `tests/integration`.
- Put authenticated provider checks in `tests/live`, declare `live = True`, and
  require exact `AGENTRIG_RUN_LIVE=1`. Missing authentication must fail, not skip.
- Keep invalid typing examples isolated from valid fixtures. Negative fixtures
  pass only when the expected static error is observed.
- Prefer exact ordered assertions for calls, events, artifacts, grades, and
  lineage so nondeterminism cannot hide behind set comparisons.
