# AgentRig repository eval guidance

- Keep reusable eval contracts and execution under `src/agentrig/evals`; keep
  AgentRig's datasets, graders, baselines, suites, and generated reports here.
- Version datasets and preserve ordered unique case selections.
- Run cases in isolated sibling contexts. A target failure skips grading for that
  case but must not prevent later cases from running.
- Keep grader execution failure distinct from a failing subject grade. Continue
  other graders unless cancellation or deadline terminates the dataset.
- Default reports and baselines to private: omit raw inputs, outputs, artifacts,
  explanations, and evidence unless retention is explicitly requested and
  redaction is reapplied.
- Treat blocked, cancelled, unavailable-live, and grader-failure evidence as
  inconclusive rather than passing.
- Reject promotion for missing or worse known grades, new nonpassing grades,
  changed known failure kinds, or out-of-tolerance resource metrics.
- Use deterministic scripted targets for offline acceptance. Keep provider-backed
  quality evaluation in separately opted-in, versioned live suites.
- Store generated reports only under the ignored `evals/reports/` area and never
  commit private payloads or credentials.
