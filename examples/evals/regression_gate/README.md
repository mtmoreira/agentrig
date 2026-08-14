# Offline eval regression gate

This example turns deterministic evaluation evidence into a release decision:

```text
EvalDataset + EvalTarget + Grader
  -> EvalRunner
  -> EvalReport
  -> EvalBaseline
  -> compare_to_baseline
  -> DeterministicPromotionPolicy
  -> promote | reject | inconclusive
```

The dataset contains two ordered, versioned cases. Each case carries private
input plus declarative expectations. An injected target runs every case in an
isolated sibling `RunContext`, and a deterministic hard grader scores the
target output against the case's required terms.

`EvalReport.from_run` uses its default retention policy, so target outputs,
artifacts, grade explanations, and grade evidence are omitted. Environment
data still passes through the shared redaction policy. `EvalBaseline` removes
all environment and optional payload data before preserving the reviewed
comparison evidence.

The tests demonstrate all three policy outcomes:

- an equivalent candidate with a new target version is promoted;
- a known hard-grade status and score regression is rejected;
- a blocked case makes the comparison inconclusive.

`suite.py` owns the provider-neutral dataset, grader, report, baseline, and
promotion composition. `scripted.py` supplies an offline target and stable
clock/identity generators; it does not access credentials, the network, or a
child process.

Run the deterministic rejection example:

```console
uv run python -m examples.evals.regression_gate.scripted
```

Or with Buck2:

```console
./buck2 run //examples/evals/regression_gate:scripted
./buck2 test //examples/evals/regression_gate:test
```

A production suite can replace the scripted target while preserving the same
dataset identity, grader descriptors, report schema, baseline, comparison, and
promotion policy.
