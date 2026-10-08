# Human-ranked model comparison

This offline example composes `EvalDataset`, `EvalRunner`, `EvalReport`,
`ScriptedTextGenerator`, and `ScriptedGrader` with the portable comparison
contracts. Two candidate configurations execute the same two synthetic briefs.
Original inputs and output files are retained explicitly with relative paths and
SHA-256 hashes. Normal eval reports remain private by default.

The scripted ballot fixture favors the lyrical candidate on one case and ties
both on the other. The equally weighted normalized pairwise scores are 0.75 and
0.25. A private preference file retains the winning whole model set and manifest
identity, without raw votes. Unknown cost remains unknown.

```sh
uv run python -m examples.evals.model_comparison.scripted
uv run python -m unittest examples.evals.model_comparison.test_scripted
uv run python -m mypy examples/evals/model_comparison
./buck2 test //examples/evals/model_comparison:test
```

The executable uses and removes a temporary directory. To inspect retained
synthetic evidence, call `await run_scripted(Path("/explicit/private/directory"))`.
The caller owns that directory and its retention. `manifest.json` is review
evidence, `rankings.json` contains private synthetic votes, and `preference.json`
is a runtime selection record. No provider is invoked, enabled, or recommended by
the example. A real application supplies explicit human ballots and its own
candidate availability checks, per-field overrides, and persisted resume policy.
