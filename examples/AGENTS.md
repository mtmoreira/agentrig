# AgentRig example guidance

- Organize leaves under `fundamentals`, `agents`, `capabilities`, `evals`, or
  `workflows` according to the concept a reader is learning.
- Keep every leaf self-contained. Do not import another example leaf or create an
  implicit example framework.
- Give each leaf its own domain types, provider-neutral composition,
  deterministic scripted entry point, focused tests, README, and BUCK targets.
- Add `live.py` only when the example has a real provider path. Keep provider SDK
  imports, model selection, sandbox settings, authentication checks, and live
  opt-in in that composition root.
- Build the scripted and live variants with the same provider-neutral workflow
  builder so substitution is demonstrated rather than asserted.
- Use `agentrig.testing` fakes. Promote a generally reusable fake or contract
  suite into that package instead of duplicating it across examples.
- Make live examples least-authority, ephemeral where possible, deadline-bounded,
  and safe to print. Output only public identifiers, normalized statuses,
  aggregate usage, and allowlisted event kinds.
- Typecheck the changed leaf explicitly, run its scripted module and tests, run
  its Buck target, and verify examples remain excluded from the wheel.
- Add or change examples in a dedicated commit separate from framework,
  provider, build, testing-infrastructure, or general documentation changes.
- Use `examples/README.md` as the index and the nearest existing leaf as the
  structural template.
