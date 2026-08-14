# Deterministic testing and evals

## Start with scripted implementations

Use exports from `agentrig.testing` to exercise the same contracts as production without network access. Choose the scripted implementation matching the boundary under test: agent runtime, generator, search provider, retriever, coding agent, image generator, tool, or grader.

Cover at least:

- the successful typed path and artifact propagation;
- malformed provider output and decoder rejection;
- preflight or constraint failure before an outcome is consumed;
- cancellation and expired deadline before execution;
- disallowed tools or insufficient authority before effects;
- normalized provider failure and script exhaustion;
- retry, denial, blocking, and repair exhaustion where applicable;
- stable safe event ordering without private payloads.

Use the contract suites in `src/agentrig/testing/` when implementing a reusable capability adapter. They verify shared portable semantics beyond one application scenario.

## Add type-level coverage

Type-check a small valid usage when the composition relies on generics, protocol substitutability, or a typed sequence. Add an intentionally incompatible fixture only when compile-time rejection is part of the promised behavior.

## Add evals for behavioral regressions

Use `EvalCase`, `EvalDataset`, an `EvalTarget`, graders, and `EvalRunner` when quality must be compared across multiple stable cases. Keep cases isolated, make grader failures inconclusive rather than passing, and retain private payloads only through an explicit redaction and retention policy.

Compare candidates to a compatible deterministic baseline. Treat missing cases, changed failure kinds, new non-passing grades, and out-of-tolerance resource use according to an explicit policy. See `examples/evals/regression_gate/`.

## Keep validation lanes separate

- Run unit and scripted example tests offline and deterministically.
- Run integration tests for package and SDK boundaries without requiring provider execution where possible.
- Run live tests only with explicit opt-in, bounded authority, deadlines, and stable assertions over typed results and safe lifecycle events.
