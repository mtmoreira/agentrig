# AgentRig examples

Examples are organized by the concept a reader is trying to learn:

- `fundamentals/` — contexts, steps, sequences, retries, and events
- `agents/` — configured agents and autonomous runtimes
- `capabilities/` — generation, coding, image, search, retrieval, and tools
- `evals/` — datasets, graders, reports, and baseline promotion
- `workflows/` — end-to-end control flow built from several AgentRig contracts

Each leaf example is self-contained. It owns its domain types, provider-neutral
composition, deterministic scripted entry point, focused tests, Buck targets,
and README. Leaf examples do not import one another, so they can be copied or
adapted without inheriting an implicit example framework.

The examples are repository documentation and test fixtures; they are not
included in the `agentrig` wheel. Provider-backed variants will use explicit
live entry points and test targets instead of changing the deterministic path.

## Available examples

- [`fundamentals/typed_sequence`](fundamentals/typed_sequence/README.md)
  composes typed function and injected steps, retries only a repeatable
  transient failure, and exposes child-run lifecycle events.
- [`workflows/review_repair_approve`](workflows/review_repair_approve/README.md)
  grades a draft, repairs it within a bound, requests approval, and publishes
  only after approval.
