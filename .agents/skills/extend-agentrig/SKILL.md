---
name: extend-agentrig
description: Extend AgentRig framework internals safely by adding or changing core primitives, public contracts, capability protocols, workflow steps, agent runtimes, eval infrastructure, scripted fakes, contract suites, provider integrations, Buck targets, or packaging. Use for implementation work inside the AgentRig repository, including new framework features and compatibility fixes; do not use merely to compose an application workflow from existing AgentRig APIs.
---

# Extend AgentRig

Implement the smallest complete framework slice while preserving AgentRig's dependency, typing, privacy, authority, and packaging contracts.

## Establish the change boundary

1. Read the root `AGENTS.md` and every scoped `AGENTS.md` governing the files involved.
2. Read the relevant section of `docs/architecture.md`. Consult `docs/development-plan.md` for roadmap context, not as a substitute for current source and tests.
3. Classify ownership with [architecture-boundaries.md](references/architecture-boundaries.md). Keep application domain behavior outside AgentRig.
4. Inspect the owning module, package `__init__.py`, BUCK target, focused unit tests, typing fixtures, scripted implementation, and reusable contract suite before designing the change.
5. Identify whether the change alters a public invariant, serialized vocabulary, dependency direction, provider authority, or package contents.

## Implement one vertical slice

1. Write the invariant and impossible states first. Follow [contract-design.md](references/contract-design.md).
2. Add or change the owning implementation without creating speculative abstractions.
3. Update every required companion surface from [change-surfaces.md](references/change-surfaces.md), including public exports and explicit BUCK dependencies.
4. Enforce capability, authority, cancellation, and deadline checks before implementation calls, scripted consumption, provider construction, or effects.
5. Normalize failures at execution boundaries. Preserve safe failure identity and artifacts while discarding unexpected exception messages and raw provider payloads.
6. Add focused deterministic evidence, then broader validation using [testing-and-build.md](references/testing-and-build.md).
7. For provider work or dependency changes, also follow [provider-integrations.md](references/provider-integrations.md).

## Preserve repository invariants

- Keep `core` provider-free and independent of every other AgentRig package.
- Keep portable contracts typed, provider-neutral, immutable at their boundaries, JSON-safe where serialized, and deterministic in ordering.
- Keep prompts, outputs, evidence content, tool arguments, credentials, repository content, and raw provider errors out of reprs, events, reports, and exceptions.
- Keep retries bounded and effect-aware. Keep grading separate from policy and repair separate from retry.
- Export supported symbols only from the owning package initializer; leave the root `agentrig.__init__` empty.
- Keep optional SDK imports lazy and ensure the base wheel imports without provider extras.
- Ask before adding or replacing a production dependency. Never hand-edit `uv.lock` or `third_party/python/locked_deps.bzl`.

## Hand off a reviewable change

Review the complete diff and confirm that the implementation, tests, typing surface, BUCK graph, exports, docs, and packaging agree. Report validations run, intentionally omitted live lanes, compatibility implications, and remaining risks. Keep framework, testing, provider, example, and general documentation changes in separate commits when each can stand alone.
