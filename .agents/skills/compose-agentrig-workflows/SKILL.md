---
name: compose-agentrig-workflows
description: Compose provider-neutral, typed AgentRig workflows and application adapters with configured agents, capability protocols, sequences, retries, graders, approvals, repair loops, scripted tests, evals, and bounded Codex live variants. Use when building or changing an application workflow that consumes AgentRig, selecting AgentRig abstractions, adapting an AgentRuntime to a capability, or adding deterministic and live workflow coverage; do not use for changing AgentRig framework internals.
---

# Compose AgentRig Workflows

Build application workflows whose types, authority, failure behavior, and test seams remain explicit.

## Start with repository context

1. Read the nearest `AGENTS.md` files in the target application.
2. Inspect the installed AgentRig version and its public exports. Treat source and tests for that version as authoritative.
3. Read the closest example before designing a new composition. Use [example-map.md](references/example-map.md) to choose one.
4. Identify private domain data, durable state transitions, external effects, and the smallest authority each operation needs.

## Compose the workflow

1. Define application-owned input and output types. Enforce domain invariants in deterministic decoders or steps, outside provider schemas and prompts.
2. Select the narrowest AgentRig abstraction using [choose-abstraction.md](references/choose-abstraction.md).
3. Build a provider-neutral workflow core. Keep provider construction, credentials, and live configuration at the composition root.
4. Preflight capability requirements and authority before consuming scripted outcomes, invoking providers, or performing effects.
5. Preserve normalized failures, artifacts, and typed outcomes across each boundary. Do not turn failures into unstructured exceptions or strings.
6. Add deterministic scripted coverage before adding an optional live path. Follow [testing-and-evals.md](references/testing-and-evals.md).
7. Add retries, grading, repair, or approval only where their semantics are explicit. Follow [composition-patterns.md](references/composition-patterns.md).
8. If using Codex, keep it behind `AgentRuntime` or a capability adapter and follow [codex-runtime.md](references/codex-runtime.md).

## Preserve architectural boundaries

- Keep business rules and state commits deterministic. Let models propose typed values; let application code validate and apply them.
- Keep provider-specific flags and SDK types out of portable contracts and workflow modules.
- Ensure an outer contract's authority and limits cover every nested step, tool, and capability.
- Keep private prompts, inputs, outputs, credentials, and raw provider payloads out of events and normalized failures.
- Model effectful work explicitly. Place approval before the effect and never retry a non-repeatable effect automatically.
- Prefer dependency injection and scripted implementations over monkeypatching provider internals.

## Validate and hand off

Run the narrowest type checks and scripted tests first, then the relevant Buck targets and broader repository checks. Keep live tests separately opted in and bounded.

Report the selected abstractions, authority boundary, deterministic test path, live coverage if any, and any behavior that remains unverified.
