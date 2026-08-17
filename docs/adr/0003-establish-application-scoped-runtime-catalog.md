# ADR 0003: Establish an application-scoped runtime catalog

- **Status:** Accepted for implementation
- **Date:** 2026-08-17
- **Target release:** AgentRig 0.2.0

## Context

AgentRig deliberately began with explicit runtime injection and no provider
registry. That kept provider selection visible while only one real autonomous
runtime existed. StoryWorld now has a concrete need to bind different workflow
agents to different provider runtimes while preserving one provider-neutral
agent contract and standardized integration boundary.

This is demonstrated reuse, but it does not justify a global registry,
automatic provider selection, or provider-specific routing in StoryWorld.
StoryWorld must own which account and runtime binding each agent may use.
AgentRig must own the portable catalog and provider adapter contracts that make
those bindings substitutable.

Authentication has a separate ownership boundary. Applications select and
resolve credentials from their environment or secret manager. AgentRig
integrations consume injected authentication only to construct the provider
client. Neither an AgentRig registration nor a portable runtime request may
contain an API key, login token, or credential payload.

## Decision

AgentRig will add an explicit, application-scoped catalog for autonomous agent
runtimes.

The catalog will be constructed and injected by an application composition
root. It will not be a module singleton, service locator, plugin loader, or
mutable process-global registry. Constructing it must not initialize a provider,
access the network, read credentials, or start a service.

Each registration will contain:

- a stable application binding ID;
- a portable runtime capability descriptor;
- an implementation satisfying `AgentRuntime`.

Registrations will not contain credentials, raw provider configuration, domain
schemas, or StoryWorld routing policy. Duplicate binding IDs fail during
construction. Resolution uses an exact binding ID and validates portable
capability requirements before returning a runtime. Unknown, disabled, or
incompatible bindings fail before provider execution.

Agent-runtime identity will become distinct from coding capability identity.
The implementation will add an `AGENT_RUNTIME` capability kind and update the
Codex runtime descriptor accordingly. Because this changes a released public
descriptor's semantics, the catalog will ship in the next pre-1.0 minor release,
AgentRig 0.2.0.

Provider integrations remain under `agentrig.integrations.<provider>`. They may
accept an injected, late-bound authentication or environment source, but they
must not discover application configuration or select a secret store. The
application owns credential references and resolution. The integration may
pass the resolved value to the provider SDK only through a non-serializable,
non-represented client-construction seam.

Provider selection remains explicit:

```text
application agent route
  -> exact runtime binding ID
  -> portable capability requirement check
  -> registered AgentRuntime
  -> provider integration
```

Automatic ranking, health-based routing, fallback, and load balancing remain
deferred until at least two real runtime integrations provide evidence for
their policy semantics.

Usage policy is split by ownership. AgentRig exposes enforceable per-execution
limits and normalized usage. Applications own project, workflow, account, and
agent budgets. A hard budget may be advertised only when the selected adapter
can enforce it; post-run observation alone is not a hard cap.

## Rejected alternatives

### Put the registry in StoryWorld

Rejected because every AgentRig consumer would otherwise recreate runtime
identity, descriptor validation, duplicate handling, and normalized resolution
semantics. StoryWorld should own route policy, not a parallel AgentRig
abstraction.

### Use a global AgentRig provider registry

Rejected because global mutable state hides dependency construction, makes
tests order-dependent, complicates application isolation, and encourages
import-time provider initialization.

### Store credentials in registrations or provider options

Rejected because these values may be represented, serialized, retained in
request data, or copied into diagnostics. Registrations describe available
runtime implementations; they are not secret containers.

### Route automatically by vendor, cost, or model

Rejected for the initial slice because only Codex has a real runtime adapter
and there is no evidence-backed comparison or fallback policy. Exact
application selection plus capability validation is sufficient.

## Consequences

- One application can bind different agents to different runtime registrations
  without importing provider types into its workflow modules.
- StoryWorld can persist safe binding and credential-reference identities while
  resolving secret values only in its composition layer.
- AgentRig needs public catalog contracts, typing coverage, deterministic tests,
  and a reusable runtime conformance suite.
- The Codex SDK factory needs a safe application-injected authentication seam;
  existing ambient authentication remains a supported composition choice.
- Adding a second provider requires a new AgentRig integration and live suite,
  not a StoryWorld adapter rewrite.
- Catalog construction and resolution remain deterministic and offline.
- AgentRig 0.1.x consumers are not silently exposed to the runtime-descriptor
  semantic change.

## Validation requirements

Implementation must prove:

- exact and deterministic resolution;
- duplicate, unknown, and incompatible bindings fail closed;
- failed resolution never invokes a runtime or resolves a credential;
- registrations and failures omit credential and private configuration values;
- two configured agents can resolve distinct scripted runtimes;
- the scripted and Codex runtimes satisfy the same portable runtime contract;
- an explicitly opted-in Codex live test uses application-supplied
  authentication without retaining it;
- the base AgentRig wheel still imports without provider extras.
