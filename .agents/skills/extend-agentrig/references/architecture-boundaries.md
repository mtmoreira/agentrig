# Architecture boundaries

Place a concept in the lowest package that can own it without reversing dependencies.

| Package | Owns | May depend on |
| --- | --- | --- |
| `agentrig.core` | Execution context, identity, cancellation, deadlines, failures, outcomes, events, artifacts, grading vocabulary, policies, redaction | Python standard library only |
| `agentrig.capabilities` | Narrow provider-neutral operation contracts and portable descriptors | `core` |
| `agentrig.agents` | Agent contracts, runtime contracts, configured agents, tool permissions and limits | `core`, portable capability/tool contracts |
| `agentrig.workflow` | Typed steps, sequencing, execution, retry, agent/workflow adaptation, grading, repair, approval | Public `core`, `agents`, and capability contracts |
| `agentrig.evals` | Reusable datasets, targets, runners, reports, baselines, comparisons | Public AgentRig APIs |
| `agentrig.testing` | Scripted implementations and reusable conformance suites | Contracts they test, never provider SDKs |
| `agentrig.integrations.<provider>` | SDK translation, provider clients, capability/runtime implementations, provider options | Public contracts plus optional provider dependencies |

## Placement decisions

- Add a new capability only when several applications or implementations share portable semantics. Keep application-specific capabilities in the consuming application.
- Compose existing primitives before adding a framework class. Add a workflow primitive only when it supplies reusable execution semantics, not domain branching.
- Adapt a configured agent to a capability when substitution is valuable. Implement a direct integration when autonomy adds no value.
- Introduce a shared provider client or base only after more than one real integration demonstrates the reuse.
- Put repository-owned eval cases under root `evals/`; keep reusable eval machinery in `agentrig.evals`.

The normative rationale is in `docs/architecture.md`, especially package boundaries, extension strategy, integration layout, security, and build strategy.
