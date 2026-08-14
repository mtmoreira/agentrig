# Companion surfaces by change type

Use this matrix to avoid implementing a public type without its exports, tests, fake, or build edge.

| Change | Inspect and usually update |
| --- | --- |
| Core primitive | Owning `core` module, `core/__init__.py`, focused unit test, serialization/redaction tests, `core/BUCK` if dependencies change |
| Capability protocol | Capability module and `__init__.py`, request/result/descriptor tests, valid typing fixture, scripted implementation, reusable contract suite, testing exports, both BUCK targets |
| Agent/runtime contract | Agent module and `__init__.py`, protocol/configuration tests, typing fixture, scripted runtime or client seam, workflow adapters, agent and testing BUCK targets |
| Workflow primitive | Workflow module and `__init__.py`, execution and failure tests, effect/retry behavior, child-context event lineage, typing fixtures, workflow BUCK target |
| Eval primitive | Eval module and `__init__.py`, schema round trip, privacy/retention, aggregate invariants, runner or baseline tests, eval BUCK target, repository eval usage if affected |
| Scripted fake or contract suite | Testing module and `__init__.py`, fake validation and exhaustion tests, shared semantic probes, typing fixture, testing BUCK target |
| Provider integration | Portable contract, injected provider seam, optional package exports, offline unit/integration tests, live target, integration BUCK target, optional dependency packaging |

## Public API review

- Export a supported symbol from its owning package only after its contract and tests are complete.
- Search existing imports before renaming, moving, or changing a signature. State compatibility consequences explicitly.
- Update architecture documentation only when the established design changes. Add an ADR when selecting a consequential provider, dependency, protocol strategy, or irreversible boundary.
- Keep examples in a separate commit. Add one only when it teaches a user-facing composition that tests alone do not explain.
