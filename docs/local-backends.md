# Replaceable local backends

The local SDK candidate `0.3.1.dev1` adds `LocalBackend` in `capabilities`,
separate from model-facing `AgentRuntime`. No Python dependency or runtime import
is added. The published `0.3.0` artifact is unchanged.

Applications construct an explicit `LocalBackendCatalog`. Resolution requires
the requested capabilities. Unknown IDs, changed descriptors and incompatible
capabilities fail before invocation; there is no discovery or dynamic import.
Adapters own configuration keys. Plans snapshot inputs, capabilities, version,
operation and effects under one SHA-256 digest. Review exports are detached.

`BackendAuthority` is supplied for one invocation; saved state cannot restore
permission. Adapters repeat preflight before effects. The catalog checks returned
operation, plan digest and action-compatible status. Cancellation and deadlines
remain native signals; unexpected adapter failures become fixed safe codes.
This contract does not itself provide runtime ownership, a journal or retries.

## Current adapters and limits

`ExistingRootlessDockerBackend` inspects one explicit endpoint through an injected
verifier. That verifier owns executable, current-user socket, rootless daemon,
architecture and transport checks. Without a verifier the plan is blocked. It
cannot create a VM, acquire an image or run a workload.

`LimaBackend` produces a **review-only** Lima 2.2.0 VZ preparation proposal for
Darwin arm64. It advertises `local-vm-plan.v1`, not a Docker execution endpoint.
Its fixed blockers cannot be cleared by hashes, flags or fabricated receipts.
`apply` always fails closed. There is no installer or lifecycle driver.

The proposal has no host mounts, inherited templates, provision scripts,
containerd installation, proxy propagation, ambient SSH key loading or automatic
application TCP/UDP forwarding. It is deliberately incomplete and must not be
passed to Lima as executable configuration. Still required:

- Review complete guest image, rootless engine, package and license inventories.
- Verify archive bytes/members, executable pins and immutable guest bytes.
- Suppress default guest dependency installation and verify cgroup delegation.
- Implement private Unix forwarding and owned staging with explicit guest paths.
- Implement bounded process handling, durable intent and exact-instance recovery;
  never automatically replay uncertain creation or startup.
- Validate the real VM, isolation, resources and recovery with explicit authority.

The existing `CommandTool` truncates captured output after collection; that is
not a streaming memory bound suitable for VM lifecycle logs. This planning slice
does not change or qualify its process behavior.

## Backend substitution

Implement the protocol, register explicitly and run `LocalBackendContractSuite`
with a fresh fixture and effect counter. `ScriptedLocalBackend` provides an
independent offline implementation. Synthetic outcomes are not runtime proof.
A verified Docker-compatible Unix backend can share the application's Docker
workload executor. Another transport needs an executor adapter. Applications
retain their engineering state machine and evidence checks.

Configuration reference: [Lima 2.2.0 defaults](https://raw.githubusercontent.com/lima-vm/lima/v2.2.0/templates/default.yaml).
