# Replaceable local backends

The local SDK candidate `0.3.1.dev3` adds `LocalBackend` in `capabilities`,
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

`CommandTool` now uses `run_bounded_process` to drain output with fixed capture
memory. Its compatible truncation mode retains at most two configured bounds
while reading and returns one combined bound. `ProcessSpec` defaults to rejecting
overflow, as required by the Lima lifecycle driver. Exact argv, private cwd,
explicit environment, deadlines and cancellation apply without a shell or retries.
An unreaped child anchors process-group cleanup even when its descendants hold
pipes open. Native cancellation waits for worker cleanup. This is not an operating
system sandbox: trusted executables can deliberately detach other processes.

## Durable lifecycle candidate

`JournaledLocalBackends` writes an uncertain intent before calling an adapter.
Exceptions, cancellation and failed completion writes retain uncertainty. Each
operation ID is single-use. Reconciliation only performs a fresh authorized
`INSPECT` for the same backend, version, instance configuration and operation;
it never replays create/start/stop or restores a saved grant.

`LocalBackendJournal` uses a caller-supplied, ancestor-verified private directory,
descriptor-relative bounded JSON, file and directory fsync, an exclusive writer
lock, and local directory identity bindings. It rejects copied state, links,
public files and concurrent writers. Interrupted staging files are preserved and
block new writes; they require explicit repair, not automatic deletion. `inspect`
is a read-only diagnostic snapshot. Execution uses the locked current state.

`LimaLifecycleBackend` implements exact create/start/stop/list command sequences
and journaled recovery. It is separate from the default review-only `LimaBackend`
and is not registered automatically. It requires an injected **trusted application
artifact verifier**, not a user-configurable readiness boolean. Before every
command, that verifier must verify the approved complete local artifact closure,
executable/configuration digests, private roots, absence of global overrides,
owned instance identity and suppression of all implicit acquisition. Without it,
every plan is blocked. No production verifier or approved guest bundle exists yet.

The driver uses a private LIMA_HOME/HOME and exact instance name; never enumerate
another account's runtime or use a default instance. VM observations must match
that name, directory, architecture and VM type. Start requires observed stopped
state; prepare requires local absence; stop requires observed running state.
Unknown results leave uncertainty. Even a running VM result does not qualify
rootless Docker, cgroups, guest-visible staging or simulation readiness.

`WorkspaceStagePlan` snapshots bounded files and binds approval to exact host and
guest roots, operation identity and file hashes. `stage_workspace` writes only a
new private operation directory using exclusive descriptor-relative creation,
fsync and intent/completion receipts. Existing directories, links, traversal,
path conflicts and oversized input are refused; partial files are retained.
`LimaLifecycleBackend.stage` fixes the mapping to its private staging root and
`/mnt/agentrig`, and refuses staging while lifecycle recovery is unresolved.
The returned guest path is a proposed mapping, **not verified guest visibility**.
An approved configuration verifier and real mount/Unix-socket qualification are
still required before a workload can use it.

Offline subprocess and filesystem integration tests exercise the real local
process runner and journal. Lima command tests use an injected scripted process;
they are **not** evidence that Lima, a VM, or a simulator was run.

## Local artifact audit

`BackendArtifactManifest` declares exact IDs, relative paths, SHA-256 hashes,
sizes and executable flags. `audit_backend_artifacts` reads one caller-supplied,
ancestor-verified private bundle directory with a mandatory finite deadline.
It inventories the exact directory tree before opening payloads, rejects extra
files, links, hardlinks and non-private modes, hashes in 64 KiB chunks, and checks
file/directory identity before returning. Nothing is downloaded, extracted,
executed, installed or written. Declare only reviewed public runtime artifacts;
hidden/authentication filenames and private-key/certificate suffixes are refused.

Limits are 128 files, depth 8, 4 GiB per file and 16 GiB total. Files must have
mode 0600 (data) or 0700 (executable), with private directories. A result verifies
the declared bytes at that point in time; it does not prove that the manifest
contains all guest dependencies, that the executable is safe, or that an archive
has safe members. Reverify before later use. The audit is deliberately not a
`LimaArtifactVerifier`: clearing the lifecycle gate still needs semantic config,
complete artifact/dependency closure, owned-instance and no-acquisition checks.

## Backend substitution

Implement the protocol, register explicitly and run `LocalBackendContractSuite`
with a fresh fixture and effect counter. `ScriptedLocalBackend` provides an
independent offline implementation. Synthetic outcomes are not runtime proof.
A verified Docker-compatible Unix backend can share the application's Docker
workload executor. Another transport needs an executor adapter. Applications
retain their engineering state machine and evidence checks.

Configuration reference: [Lima 2.2.0 defaults](https://raw.githubusercontent.com/lima-vm/lima/v2.2.0/templates/default.yaml).
